"""Pruebas de construccion y validacion XML. Sin dependencias de pytest ni de red."""
from lxml import etree

from app.models import (
    ConversionResult, OcrResult, RuleOrigin, RuleOutcome, RuleResult, RuleSeverity,
    SemanticNode, SemanticOutput, SemanticRelation, SemanticResult, StageIssue, StageStatus,
)
from app.xml_builder import build_xml_document
from app.xml_validator import MATHML_NS, NSMAP, validate_xml

MATHML = f'<math xmlns="{MATHML_NS}"><mi>x</mi><mo>=</mo><mn>0</mn></math>'


def make_ocr(latex="x = 0", path="C:/Users/privado/img/f.png"):
    return OcrResult(image_ref=path, status=StageStatus.OK, latex_raw=latex)


def make_conv(mathml=MATHML):
    return ConversionResult(latex="x = 0", status=StageStatus.OK, mathml=mathml)


def make_sem():
    out = SemanticOutput(
        expression_type="equation",
        root=SemanticNode(id="n0", kind="equation", symbol="=", children=[
            SemanticNode(id="n1", kind="variable", symbol="x"),
            SemanticNode(id="n2", kind="number", symbol="0"),
        ]),
        relations=[SemanticRelation(type="equality", left="n1", right="n2")],
        requires_review=False,
    )
    return SemanticResult(status=StageStatus.OK, output=out, provider="scripted", model="fake-1", attempts=1)


def make_rules():
    return [RuleResult(rule_id="R-001", rule_version="1.0", origin=RuleOrigin.PROJECT_HEURISTIC,
                       outcome=RuleOutcome.PASS, severity=RuleSeverity.INFO, message="ok")]


def full_doc(**kw):
    args = dict(run_id="run-1", ocr=make_ocr(), conversion=make_conv(), semantic=make_sem(),
                rules=make_rules(), rules_catalog_version="0.1.0")
    args.update(kw)
    return build_xml_document(**args)


def errcodes(result):
    return [e.code for e in result.errors]


def test_documento_completo_valido():
    r = full_doc()
    assert r.status is StageStatus.OK, r.errors
    assert r.well_formed and r.schema_valid
    root = etree.fromstring(r.xml)
    assert root.xpath("string(t:validation/@overall)", namespaces=NSMAP) == "VALIDATED"
    assert len(root.xpath("//m:math", namespaces={"m": MATHML_NS})) == 1


def test_sin_agente_semantico_no_aparenta_validado():
    r = full_doc(semantic=None)
    assert r.status is StageStatus.OK, r.errors  # estructuralmente valido
    root = etree.fromstring(r.xml)
    assert root.xpath("string(t:validation/@overall)", namespaces=NSMAP) == "NOT_VALIDATED"
    reasons = root.xpath("t:validation/t:reason/text()", namespaces=NSMAP)
    assert "semantic_agent: SKIPPED" in reasons


def test_conversion_fallida_no_incluye_mathml_y_lista_errores():
    conv = ConversionResult(latex="\\frac{1}", status=StageStatus.FAILED,
                            errors=[StageIssue(stage="converter", code="X", message="falla")])
    r = full_doc(conversion=conv, semantic=None)
    assert r.status is StageStatus.OK, r.errors
    root = etree.fromstring(r.xml)
    assert not root.xpath("t:mathml", namespaces=NSMAP)
    assert root.xpath("t:issues/t:issue[@code='X']", namespaces=NSMAP)
    assert root.xpath("string(t:validation/@overall)", namespaces=NSMAP) == "NOT_VALIDATED"


def test_caracteres_especiales_se_escapan_y_ida_y_vuelta():
    latex = 'a<b & c>"d\' \\sum'
    r = full_doc(ocr=make_ocr(latex))
    assert r.status is StageStatus.OK, r.errors
    root = etree.fromstring(r.xml)
    assert root.xpath("string(t:ocr/t:latexRaw)", namespaces=NSMAP) == latex
    assert b"a&lt;b &amp; c&gt;" in r.xml


def test_caracteres_ilegales_se_sustituyen():
    r = full_doc(ocr=make_ocr("x\x00\x08y"))
    assert r.status is StageStatus.OK, r.errors
    root = etree.fromstring(r.xml)
    assert root.xpath("string(t:ocr/t:latexRaw)", namespaces=NSMAP) == "x\ufffd\ufffdy"


def test_solo_se_guarda_nombre_de_archivo():
    r = full_doc()
    assert b"privado" not in r.xml and b"C:/Users" not in r.xml
    root = etree.fromstring(r.xml)
    assert root.xpath("string(t:input/t:imageRef)", namespaces=NSMAP) == "f.png"


def test_xsd_rechaza_resultado_de_regla_desconocido():
    r = full_doc()
    bad = r.xml.replace(b'outcome="PASS"', b'outcome="MAYBE"')
    v = validate_xml(bad)
    assert v.status is StageStatus.FAILED and "XSD_INVALID" in errcodes(v)


def test_xsd_rechaza_ids_de_nodo_duplicados():
    bad = full_doc().xml.replace(b'id="n2"', b'id="n1"')
    v = validate_xml(bad)
    assert v.status is StageStatus.FAILED and "XSD_INVALID" in errcodes(v)


def test_xsd_rechaza_relacion_con_id_inexistente():
    bad = full_doc().xml.replace(b'right="n2"', b'right="zzz"')
    v = validate_xml(bad)
    assert v.status is StageStatus.FAILED and "XSD_INVALID" in errcodes(v)


def test_xsd_rechaza_falta_de_run_id():
    bad = full_doc().xml.replace(b' runId="run-1"', b"")
    v = validate_xml(bad)
    assert "XSD_INVALID" in errcodes(v)


def test_xml_mal_formado():
    v = validate_xml(b"<result><a></result>")
    assert not v.well_formed and "XML_NOT_WELL_FORMED" in errcodes(v)


def test_raiz_incorrecta():
    v = validate_xml(b"<otra/>")
    assert v.well_formed and not v.schema_valid


def test_xxe_rechazado_y_no_se_lee_el_archivo():
    evil = (b'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY x SYSTEM "file:///etc/passwd">]>'
            b"<r>&x;</r>")
    v = validate_xml(evil)
    assert v.status is StageStatus.FAILED and "DTD_NOT_ALLOWED" in errcodes(v)
    assert all("root:" not in e.message for e in v.errors)


def test_incoherencia_validated_con_etapa_fallida():
    ok = full_doc(semantic=None)  # NOT_VALIDATED por semantic SKIPPED
    forged = ok.xml.replace(b'overall="NOT_VALIDATED"', b'overall="VALIDATED"')
    v = validate_xml(forged)
    assert v.schema_valid and v.status is StageStatus.FAILED
    assert "OVERALL_INCONSISTENT" in errcodes(v)


def test_etapa_ok_sin_contenido():
    doc = full_doc().xml
    root = etree.fromstring(doc)
    sem = root.xpath("t:semantic", namespaces=NSMAP)[0]
    root.remove(sem)
    v = validate_xml(etree.tostring(root, xml_declaration=True, encoding="UTF-8"))
    assert "STAGE_CONTENT_MISSING" in errcodes(v)


def test_la_construccion_devuelve_error_si_el_documento_es_invalido():
    # ids duplicados en el resultado semantico -> el XSD lo detecta al validar el propio documento
    sem = make_sem()
    sem.output.root.children[1].id = "n1"
    r = full_doc(semantic=sem)
    assert r.status is StageStatus.FAILED and "XSD_INVALID" in errcodes(r)
