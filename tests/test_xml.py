"""Pruebas de construccion y validacion XML. Sin dependencias de red ni de MathCAT."""
from lxml import etree

from app.models import (
    AudioOutput, BrailleOutput, ConversionResult, EnrichmentResult, OcrResult, RuleOrigin, RuleOutcome,
    RuleResult, RuleSeverity, SemanticOutput, SemanticResult, SpeechOutput, StageIssue, StageStatus,
)
from app.xml_builder import build_xml_document
from app.xml_validator import MATHML_NS, NSMAP, validate_xml

MATHML = f'<math xmlns="{MATHML_NS}"><mi>x</mi><mo>=</mo><mn>0</mn></math>'
OK, FAILED, SKIPPED = StageStatus.OK, StageStatus.FAILED, StageStatus.SKIPPED


def make_ocr(latex="x = 0", path="C:/Users/privado/img/f.png"):
    return OcrResult(image_ref=path, status=OK, latex_raw=latex)


def make_conv(mathml=MATHML):
    return ConversionResult(latex="x = 0", status=OK, mathml=mathml)


def make_sem():
    out = SemanticOutput.model_validate({
        "vocabulary_version": "1.0.0", "root_id": "n0", "review_required": False,
        "nodes": [
            {"id": "n0", "node_type": "structure", "intent": "equation", "evaluation_order": 3,
             "operands": [{"role": "statement", "node_id": "n1"}]},
            {"id": "n1", "node_type": "relation", "intent": "equality", "symbol": "=", "evaluation_order": 2,
             "operands": [{"role": "left", "node_id": "n2"}, {"role": "right", "node_id": "n3"}]},
            {"id": "n2", "node_type": "element", "intent": "variable", "symbol": "x", "evaluation_order": 0},
            {"id": "n3", "node_type": "element", "intent": "number", "symbol": "0", "evaluation_order": 1},
        ]})
    return SemanticResult(status=OK, output=out, provider="scripted", model="fake-1", attempts=1,
                          input_latex_sha256="a" * 64, input_mathml_sha256="b" * 64)


def make_rules():
    return [RuleResult(rule_id="R-001", rule_version="1.0", origin=RuleOrigin.PROJECT_HEURISTIC,
                       outcome=RuleOutcome.PASS, severity=RuleSeverity.INFO, message="ok")]


def make_braille(**kw):
    args = dict(status=OK, braille_code="cmu", mathcat_code="CMU", language="es", text_unicode="⠼⠁⠃", mathcat_version="fake-0")
    args.update(kw)
    return BrailleOutput(**args)


def make_speech(**kw):
    args = dict(status=OK, language="es", style="ClearSpeak", verbosity="Medium", text="x igual a cero")
    args.update(kw)
    return SpeechOutput(**args)


def make_audio(**kw):
    args = dict(status=OK, provider="fake-tts", language="es", audio_format="wav", file_name="lectura.wav",
                file_path="C:/Users/privado/a/lectura.wav", size_bytes=1000)
    args.update(kw)
    return AudioOutput(**args)


def full_doc(**kw):
    args = dict(run_id="run-1", ocr=make_ocr(), conversion=make_conv(), semantic=make_sem(),
                rules=make_rules(), rules_catalog_version="0.2.0")
    args.update(kw)
    return build_xml_document(**args)


def errcodes(result):
    return [e.code for e in result.errors]


def overall(r):
    return etree.fromstring(r.xml).xpath("string(t:validation/@overall)", namespaces=NSMAP)


# ------------------------------------------------------------- documento base
def test_documento_completo_valido():
    r = full_doc()
    assert r.status is OK, r.errors
    assert r.well_formed and r.schema_valid and overall(r) == "VALIDATED"
    root = etree.fromstring(r.xml)
    assert len(root.xpath("//m:math", namespaces={"m": MATHML_NS})) == 1
    node = root.xpath("t:semantic/t:nodes/t:node[@id='n1']", namespaces=NSMAP)[0]
    assert (node.get("intent"), node.get("symbol"), node.get("evaluationOrder")) == ("equality", "=", "2")


def test_sin_agente_semantico_no_aparenta_validado():
    r = full_doc(semantic=None)
    assert r.status is OK, r.errors
    assert overall(r) == "NOT_VALIDATED"
    reasons = etree.fromstring(r.xml).xpath("t:validation/t:reason/text()", namespaces=NSMAP)
    assert "semantic_agent: SKIPPED" in reasons


def test_conversion_fallida_no_incluye_mathml_y_lista_errores():
    conv = ConversionResult(latex="\\frac{1}", status=FAILED, errors=[StageIssue(stage="converter", code="X", message="falla")])
    r = full_doc(conversion=conv, semantic=None)
    assert r.status is OK, r.errors
    root = etree.fromstring(r.xml)
    assert not root.xpath("t:mathml", namespaces=NSMAP) and root.xpath("t:issues/t:issue[@code='X']", namespaces=NSMAP)
    assert overall(r) == "NOT_VALIDATED"


def test_caracteres_especiales_ida_y_vuelta_e_ilegales():
    latex = 'a<b & c>"d\' \\sum'
    r = full_doc(ocr=make_ocr(latex))
    assert r.status is OK and etree.fromstring(r.xml).xpath("string(t:ocr/t:latexRaw)", namespaces=NSMAP) == latex
    assert b"a&lt;b &amp; c&gt;" in r.xml
    r2 = full_doc(ocr=make_ocr("x\x00\x08y"))
    assert etree.fromstring(r2.xml).xpath("string(t:ocr/t:latexRaw)", namespaces=NSMAP) == "x\ufffd\ufffdy"


def test_solo_se_guarda_nombre_de_archivo():
    r = full_doc(audio=make_audio(), speech=make_speech())
    assert b"privado" not in r.xml and b"C:/Users" not in r.xml
    root = etree.fromstring(r.xml)
    assert root.xpath("string(t:input/t:imageRef)", namespaces=NSMAP) == "f.png"
    assert root.xpath("string(t:audio/t:fileRef)", namespaces=NSMAP) == "lectura.wav"


# --------------------------------------------- braille, lectura y audio (estados independientes)
def test_braille_lectura_y_audio_se_reflejan_con_estados_independientes():
    r = full_doc(braille=make_braille(), speech=make_speech(), audio=make_audio(), required_stages={"braille", "speech"})
    assert r.status is OK, r.errors
    assert overall(r) == "VALIDATED"
    root = etree.fromstring(r.xml)
    assert root.xpath("string(t:braille/t:unicode)", namespaces=NSMAP) == "⠼⠁⠃"
    assert root.xpath("string(t:braille/@code)", namespaces=NSMAP) == "cmu"
    assert root.xpath("string(t:speech/@language)", namespaces=NSMAP) == "es"
    assert root.xpath("string(t:speech/t:text)", namespaces=NSMAP) == "x igual a cero"
    st = {s.get("name"): s.get("status") for s in root.xpath("t:stages/t:stage", namespaces=NSMAP)}
    assert (st["braille"], st["speech"], st["audio"]) == ("OK", "OK", "OK")
    v = root.xpath("t:validation", namespaces=NSMAP)[0]
    assert (v.get("brailleGenerated"), v.get("speechGenerated"), v.get("audioGenerated")) == ("true", "true", "true")


def test_audio_omitido_por_falta_de_tts_no_invalida_un_braille_correcto():
    audio = AudioOutput(status=SKIPPED, warnings=[StageIssue(stage="audio", code="NO_TTS_PROVIDER", message="sin TTS")])
    r = full_doc(braille=make_braille(), speech=make_speech(), audio=audio, required_stages={"braille", "speech"})
    assert r.status is OK, r.errors
    assert overall(r) == "VALIDATED"
    root = etree.fromstring(r.xml)
    assert not root.xpath("t:audio", namespaces=NSMAP)                       # no se afirma ningun archivo
    assert root.xpath("string(t:validation/@audioGenerated)", namespaces=NSMAP) == "false"
    assert root.xpath("t:issues/t:issue[@code='NO_TTS_PROVIDER']", namespaces=NSMAP)


def test_braille_obligatorio_fallido_impide_validar():
    bad = BrailleOutput(status=FAILED, errors=[StageIssue(stage="braille", code="MATHCAT_CALL_FAILED", message="x")])
    r = full_doc(braille=bad, required_stages={"braille"})
    assert r.status is OK, r.errors
    assert overall(r) == "NOT_VALIDATED"
    assert "braille: FAILED" in etree.fromstring(r.xml).xpath("t:validation/t:reason/text()", namespaces=NSMAP)


def test_cualquier_etapa_fallida_impide_validar_aunque_sea_opcional():
    failed_audio = AudioOutput(status=FAILED, errors=[StageIssue(stage="audio", code="TTS_FAILED", message="x")])
    r = full_doc(braille=make_braille(), speech=make_speech(), audio=failed_audio, required_stages={"braille", "speech"})
    assert overall(r) == "NOT_VALIDATED"


def test_enriquecimiento_se_refleja_con_mathml_en_su_espacio_de_nombres():
    enriched = f'<math xmlns="{MATHML_NS}"><msup intent="power($b,$e)"><mi arg="b">x</mi><mn arg="e">2</mn></msup></math>'
    enr = EnrichmentResult(status=OK, enabled=True, mathml=enriched, applied=["n0:power->power"], mathcat_checked=True)
    r = full_doc(enrichment=enr)
    assert r.status is OK, r.errors
    root = etree.fromstring(r.xml)
    assert root.xpath("string(t:enrichment/@mathCatChecked)", namespaces=NSMAP) == "true"
    assert root.xpath("t:enrichment/t:enrichedMathml/m:math/m:msup[@intent]", namespaces={**NSMAP, "m": MATHML_NS})


# --------------------------------------------------- el XSD rechaza lo estructural
def test_xsd_rechaza_ids_duplicados_referencias_y_raiz_inexistentes():
    xml = full_doc().xml
    for bad in (xml.replace(b'<node id="n3"', b'<node id="n2"', 1),
                xml.replace(b'nodeId="n3"', b'nodeId="zzz"', 1),
                xml.replace(b'rootId="n0"', b'rootId="zzz"', 1)):
        v = validate_xml(bad)
        assert v.status is FAILED and "XSD_INVALID" in errcodes(v)


def test_xsd_rechaza_valores_fuera_de_enumeracion_y_atributos_ausentes():
    xml = full_doc().xml
    assert "XSD_INVALID" in errcodes(validate_xml(xml.replace(b'outcome="PASS"', b'outcome="MAYBE"')))
    assert "XSD_INVALID" in errcodes(validate_xml(xml.replace(b' runId="run-1"', b"")))
    assert "XSD_INVALID" in errcodes(validate_xml(xml.replace(b'nodeType="relation"', b'nodeType="otro"')))


def test_xml_mal_formado_raiz_incorrecta_y_xxe():
    assert "XML_NOT_WELL_FORMED" in errcodes(validate_xml(b"<result><a></result>"))
    v = validate_xml(b"<otra/>")
    assert v.well_formed and not v.schema_valid
    evil = b'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY x SYSTEM "file:///etc/passwd">]><r>&x;</r>'
    v = validate_xml(evil)
    assert "DTD_NOT_ALLOWED" in errcodes(v) and all("root:" not in e.message for e in v.errors)


# ------------------------------------------- coherencia que el XSD no puede expresar
def test_incoherencias_entre_estados_y_facetas():
    ok = full_doc(braille=make_braille(), speech=make_speech(), required_stages={"braille", "speech"}).xml
    forged = ok.replace(b'overall="VALIDATED"', b'overall="VALIDATED"').replace(b'name="braille" status="OK"', b'name="braille" status="FAILED"')
    v = validate_xml(forged)
    assert v.schema_valid and "OVERALL_INCONSISTENT" in errcodes(v)
    v = validate_xml(ok.replace(b'brailleGenerated="true"', b'brailleGenerated="false"'))
    assert "FACET_INCONSISTENT" in errcodes(v)
    v = validate_xml(ok.replace(b'semanticReviewRequired="false"', b'semanticReviewRequired="true"'))
    assert "FACET_INCONSISTENT" in errcodes(v)


def test_etapa_ok_sin_contenido_o_contenido_sin_etapa_ok():
    root = etree.fromstring(full_doc(braille=make_braille(), speech=make_speech()).xml)
    root.remove(root.xpath("t:braille", namespaces=NSMAP)[0])
    assert "STAGE_CONTENT_MISSING" in errcodes(validate_xml(etree.tostring(root, xml_declaration=True, encoding="UTF-8")))
    root = etree.fromstring(full_doc(braille=make_braille(), speech=make_speech()).xml)
    root.xpath("t:stages/t:stage[@name='braille']", namespaces=NSMAP)[0].set("status", "SKIPPED")
    root.xpath("t:validation", namespaces=NSMAP)[0].set("brailleGenerated", "false")
    assert "STAGE_CONTENT_UNEXPECTED" in errcodes(validate_xml(etree.tostring(root, xml_declaration=True, encoding="UTF-8")))


def test_las_etapas_nucleo_no_pueden_declararse_opcionales():
    root = etree.fromstring(full_doc().xml)
    root.xpath("t:stages/t:stage[@name='ocr']", namespaces=NSMAP)[0].set("required", "false")
    assert "CORE_STAGE_NOT_REQUIRED" in errcodes(validate_xml(etree.tostring(root, xml_declaration=True, encoding="UTF-8")))


def test_ciclo_e_intent_desconocido_en_el_xml():
    root = etree.fromstring(full_doc().xml)
    leaf = root.xpath("t:semantic/t:nodes/t:node[@id='n2']", namespaces=NSMAP)[0]
    op = etree.SubElement(leaf, "{%s}operand" % NSMAP["t"]); op.set("role", "operand"); op.set("nodeId", "n0")
    v = validate_xml(etree.tostring(root, xml_declaration=True, encoding="UTF-8"))
    assert v.schema_valid and "CYCLE_IN_XML" in errcodes(v)       # el XSD no detecta ciclos; la coherencia si
    v = validate_xml(full_doc().xml.replace(b'intent="equality"', b'intent="igualdad"'))
    assert v.schema_valid and "UNKNOWN_INTENT_IN_XML" in errcodes(v)


def test_nodo_ambiguo_con_intent_se_rechaza_en_el_xml():
    xml = full_doc().xml.replace(b'ambiguous="false"', b'ambiguous="true"', 1)
    assert "AMBIGUOUS_HAS_INTENT_IN_XML" in errcodes(validate_xml(xml))


def test_la_construccion_devuelve_error_si_el_documento_es_invalido():
    sem = make_sem()
    sem.output.nodes[3].id = "n2"                                  # id duplicado
    r = full_doc(semantic=sem)
    assert r.status is FAILED and "XSD_INVALID" in errcodes(r)
