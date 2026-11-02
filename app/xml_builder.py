"""Construccion del documento XML de resultados con lxml.etree.

* Los textos se insertan como nodos de texto/atributos: lxml escapa ``< > & "`` automaticamente.
  Los caracteres prohibidos por XML 1.0 se sustituyen por U+FFFD.
* Solo se guarda el NOMBRE de los archivos (imagen, audio), nunca rutas personales.
* ``validation/@overall`` es VALIDATED unicamente si ninguna etapa fallo, todas las etapas
  OBLIGATORIAS estan OK y no hay errores registrados; en otro caso es NOT_VALIDATED y se listan los
  motivos. Una etapa opcional omitida (p. ej. audio sin proveedor TTS) no impide VALIDATED.
* No se calcula ninguna puntuacion agregada.
* Antes de devolver, el documento se valida con xml_validator.
* Un XML valido NO demuestra que la transcripcion braille o la interpretacion sean correctas.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from lxml import etree

from app.models import (
    AudioOutput,
    BrailleOutput,
    ConversionResult,
    EnrichmentResult,
    OcrResult,
    RuleResult,
    SemanticResult,
    SpeechOutput,
    StageIssue,
    StageStatus,
    XmlResult,
    utc_now,
)
from app.xml_validator import MATHML_NS, RESULT_NS, make_safe_parser, validate_xml

STAGE = "xml_builder"
SCHEMA_VERSION = "1.0"
NSMAP = {None: RESULT_NS, "m": MATHML_NS}
CORE_STAGES = ("ocr", "converter", "semantic_agent", "rules")
ALL_STAGES = CORE_STAGES + ("enrichment", "braille", "speech", "audio")

_ILLEGAL = re.compile("[^\u0009\u000A\u000D\u0020-\uD7FF\uE000-\uFFFD\U00010000-\U0010FFFF]")


def _clean(value: object) -> str:
    return _ILLEGAL.sub("\ufffd", str(value))


def _q(tag: str) -> str:
    return f"{{{RESULT_NS}}}{tag}"


def _b(flag: bool) -> str:
    return "true" if flag else "false"


def _sub(parent: etree._Element, tag: str, text: Optional[object] = None, **attrs: Optional[object]) -> etree._Element:
    el = etree.SubElement(parent, _q(tag))
    if text is not None:
        el.text = _clean(text)
    for key, value in attrs.items():
        if value is not None:
            el.set(key, _clean(value.value if hasattr(value, "value") else value))
    return el


def _status(result: object) -> StageStatus:
    return result.status if result is not None else StageStatus.SKIPPED


def _parse_mathml(mathml: str) -> etree._Element:
    return etree.fromstring(mathml.encode("utf-8"), parser=make_safe_parser())


def build_xml_document(
    *,
    run_id: str,
    image_ref: Optional[str] = None,
    ocr: Optional[OcrResult] = None,
    conversion: Optional[ConversionResult] = None,
    semantic: Optional[SemanticResult] = None,
    rules: Optional[list[RuleResult]] = None,
    rules_catalog_version: Optional[str] = None,
    rules_status: Optional[StageStatus] = None,
    enrichment: Optional[EnrichmentResult] = None,
    braille: Optional[BrailleOutput] = None,
    speech: Optional[SpeechOutput] = None,
    audio: Optional[AudioOutput] = None,
    required_stages: Optional[Iterable[str]] = None,
    created_at: Optional[datetime] = None,
    validate: bool = True,
) -> XmlResult:
    """Construye (y por defecto valida) el documento XML de resultados."""
    rules = rules or []
    if rules_status is None:
        rules_status = StageStatus.OK if rules else StageStatus.SKIPPED
    required = set(CORE_STAGES) | set(required_stages or ())
    statuses = {
        "ocr": _status(ocr), "converter": _status(conversion), "semantic_agent": _status(semantic),
        "rules": rules_status, "enrichment": _status(enrichment), "braille": _status(braille),
        "speech": _status(speech), "audio": _status(audio),
    }

    root = etree.Element(_q("result"), nsmap=NSMAP, schemaVersion=SCHEMA_VERSION, runId=_clean(run_id))

    meta = _sub(root, "metadata")
    _sub(meta, "createdAt", (created_at or utc_now()).isoformat())
    tools = _sub(meta, "tools")
    if ocr is not None:
        _sub(tools, "tool", role="ocr", name=ocr.engine, version=ocr.engine_version)
    if conversion is not None:
        _sub(tools, "tool", role="converter", name=conversion.converter, version=conversion.converter_version)
    if semantic is not None and semantic.provider:
        _sub(tools, "tool", role="semantic_agent", name=semantic.provider, version=semantic.model)
    mathcat_version = next((o.mathcat_version for o in (braille, speech) if o is not None and o.mathcat_version), None)
    if braille is not None or speech is not None:
        _sub(tools, "tool", role="mathcat", name="MathCAT", version=mathcat_version)
    if audio is not None and audio.provider:
        _sub(tools, "tool", role="tts", name=audio.provider)
    if rules_catalog_version:
        _sub(meta, "rulesCatalogVersion", rules_catalog_version)

    ref = image_ref or (ocr.image_ref if ocr is not None else "")
    _sub(_sub(root, "input"), "imageRef", Path(ref).name if ref else "")

    stages_el = _sub(root, "stages")
    for name in ALL_STAGES:
        _sub(stages_el, "stage", name=name, status=statuses[name], required=_b(name in required))

    issues: list[tuple[str, StageIssue]] = []
    for result in (ocr, conversion, semantic, enrichment, braille, speech, audio):
        if result is not None:
            issues += [("error", i) for i in result.errors] + [("warning", i) for i in result.warnings]

    if ocr is not None and (ocr.latex_raw is not None or ocr.latex_reviewed is not None):
        ocr_el = _sub(root, "ocr", attempt=ocr.attempt, reviewedByHuman=_b(ocr.reviewed_by_human))
        if ocr.latex_raw is not None:
            _sub(ocr_el, "latexRaw", ocr.latex_raw)
        if ocr.latex_reviewed is not None:
            _sub(ocr_el, "latexReviewed", ocr.latex_reviewed)

    if conversion is not None and conversion.mathml:
        try:
            _sub(root, "mathml", kind=conversion.mathml_kind).append(_parse_mathml(conversion.mathml))
        except etree.XMLSyntaxError as exc:
            issues.append(("error", StageIssue(stage=STAGE, code="MATHML_UNPARSEABLE", message=str(exc))))

    if enrichment is not None:
        enr = _sub(root, "enrichment", mathCatChecked=_b(enrichment.mathcat_checked))
        for item in enrichment.applied:
            _sub(enr, "appliedIntent", item)
        if enrichment.status is StageStatus.OK and enrichment.mathml:
            try:
                _sub(enr, "enrichedMathml").append(_parse_mathml(enrichment.mathml))
            except etree.XMLSyntaxError as exc:
                issues.append(("error", StageIssue(stage=STAGE, code="ENRICHED_MATHML_UNPARSEABLE", message=str(exc))))

    semantic_review = False
    if semantic is not None and semantic.output is not None:
        out = semantic.output
        semantic_review = out.review_required
        sem = _sub(root, "semantic", vocabularyVersion=out.vocabulary_version, rootId=out.root_id,
                   reviewRequired=_b(out.review_required), provider=semantic.provider, model=semantic.model,
                   inputLatexSha256=semantic.input_latex_sha256, inputMathmlSha256=semantic.input_mathml_sha256)
        nodes_el = _sub(sem, "nodes")
        for n in out.nodes:
            node_el = _sub(nodes_el, "node", id=n.id, nodeType=n.node_type, intent=n.intent, symbol=n.symbol,
                           implicit=_b(n.implicit), evaluationOrder=n.evaluation_order, evidence=n.evidence,
                           ambiguous=_b(n.ambiguous), reviewRequired=_b(n.review_required))
            for op in n.operands:
                _sub(node_el, "operand", role=op.role, nodeId=op.node_id)
            for cand in n.candidate_intents:
                _sub(node_el, "candidateIntent", cand)
            for key, value in n.properties.items():
                _sub(node_el, "property", name=key, value=value)
            if n.justification:
                _sub(node_el, "justification", n.justification)
        links_el = _sub(sem, "links")
        for link in out.links:
            _sub(links_el, "link", type=link.type, source=link.source, target=link.target, evidence=link.evidence)
        assumptions = _sub(sem, "assumptions")
        for text in out.assumptions:
            _sub(assumptions, "assumption", text)
        ambiguities = _sub(sem, "ambiguities")
        for text in out.ambiguities:
            _sub(ambiguities, "ambiguity", text)

    if braille is not None and braille.text is not None:
        b_el = _sub(root, "braille", code=braille.braille_code or "", mathcatCode=braille.mathcat_code,
                    language=braille.language, reviewRequired=_b(braille.review_required),
                    reviewedByHuman=_b(braille.reviewed_by_human), mathcatVersion=braille.mathcat_version)
        if braille.text_unicode is not None:
            _sub(b_el, "unicode", braille.text_unicode)
        if braille.text_unicode_reviewed is not None:
            _sub(b_el, "unicodeReviewed", braille.text_unicode_reviewed)

    if speech is not None and speech.text:
        s_el = _sub(root, "speech", language=speech.language or "", style=speech.style, verbosity=speech.verbosity,
                    reviewRequired=_b(speech.review_required), mathcatVersion=speech.mathcat_version)
        _sub(s_el, "text", speech.text)

    if audio is not None and audio.status is StageStatus.OK and audio.file_name:
        a_el = _sub(root, "audio", provider=audio.provider or "", language=audio.language,
                    format=audio.audio_format or "", sizeBytes=audio.size_bytes or 0, voice=audio.voice,
                    reviewRequired=_b(audio.review_required))
        _sub(a_el, "fileRef", Path(audio.file_name).name)

    rules_el = _sub(root, "rules", catalogVersion=None)
    for rule in rules:
        r_el = _sub(rules_el, "rule", id=rule.rule_id, version=rule.rule_version, origin=rule.origin,
                    outcome=rule.outcome, severity=rule.severity)
        _sub(r_el, "message", rule.message)
        for evidence in rule.evidence_refs:
            _sub(r_el, "evidenceRef", evidence)

    reasons = [f"{name}: {st.value}" for name, st in statuses.items()
               if st is StageStatus.FAILED or (name in required and st is not StageStatus.OK)]
    if any(level == "error" for level, _ in issues):
        reasons.append("hay errores registrados en las etapas")
    validation_el = _sub(root, "validation", overall="NOT_VALIDATED" if reasons else "VALIDATED",
                         semanticReviewRequired=_b(semantic_review),
                         brailleGenerated=_b(statuses["braille"] is StageStatus.OK),
                         speechGenerated=_b(statuses["speech"] is StageStatus.OK),
                         audioGenerated=_b(statuses["audio"] is StageStatus.OK))
    for reason in reasons:
        _sub(validation_el, "reason", reason)

    issues_el = _sub(root, "issues")
    for level, issue in issues:
        _sub(issues_el, "issue", issue.message, stage=issue.stage, code=issue.code, level=level)

    data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", pretty_print=True)
    if not validate:
        return XmlResult(status=StageStatus.OK, xml=data)
    return validate_xml(data)
