"""Construccion del documento XML de resultados con lxml.etree.

* Los textos se insertan como nodos de texto/atributos: lxml escapa
  ``< > & "`` automaticamente. Los caracteres prohibidos por XML 1.0 se
  sustituyen por U+FFFD (el OCR puede producir basura).
* Solo se guarda el NOMBRE del archivo de imagen, no su ruta completa.
* ``validation/@overall`` es VALIDATED unicamente si todas las etapas estan OK y
  no hay errores; en otro caso es NOT_VALIDATED y se listan los motivos.
  No se calcula ninguna puntuacion agregada.
* Antes de devolver, el documento se valida con xml_validator.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from lxml import etree

from app.models import (
    ConversionResult,
    OcrResult,
    RuleResult,
    SemanticNode,
    SemanticResult,
    StageIssue,
    StageStatus,
    XmlResult,
    utc_now,
)
from app.xml_validator import MATHML_NS, RESULT_NS, make_safe_parser, validate_xml

STAGE = "xml_builder"
SCHEMA_VERSION = "1.0"
NSMAP = {None: RESULT_NS, "m": MATHML_NS}

_ILLEGAL = re.compile("[^\u0009\u000A\u000D\u0020-\uD7FF\uE000-\uFFFD\U00010000-\U0010FFFF]")


def _clean(value: object) -> str:
    return _ILLEGAL.sub("\ufffd", str(value))


def _q(tag: str) -> str:
    return f"{{{RESULT_NS}}}{tag}"


def _sub(parent: etree._Element, tag: str, text: Optional[object] = None, **attrs: Optional[object]) -> etree._Element:
    el = etree.SubElement(parent, _q(tag))
    if text is not None:
        el.text = _clean(text)
    for key, value in attrs.items():
        if value is not None:
            el.set(key, _clean(value.value if hasattr(value, "value") else value))
    return el


def _add_node(parent: etree._Element, node: SemanticNode) -> None:
    el = _sub(parent, "node", id=node.id, kind=node.kind, evidence=node.evidence, role=node.role)
    if node.symbol is not None:
        _sub(el, "symbol", node.symbol)
    for child in node.children:
        _add_node(el, child)


def _stage_status(result: object) -> StageStatus:
    return result.status if result is not None else StageStatus.SKIPPED


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
    created_at: Optional[datetime] = None,
    validate: bool = True,
) -> XmlResult:
    """Construye (y por defecto valida) el documento XML de resultados."""
    rules = rules or []
    if rules_status is None:
        rules_status = StageStatus.OK if rules else StageStatus.SKIPPED
    statuses = {
        "ocr": _stage_status(ocr),
        "converter": _stage_status(conversion),
        "semantic_agent": _stage_status(semantic),
        "rules": rules_status,
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
    if rules_catalog_version:
        _sub(meta, "rulesCatalogVersion", rules_catalog_version)

    ref = image_ref or (ocr.image_ref if ocr is not None else "")
    _sub(_sub(root, "input"), "imageRef", Path(ref).name if ref else "")

    stages_el = _sub(root, "stages")
    for name, status in statuses.items():
        _sub(stages_el, "stage", name=name, status=status)

    issues: list[tuple[str, StageIssue]] = []
    for result in (ocr, conversion, semantic):
        if result is not None:
            issues += [("error", i) for i in result.errors] + [("warning", i) for i in result.warnings]

    if ocr is not None and (ocr.latex_raw is not None or ocr.latex_reviewed is not None):
        ocr_el = _sub(root, "ocr", attempt=ocr.attempt, reviewedByHuman="true" if ocr.reviewed_by_human else "false")
        if ocr.latex_raw is not None:
            _sub(ocr_el, "latexRaw", ocr.latex_raw)
        if ocr.latex_reviewed is not None:
            _sub(ocr_el, "latexReviewed", ocr.latex_reviewed)

    if conversion is not None and conversion.mathml:
        try:
            math_el = etree.fromstring(conversion.mathml.encode("utf-8"), parser=make_safe_parser())
            _sub(root, "mathml", kind=conversion.mathml_kind).append(math_el)
        except etree.XMLSyntaxError as exc:
            issues.append(("error", StageIssue(stage=STAGE, code="MATHML_UNPARSEABLE", message=str(exc))))

    if semantic is not None and semantic.output is not None:
        out = semantic.output
        sem = _sub(root, "semantic", provider=semantic.provider, model=semantic.model)
        _sub(sem, "expressionType", out.expression_type)
        _add_node(_sub(sem, "root"), out.root)
        rel_el = _sub(sem, "relations")
        for rel in out.relations:
            _sub(rel_el, "relation", type=rel.type, left=rel.left, right=rel.right, evidence=rel.evidence)
        assumptions = _sub(sem, "assumptions")
        for text in out.assumptions:
            _sub(assumptions, "assumption", text)
        ambiguities = _sub(sem, "ambiguities")
        for text in out.ambiguities:
            _sub(ambiguities, "ambiguity", text)
        _sub(sem, "requiresReview", "true" if out.requires_review else "false")

    rules_el = _sub(root, "rules", catalogVersion=None)
    for rule in rules:
        r_el = _sub(rules_el, "rule", id=rule.rule_id, version=rule.rule_version, origin=rule.origin,
                    outcome=rule.outcome, severity=rule.severity)
        _sub(r_el, "message", rule.message)
        for evidence in rule.evidence_refs:
            _sub(r_el, "evidenceRef", evidence)

    reasons = [f"{name}: {status.value}" for name, status in statuses.items() if status is not StageStatus.OK]
    has_errors = any(level == "error" for level, _ in issues)
    if has_errors:
        reasons.append("hay errores registrados en las etapas")
    validation_el = _sub(root, "validation", overall="NOT_VALIDATED" if reasons else "VALIDATED")
    for reason in reasons:
        _sub(validation_el, "reason", reason)

    issues_el = _sub(root, "issues")
    for level, issue in issues:
        _sub(issues_el, "issue", issue.message, stage=issue.stage, code=issue.code, level=level)

    data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", pretty_print=True)
    if not validate:
        return XmlResult(status=StageStatus.OK, xml=data)
    return validate_xml(data)
