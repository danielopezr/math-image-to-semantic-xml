"""Validacion de documentos XML de resultados (lxml).

Dos niveles, informados por separado:
  1. Estructural: buena formacion + validez frente a result.xsd (XSD 1.0).
  2. Coherencia entre campos: comprobaciones que el XSD no puede expresar.

Seguridad (XXE): el parser no resuelve entidades, no accede a la red y no carga
DTD; ademas se rechaza cualquier documento que declare DOCTYPE o ENTITY.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

from lxml import etree

from app.models import StageIssue, StageStatus, XmlResult

STAGE = "xml_validator"
RESULT_NS = "urn:math-semantic-ocr:result:1"
MATHML_NS = "http://www.w3.org/1998/Math/MathML"
XSD_PATH = Path(__file__).parent / "schemas" / "result.xsd"
NSMAP = {"t": RESULT_NS}

_DTD_PATTERN = re.compile(rb"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)


def make_safe_parser() -> etree.XMLParser:
    return etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        dtd_validation=False,
        huge_tree=False,
    )


@lru_cache(maxsize=1)
def load_schema() -> etree.XMLSchema:
    # Archivo local; el parser de esquemas tampoco accede a la red.
    return etree.XMLSchema(etree.parse(str(XSD_PATH), parser=make_safe_parser()))


def _issue(code: str, message: str) -> StageIssue:
    return StageIssue(stage=STAGE, code=code, message=message)


def _cross_checks(root: etree._Element) -> list[StageIssue]:
    """Coherencia entre campos que XSD 1.0 no puede verificar."""
    problems: list[StageIssue] = []
    stage_els = root.xpath("t:stages/t:stage", namespaces=NSMAP)
    statuses = {s.get("name"): s.get("status") for s in stage_els}
    required = {s.get("name"): s.get("required") == "true" for s in stage_els}
    val = root.xpath("t:validation", namespaces=NSMAP)[0]

    if val.get("overall") == "VALIDATED":
        bad = sorted(f"{n}={st}" for n, st in statuses.items()
                     if st == "FAILED" or (required.get(n) and st != "OK"))
        if bad:
            problems.append(_issue("OVERALL_INCONSISTENT",
                                   "validation=VALIDATED pero hay etapas fallidas u obligatorias sin OK: " + ", ".join(bad)))
    for core in ("ocr", "converter", "semantic_agent", "rules"):
        if not required.get(core):
            problems.append(_issue("CORE_STAGE_NOT_REQUIRED", f"La etapa '{core}' siempre es obligatoria."))

    expected = {"ocr": "t:ocr", "converter": "t:mathml", "semantic_agent": "t:semantic",
                "braille": "t:braille", "speech": "t:speech", "audio": "t:audio"}
    for stage, path in expected.items():
        if statuses.get(stage) == "OK" and not root.xpath(path, namespaces=NSMAP):
            problems.append(_issue("STAGE_CONTENT_MISSING", f"La etapa '{stage}' esta OK pero falta su contenido ({path})."))
        if statuses.get(stage) != "OK" and stage in {"braille", "speech", "audio"} and root.xpath(path, namespaces=NSMAP):
            problems.append(_issue("STAGE_CONTENT_UNEXPECTED", f"Hay contenido de '{stage}' pero la etapa no esta OK."))

    for attr, stage in (("brailleGenerated", "braille"), ("speechGenerated", "speech"), ("audioGenerated", "audio")):
        if (val.get(attr) == "true") != (statuses.get(stage) == "OK"):
            problems.append(_issue("FACET_INCONSISTENT", f"validation/@{attr} no coincide con el estado de '{stage}'."))
    sem = root.xpath("t:semantic", namespaces=NSMAP)
    declared = sem[0].get("reviewRequired") == "true" if sem else False
    if (val.get("semanticReviewRequired") == "true") != declared:
        problems.append(_issue("FACET_INCONSISTENT", "validation/@semanticReviewRequired no coincide con semantic/@reviewRequired."))

    if sem:
        problems += _semantic_checks(sem[0])
    return problems


def _semantic_checks(sem: etree._Element) -> list[StageIssue]:
    """Ciclos y pertenencia al vocabulario (el XSD solo garantiza ids y referencias existentes)."""
    problems: list[StageIssue] = []
    children: dict[str, list[str]] = {}
    for node in sem.xpath("t:nodes/t:node", namespaces=NSMAP):
        children[node.get("id")] = [o.get("nodeId") for o in node.xpath("t:operand", namespaces=NSMAP)]
    state: dict[str, int] = {}

    def dfs(nid: str) -> bool:
        state[nid] = 1
        for child in children.get(nid, []):
            if state.get(child) == 1 or (child not in state and dfs(child)):
                return True
        state[nid] = 2
        return False

    if any(nid not in state and dfs(nid) for nid in list(children)):
        problems.append(_issue("CYCLE_IN_XML", "El arbol semantico del XML contiene una referencia circular."))
    try:
        from app.intents import load_vocabulary

        vocab = load_vocabulary()
        if sem.get("vocabularyVersion") != vocab.version:
            problems.append(_issue("VOCABULARY_VERSION", f"Version de vocabulario {sem.get('vocabularyVersion')!r}; se esperaba {vocab.version!r}."))
        for node in sem.xpath("t:nodes/t:node", namespaces=NSMAP):
            intent = node.get("intent")
            if intent is not None and vocab.get(intent) is None:
                problems.append(_issue("UNKNOWN_INTENT_IN_XML", f"Intent fuera del vocabulario en el nodo {node.get('id')}: {intent!r}."))
            if node.get("ambiguous") == "true" and intent is not None:
                problems.append(_issue("AMBIGUOUS_HAS_INTENT_IN_XML", f"El nodo ambiguo {node.get('id')} no puede tener intent."))
    except Exception as exc:  # noqa: BLE001  (vocabulario ilegible: no se puede comprobar)
        problems.append(_issue("VOCABULARY_UNAVAILABLE", f"No se pudo cargar el vocabulario: {exc}"))
    return problems


def validate_xml(data: Union[bytes, str]) -> XmlResult:
    """Valida un documento de resultados. Nunca lanza excepciones por contenido invalido."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    result = XmlResult(status=StageStatus.FAILED, xml=raw)

    if _DTD_PATTERN.search(raw):
        result.errors.append(_issue("DTD_NOT_ALLOWED", "Se rechazan documentos con DOCTYPE o ENTITY (proteccion XXE)."))
        return result

    try:
        root = etree.fromstring(raw, parser=make_safe_parser())
    except etree.XMLSyntaxError as exc:
        result.errors.append(_issue("XML_NOT_WELL_FORMED", f"XML mal formado: {exc}"))
        return result
    except ValueError as exc:
        result.errors.append(_issue("XML_UNPARSEABLE", f"No se pudo analizar el XML: {exc}"))
        return result
    result.well_formed = True

    if root.getroottree().docinfo.doctype:
        result.errors.append(_issue("DTD_NOT_ALLOWED", "El documento declara un DOCTYPE."))
        return result

    schema = load_schema()
    if not schema.validate(root):
        for err in schema.error_log:
            result.errors.append(_issue("XSD_INVALID", f"linea {err.line}: {err.message}"))
        return result
    result.schema_valid = True

    problems = _cross_checks(root)
    if problems:
        result.errors.extend(problems)
        return result

    result.status = StageStatus.OK
    return result


def overall_status(data: Union[bytes, str]) -> Optional[str]:
    """Devuelve el valor de validation/@overall del documento, o None si no se puede leer."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    if _DTD_PATTERN.search(raw):
        return None
    try:
        root = etree.fromstring(raw, parser=make_safe_parser())
    except (etree.XMLSyntaxError, ValueError):
        return None
    value = root.xpath("string(t:validation/@overall)", namespaces=NSMAP)
    return value or None
