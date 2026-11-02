"""Enriquecimiento OPCIONAL del MathML con atributos ``intent`` (MathML 4, Working Draft).

Garantias:
* Solo anade atributos ``intent`` y ``arg``; nunca cambia elementos, texto ni estructura.
  ``verify_unchanged`` lo comprueba comparando el MathML sin esos atributos.
* No sobrescribe un ``intent`` que ya exista (la intencion del autor manda).
* Solo usa nodos con intent definitivo, sin ambiguedad ni revision pendiente.
* Solo aplica una regla si la alineacion es inequivoca: mismo numero de elementos MathML y de
  nodos del arbol (en orden de documento) y mismos simbolos en las hojas correspondientes.
* Si se dispone de MathCAT, el resultado se acepta solo si MathCAT lo procesa en modo estricto
  (IntentErrorRecovery=Error). Si no hay MathCAT, el resultado se marca como no comprobado.

Los nombres de concepto de intent_mathml_mapping.json NO estan verificados contra una
especificacion cerrada (ver el propio archivo), por eso la funcion esta desactivada por defecto.
"""
from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from app.mathcat import MathCAT, MathCATError
from app.models import EnrichmentResult, SemanticNode, SemanticOutput, StageIssue, StageStatus

logger = logging.getLogger(__name__)
STAGE = "enrichment"
MATHML_NS = "http://www.w3.org/1998/Math/MathML"
MAPPING_PATH = Path(__file__).with_name("intent_mathml_mapping.json")
_NCNAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")
_ADDED_ATTRS = ("intent", "arg")


def _issue(code: str, message: str) -> StageIssue:
    return StageIssue(stage=STAGE, code=code, message=message)


def load_mapping(path: Optional[Path] = None) -> dict:
    mapping = json.loads(Path(path or MAPPING_PATH).read_text(encoding="utf-8"))
    for rule in mapping["rules"]:
        if not _NCNAME.match(rule["concept"]):
            raise ValueError(f"Concepto invalido (debe ser un NCName): {rule['concept']!r}")
        if len(rule["argument_roles"]) != len(rule["argument_names"]):
            raise ValueError(f"Regla {rule['intent_id']}: roles y nombres de argumento no coinciden.")
        if not all(_NCNAME.match(a) for a in rule["argument_names"]):
            raise ValueError(f"Regla {rule['intent_id']}: nombre de argumento invalido.")
    return mapping


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _preorder(output: SemanticOutput) -> list[SemanticNode]:
    by_id = {n.id: n for n in output.nodes}
    order: list[SemanticNode] = []
    seen: set[str] = set()

    def visit(nid: str) -> None:
        if nid in seen or nid not in by_id:
            return
        seen.add(nid)
        order.append(by_id[nid])
        for op in by_id[nid].operands:
            visit(op.node_id)

    visit(output.root_id)
    return order


def canonical(el: ET.Element) -> tuple:
    """Forma comparable de un elemento ignorando los atributos que anade el enriquecimiento."""
    attrs = tuple(sorted((k, v) for k, v in el.attrib.items() if k not in _ADDED_ATTRS))
    return (el.tag, (el.text or "").strip(), attrs, tuple(canonical(c) for c in el), (el.tail or "").strip())


def verify_unchanged(original: str, enriched: str) -> bool:
    try:
        return canonical(ET.fromstring(original)) == canonical(ET.fromstring(enriched))
    except ET.ParseError:
        return False


def enrich_mathml(
    mathml: str,
    semantic: SemanticOutput,
    *,
    mathcat: Optional[MathCAT] = None,
    mapping: Optional[dict] = None,
) -> EnrichmentResult:
    """Devuelve un EnrichmentResult. Nunca lanza excepciones; ante cualquier duda no modifica nada."""
    result = EnrichmentResult(status=StageStatus.SKIPPED, enabled=True)
    mapping = mapping or load_mapping()
    ET.register_namespace("", MATHML_NS)
    try:
        root = ET.fromstring(mathml)
    except ET.ParseError as exc:
        result.status = StageStatus.FAILED
        result.errors.append(_issue("MATHML_UNPARSEABLE", str(exc)))
        return result

    nodes = _preorder(semantic)
    for rule in mapping["rules"]:
        elements = [el for el in root.iter() if isinstance(el.tag, str) and _local(el.tag) == rule["mathml_element"]]
        wanted = [n for n in nodes if n.intent == rule["intent_id"]]
        if not elements and not wanted:
            continue
        if len(elements) != len(wanted):
            result.warnings.append(_issue(
                "COUNT_MISMATCH", f"{rule['intent_id']}: {len(wanted)} nodo(s) frente a {len(elements)} <{rule['mathml_element']}>."))
            continue
        for el, node in zip(elements, wanted):
            if node.review_required or node.ambiguous or node.evidence != "directly_supported":
                result.warnings.append(_issue("NODE_NOT_RELIABLE", f"{node.id}: nodo ambiguo o dependiente del contexto; se omite."))
                continue
            if "intent" in el.attrib:
                result.warnings.append(_issue("INTENT_ALREADY_PRESENT", f"<{rule['mathml_element']}> ya tiene intent; se respeta."))
                continue
            children = list(el)
            roles = rule["argument_roles"]
            if len(children) != len(roles) or any("arg" in c.attrib for c in children):
                result.warnings.append(_issue("SHAPE_MISMATCH", f"{node.id}: la forma de <{rule['mathml_element']}> no coincide."))
                continue
            by_role = {op.role: op.node_id for op in node.operands}
            by_id = {n.id: n for n in semantic.nodes}
            misaligned = False
            for child, role in zip(children, roles):
                operand = by_id.get(by_role.get(role, ""))
                if operand is None:
                    misaligned = True
                elif operand.symbol and _local(child.tag) in {"mi", "mn", "mo"} and (child.text or "").strip() != operand.symbol:
                    misaligned = True
            if misaligned:
                result.warnings.append(_issue("MISALIGNED", f"{node.id}: los operandos no coinciden con los hijos de MathML."))
                continue
            for child, name in zip(children, rule["argument_names"]):
                child.set("arg", name)
            el.set("intent", f"{rule['concept']}({','.join('$' + a for a in rule['argument_names'])})")
            result.applied.append(f"{node.id}:{rule['intent_id']}->{rule['concept']}")

    if not result.applied:
        result.warnings.append(_issue("NOTHING_TO_ENRICH", "No hay nada que enriquecer con seguridad."))
        return result

    enriched = ET.tostring(root, encoding="unicode")
    if not verify_unchanged(mathml, enriched):
        result.applied.clear()
        result.errors.append(_issue("MATHML_CHANGED", "El enriquecimiento alteraria el MathML: se descarta."))
        result.status = StageStatus.FAILED
        return result

    if mathcat is not None:
        try:
            mathcat.check_intent_attributes(enriched)
            result.mathcat_checked = True
        except MathCATError as exc:
            result.applied.clear()
            result.warnings.append(_issue("REJECTED_BY_MATHCAT", f"MathCAT no acepto el MathML enriquecido: {exc.message}"))
            return result
    else:
        result.warnings.append(_issue("NOT_CHECKED_WITH_MATHCAT", "Sin MathCAT no se pudo comprobar la compatibilidad."))

    result.mathml = enriched
    result.status = StageStatus.OK
    return result
