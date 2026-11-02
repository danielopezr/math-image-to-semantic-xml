"""Etapa 5: motor de reglas de intencion.

Separacion de responsabilidades:
* La validacion ESTRUCTURAL del XML vive en xml_validator (XSD + coherencia).
* Aqui se evaluan reglas sobre MathML y sobre la interpretacion semantica.

Cada regla del catalogo (rules_catalog.json) declara su ``origin``:
  w3c_normative      -> lo define una especificacion del W3C
  project_heuristic  -> criterio propio del proyecto (NO es del W3C)
  llm_derived        -> conclusion generada por el LLM

Resultados: PASS, FAIL, REVIEW, NOT_APPLICABLE. Sin datos suficientes se
devuelve NOT_APPLICABLE o REVIEW, nunca PASS. No se calcula puntuacion agregada.
"""
from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from app.models import (
    RuleOrigin,
    RuleOutcome,
    RuleResult,
    RuleSeverity,
    SemanticNode,
    SemanticOutput,
    SemanticResult,
    StageStatus,
)
from app.semantic_tree import errors_only, validate_semantic_output

logger = logging.getLogger(__name__)
CATALOG_PATH = Path(__file__).with_name("rules_catalog.json")
MATHML_NS = "http://www.w3.org/1998/Math/MathML"

# Elementos de MathML Core (lista contrastada con el primer borrador publico y con
# extractos del CR de 2025-06-24; volver a contrastar con el CR vigente).
MATHML_CORE_ELEMENTS = frozenset({
    "annotation", "annotation-xml", "maction", "math", "merror", "mfrac", "mi",
    "mmultiscripts", "mn", "mo", "mover", "mpadded", "mphantom", "mprescripts",
    "mroot", "mrow", "ms", "mspace", "msqrt", "mstyle", "msub", "msubsup", "msup",
    "mtable", "mtd", "mtext", "mtr", "munder", "munderover", "none", "semantics",
})
TOKEN_ELEMENTS = frozenset({"mi", "mn", "mo", "mtext", "ms"})
# Correspondencia habitual (NO definida por el W3C) entre elementos de presentacion e intents del proyecto.
STRUCTURE_MAP = {
    "mfrac": frozenset({"fraction", "division"}),
    "msqrt": frozenset({"root"}),
    "mroot": frozenset({"root"}),
    "msup": frozenset({"power", "exponential"}),
}


class CatalogError(ValueError):
    """El catalogo es inconsistente."""


@dataclass
class Context:
    mathml_root: Optional[ET.Element] = None
    mathml_error: Optional[str] = None
    semantic: Optional[SemanticOutput] = None
    elements: list[ET.Element] = field(default_factory=list)
    tokens: set[str] = field(default_factory=set)
    local_names: set[str] = field(default_factory=set)  # solo elementos en el espacio de nombres MathML


Decision = tuple[RuleOutcome, str, list[str]]


def _norm(symbol: str) -> str:
    return symbol.replace("\u2212", "-").strip()


def _split(tag: object) -> tuple[str, str]:
    if not isinstance(tag, str):
        return "", ""
    if tag.startswith("{"):
        ns, local = tag[1:].split("}", 1)
        return ns, local
    return "", tag


# ------------------------------------------------------------------- reglas
def mml_001(ctx: Context) -> Decision:
    if ctx.mathml_error:
        return RuleOutcome.FAIL, f"MathML no analizable: {ctx.mathml_error}", []
    if ctx.mathml_root.tag == f"{{{MATHML_NS}}}math":
        return RuleOutcome.PASS, "La raiz es <math> en el espacio de nombres MathML.", []
    return RuleOutcome.FAIL, f"La raiz es {ctx.mathml_root.tag!r}, no {{MathML}}math.", []


def mml_002(ctx: Context) -> Decision:
    if ctx.mathml_root is None:
        return RuleOutcome.NOT_APPLICABLE, "No hay MathML analizable.", []
    outside = sorted(n for n in ctx.local_names if n not in MATHML_CORE_ELEMENTS)
    foreign = sorted({el.tag for el in ctx.elements if _split(el.tag)[0] != MATHML_NS})
    if not outside and not foreign:
        return RuleOutcome.PASS, "Todos los elementos pertenecen a MathML Core.", []
    parts = []
    if outside:
        parts.append("fuera de Core: " + ", ".join(outside))
    if foreign:
        parts.append("fuera del espacio de nombres MathML: " + ", ".join(foreign))
    return RuleOutcome.REVIEW, "Elementos " + "; ".join(parts) + ".", outside + foreign


def mml_003(ctx: Context) -> Decision:
    if ctx.mathml_root is None:
        return RuleOutcome.NOT_APPLICABLE, "No hay MathML analizable.", []
    with_intent = sorted({_split(el.tag)[1] for el in ctx.elements if "intent" in el.attrib})
    if not with_intent:
        return RuleOutcome.NOT_APPLICABLE, "Ningun elemento usa 'intent'.", []
    return (RuleOutcome.REVIEW,
            "Se usa 'intent' (MathML 4, aun Working Draft) en: " + ", ".join(with_intent) +
            ". No se evalua su gramatica.", with_intent)


def _nodes(ctx: Context) -> list[SemanticNode]:
    return list(ctx.semantic.nodes) if ctx.semantic is not None else []


def sem_001(ctx: Context) -> Decision:
    if ctx.semantic is None:
        return RuleOutcome.REVIEW, "No hay interpretacion semantica valida.", []
    return RuleOutcome.PASS, "Hay una interpretacion semantica que cumple el esquema.", []


def sem_002(ctx: Context) -> Decision:
    if ctx.semantic is None or ctx.mathml_root is None:
        return RuleOutcome.NOT_APPLICABLE, "Faltan la interpretacion o un MathML analizable.", []
    missing = [n for n in _nodes(ctx) if n.symbol and _norm(n.symbol) not in ctx.tokens]
    if not missing:
        return RuleOutcome.PASS, "Todos los simbolos de la interpretacion aparecen en el MathML.", []
    return (RuleOutcome.FAIL,
            "Simbolos ausentes del MathML: " + ", ".join(sorted({n.symbol for n in missing})),
            [n.id for n in missing])


def sem_003(ctx: Context) -> Decision:
    if ctx.semantic is None or ctx.mathml_root is None:
        return RuleOutcome.NOT_APPLICABLE, "Faltan la interpretacion o un MathML analizable.", []
    if "=" not in ctx.tokens:
        return RuleOutcome.NOT_APPLICABLE, "El MathML no contiene '='.", []
    if any(n.intent == "equality" for n in _nodes(ctx)):
        return RuleOutcome.PASS, "La igualdad del MathML esta reflejada como intent 'equality'.", []
    return RuleOutcome.REVIEW, "El MathML contiene '=' pero la interpretacion no declara un nodo 'equality'.", []


def sem_004(ctx: Context) -> Decision:
    if ctx.semantic is None or ctx.mathml_root is None:
        return RuleOutcome.NOT_APPLICABLE, "Faltan la interpretacion o un MathML analizable.", []
    present = {el: kinds for el, kinds in STRUCTURE_MAP.items() if el in ctx.local_names}
    if not present:
        return RuleOutcome.NOT_APPLICABLE, "El MathML no contiene mfrac, msqrt, mroot ni msup.", []
    intents = {n.intent for n in _nodes(ctx) if n.intent}
    missing = sorted(f"{el}->{'|'.join(sorted(kinds))}" for el, kinds in present.items() if not (kinds & intents))
    if not missing:
        return RuleOutcome.PASS, "Las estructuras presentes tienen un intent correspondiente.", []
    return RuleOutcome.REVIEW, "Sin intent correspondiente para: " + ", ".join(missing) + ".", missing


def sem_005(ctx: Context) -> Decision:
    if ctx.semantic is None:
        return RuleOutcome.NOT_APPLICABLE, "No hay interpretacion semantica.", []
    flagged = [n.id for n in _nodes(ctx) if n.ambiguous or n.review_required]
    if (flagged or ctx.semantic.ambiguities) and not ctx.semantic.review_required:
        return (RuleOutcome.FAIL,
                "Hay ambiguedades o nodos marcados para revision pero review_required es falso.", flagged)
    return RuleOutcome.PASS, "Las ambiguedades (si las hay) implican revision.", flagged


def sem_006(ctx: Context) -> Decision:
    if ctx.semantic is None:
        return RuleOutcome.NOT_APPLICABLE, "No hay interpretacion semantica.", []
    refs = [n.id for n in _nodes(ctx) if n.evidence == "context_dependent"]
    refs += [f"link:{l.type}:{l.source}->{l.target}" for l in ctx.semantic.links if l.evidence == "context_dependent"]
    if not refs:
        return RuleOutcome.NOT_APPLICABLE, "No hay elementos dependientes del contexto.", []
    if ctx.semantic.assumptions:
        return RuleOutcome.PASS, "Los elementos dependientes del contexto tienen supuestos declarados.", refs
    return RuleOutcome.REVIEW, "Hay elementos dependientes del contexto sin supuestos declarados.", refs


def sem_007(ctx: Context) -> Decision:
    if ctx.semantic is None:
        return RuleOutcome.NOT_APPLICABLE, "No hay interpretacion semantica.", []
    if ctx.semantic.review_required:
        return RuleOutcome.REVIEW, "El LLM declara que su interpretacion requiere revision.", []
    return RuleOutcome.PASS, "El LLM no declara necesidad de revision.", []


def sem_008(ctx: Context) -> Decision:
    """Revalida el arbol contra el vocabulario controlado (defensa en profundidad)."""
    if ctx.semantic is None:
        return RuleOutcome.NOT_APPLICABLE, "No hay interpretacion semantica.", []
    errors = errors_only(validate_semantic_output(ctx.semantic))
    if not errors:
        return RuleOutcome.PASS, "El arbol cumple el vocabulario controlado y su estructura.", []
    shown = "; ".join(f"[{p.code}] {p.message}" for p in errors[:3])
    return RuleOutcome.FAIL, f"{len(errors)} problema(s) estructurales: {shown}", [p.node_id for p in errors if p.node_id]


IMPLEMENTATIONS: dict[str, Callable[[Context], Decision]] = {
    "MML-001": mml_001, "MML-002": mml_002, "MML-003": mml_003,
    "SEM-001": sem_001, "SEM-002": sem_002, "SEM-003": sem_003, "SEM-004": sem_004,
    "SEM-005": sem_005, "SEM-006": sem_006, "SEM-007": sem_007, "SEM-008": sem_008,
}


# ------------------------------------------------------------------ catalogo
_REQUIRED_RULE_KEYS = {"id", "version", "title", "description", "applies_when", "expected",
                       "severity", "origin", "source", "examples"}


def validate_catalog(catalog: dict) -> None:
    """Lanza CatalogError si el catalogo es inconsistente con el codigo."""
    if not catalog.get("catalog_version"):
        raise CatalogError("Falta catalog_version.")
    rules = catalog.get("rules")
    if not isinstance(rules, list) or not rules:
        raise CatalogError("El catalogo no contiene reglas.")
    fixtures = catalog.get("fixtures", {})
    seen: set[str] = set()
    for rule in rules:
        missing = _REQUIRED_RULE_KEYS - set(rule)
        if missing:
            raise CatalogError(f"Regla {rule.get('id', '?')}: faltan campos {sorted(missing)}")
        rid = rule["id"]
        if rid in seen:
            raise CatalogError(f"Id de regla duplicado: {rid}")
        seen.add(rid)
        if rid not in IMPLEMENTATIONS:
            raise CatalogError(f"La regla {rid} no tiene implementacion.")
        try:
            RuleOrigin(rule["origin"]); RuleSeverity(rule["severity"])
        except ValueError as exc:
            raise CatalogError(f"Regla {rid}: {exc}") from exc
        if not rule["examples"]:
            raise CatalogError(f"La regla {rid} no tiene ejemplos de prueba.")
        for ex in rule["examples"]:
            try:
                RuleOutcome(ex["expected"])
            except ValueError as exc:
                raise CatalogError(f"Regla {rid}: resultado esperado invalido ({exc})") from exc
            if ex["mathml"] not in fixtures.get("mathml", {}):
                raise CatalogError(f"Regla {rid}: fixture mathml desconocida {ex['mathml']!r}")
            if ex["semantic"] is not None and ex["semantic"] not in fixtures.get("semantic", {}):
                raise CatalogError(f"Regla {rid}: fixture semantic desconocida {ex['semantic']!r}")
    orphans = sorted(set(IMPLEMENTATIONS) - seen)
    if orphans:
        raise CatalogError(f"Implementaciones sin regla en el catalogo: {orphans}")


def load_catalog(path: Optional[Path] = None) -> dict:
    catalog = json.loads(Path(path or CATALOG_PATH).read_text(encoding="utf-8"))
    validate_catalog(catalog)
    return catalog


# ------------------------------------------------------------------- motor
def _build_context(mathml: Optional[str], semantic_result: Optional[SemanticResult]) -> Context:
    ctx = Context()
    if semantic_result is not None and semantic_result.status is StageStatus.OK and semantic_result.output is not None:
        ctx.semantic = semantic_result.output
    if not mathml or not mathml.strip():
        ctx.mathml_error = "No hay MathML."
        return ctx
    try:
        root = ET.fromstring(mathml)
    except ET.ParseError as exc:
        ctx.mathml_error = str(exc)
        return ctx
    ctx.mathml_root = root
    ctx.elements = [el for el in root.iter() if isinstance(el.tag, str)]
    for el in ctx.elements:
        ns, local = _split(el.tag)
        if ns == MATHML_NS:
            ctx.local_names.add(local)
            if local in TOKEN_ELEMENTS and el.text and el.text.strip():
                ctx.tokens.add(_norm(el.text))
    return ctx


def evaluate_intents(
    mathml: Optional[str],
    semantic_result: Optional[SemanticResult],
    *,
    catalog: Optional[dict] = None,
) -> list[RuleResult]:
    """Evalua todas las reglas del catalogo. Una regla que falla internamente da REVIEW."""
    catalog = catalog or load_catalog()
    ctx = _build_context(mathml, semantic_result)
    results: list[RuleResult] = []
    for rule in catalog["rules"]:
        try:
            outcome, message, refs = IMPLEMENTATIONS[rule["id"]](ctx)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error evaluando la regla %s", rule["id"])
            outcome, message, refs = RuleOutcome.REVIEW, f"Error interno al evaluar la regla: {exc}", []
        results.append(RuleResult(
            rule_id=rule["id"],
            rule_version=rule["version"],
            origin=RuleOrigin(rule["origin"]),
            outcome=outcome,
            severity=RuleSeverity(rule["severity"]),
            message=message,
            evidence_refs=list(refs),
        ))
    return results
