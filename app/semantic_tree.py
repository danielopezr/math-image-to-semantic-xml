"""Arbol semantico: validacion estructural, orden de evaluacion y metricas.

El arbol se representa como grafo plano (``SemanticOutput.nodes`` + ``Operand`` por id) para
poder DETECTAR ciclos, padres multiples y nodos huerfanos, cosa que un JSON anidado oculta.

La validacion comprueba forma y coherencia con el vocabulario controlado. NO demuestra que
la interpretacion sea matematicamente correcta.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from app.intents import IntentDef, Vocabulary, load_vocabulary
from app.models import SemanticNode, SemanticOutput

TOKEN_ELEMENTS = frozenset({"mi", "mn", "mo", "mtext", "ms"})
# Tokens del MathML que no necesitan un nodo propio (delimitadores y operadores invisibles).
UNCOVERED_IGNORED = frozenset({"(", ")", "[", "]", "{", "}", ",", ";", "|", "\u2061", "\u2062", "\u2063", "\u2064"})


@dataclass(frozen=True)
class SemanticProblem:
    code: str
    message: str
    severity: str = "error"  # "error" | "warning"
    node_id: Optional[str] = None


def _norm(symbol: str) -> str:
    return symbol.replace("\u2212", "-").strip()


def mathml_tokens(mathml: Optional[str]) -> set[str]:
    tokens: set[str] = set()
    if not mathml:
        return tokens
    try:
        root = ET.fromstring(mathml)
    except ET.ParseError:
        return tokens
    for el in root.iter():
        local = el.tag.split("}", 1)[-1] if isinstance(el.tag, str) else ""
        if local in TOKEN_ELEMENTS and el.text and el.text.strip():
            tokens.add(_norm(el.text))
    return tokens


def children_ids(node: SemanticNode) -> list[str]:
    return [op.node_id for op in node.operands]


def _index(output: SemanticOutput) -> tuple[dict[str, SemanticNode], list[SemanticProblem]]:
    by_id: dict[str, SemanticNode] = {}
    problems: list[SemanticProblem] = []
    for node in output.nodes:
        if node.id in by_id:
            problems.append(SemanticProblem("DUPLICATE_ID", f"Id de nodo duplicado: {node.id}", node_id=node.id))
        else:
            by_id[node.id] = node
    return by_id, problems


def _find_cycle(by_id: dict[str, SemanticNode]) -> Optional[str]:
    WHITE, GREY, BLACK = 0, 1, 2
    color = {i: WHITE for i in by_id}
    for start in by_id:
        if color[start] != WHITE:
            continue
        stack = [(start, iter(children_ids(by_id[start])))]
        color[start] = GREY
        while stack:
            current, it = stack[-1]
            for child in it:
                if child not in by_id:
                    continue
                if color[child] == GREY:
                    return child
                if color[child] == WHITE:
                    color[child] = GREY
                    stack.append((child, iter(children_ids(by_id[child]))))
                    break
            else:
                color[current] = BLACK
                stack.pop()
    return None


def _reachable(by_id: dict[str, SemanticNode], root_id: str) -> set[str]:
    seen: set[str] = set()
    stack = [root_id]
    while stack:
        current = stack.pop()
        if current in seen or current not in by_id:
            continue
        seen.add(current)
        stack.extend(children_ids(by_id[current]))
    return seen


def _check_roles(node: SemanticNode, d: IntentDef, out: list[SemanticProblem]) -> None:
    roles = Counter(op.role for op in node.operands)
    allowed = set(d.role_names)
    for role in roles:
        if role not in allowed:
            out.append(SemanticProblem("BAD_ROLE", f"{node.id}: el rol '{role}' no existe en '{d.id}'.", node_id=node.id))
    for r in d.roles:
        if roles.get(r.name, 0) > 1:
            out.append(SemanticProblem("DUPLICATE_ROLE", f"{node.id}: rol repetido '{r.name}'.", node_id=node.id))
        if r.required and roles.get(r.name, 0) == 0:
            out.append(SemanticProblem("MISSING_ROLE", f"{node.id}: falta el operando '{r.name}' de '{d.id}'.", node_id=node.id))
    if d.variadic_role and roles.get(d.variadic_role, 0) < d.variadic_min:
        out.append(SemanticProblem(
            "BAD_ARITY", f"{node.id}: '{d.id}' necesita al menos {d.variadic_min} operando(s) '{d.variadic_role}'.",
            node_id=node.id))
    for group in d.both_or_none:
        present = [g for g in group if roles.get(g, 0) > 0]
        if 0 < len(present) < len(group):
            out.append(SemanticProblem(
                "BOTH_OR_NONE", f"{node.id}: {' y '.join(group)} deben aparecer juntos o ninguno.", node_id=node.id))


def _check_properties(node: SemanticNode, d: IntentDef, out: list[SemanticProblem]) -> None:
    for key, value in node.properties.items():
        allowed = d.properties.get(key)
        if allowed is None:
            out.append(SemanticProblem("BAD_PROPERTY", f"{node.id}: propiedad '{key}' no permitida en '{d.id}'.", node_id=node.id))
        elif allowed == "integer":
            if not str(value).isdigit() or int(value) < 1:
                out.append(SemanticProblem("BAD_PROPERTY", f"{node.id}: '{key}' debe ser un entero positivo.", node_id=node.id))
        elif value not in allowed:
            out.append(SemanticProblem("BAD_PROPERTY", f"{node.id}: valor invalido '{value}' para '{key}'.", node_id=node.id))
    for key in d.required_properties:
        if key not in node.properties:
            out.append(SemanticProblem("MISSING_PROPERTY", f"{node.id}: falta la propiedad '{key}'.", node_id=node.id))


def _check_node(node: SemanticNode, vocab: Vocabulary, by_id: dict[str, SemanticNode], out: list[SemanticProblem]) -> None:
    nid = node.id
    if node.node_type not in vocab.node_types:
        out.append(SemanticProblem("BAD_NODE_TYPE", f"{nid}: tipo de nodo no permitido: {node.node_type!r}.", node_id=nid))
        return
    if node.symbol is not None and not node.symbol.strip():
        out.append(SemanticProblem("EMPTY_SYMBOL", f"{nid}: simbolo vacio.", node_id=nid))

    if node.ambiguous:
        if node.node_type != "unresolved":
            out.append(SemanticProblem("AMBIGUOUS_TYPE", f"{nid}: un nodo ambiguo debe tener node_type 'unresolved'.", node_id=nid))
        if node.intent is not None:
            out.append(SemanticProblem("AMBIGUOUS_HAS_INTENT", f"{nid}: un nodo ambiguo no puede tener intent definitivo.", node_id=nid))
        cands = node.candidate_intents
        if len(cands) < 2 or len(set(cands)) != len(cands):
            out.append(SemanticProblem("BAD_CANDIDATES", f"{nid}: se necesitan al menos 2 candidatos distintos.", node_id=nid))
        for c in cands:
            if vocab.get(c) is None:
                out.append(SemanticProblem("UNKNOWN_CANDIDATE", f"{nid}: candidato fuera del vocabulario: {c!r}.", node_id=nid))
        if not node.review_required:
            out.append(SemanticProblem("AMBIGUOUS_NO_REVIEW", f"{nid}: un nodo ambiguo exige review_required.", node_id=nid))
        if not node.justification:
            out.append(SemanticProblem("NO_JUSTIFICATION", f"{nid}: falta justificar la ambiguedad.", "warning", nid))
        for op in node.operands:
            if op.role != vocab.unresolved_child_role:
                out.append(SemanticProblem(
                    "BAD_ROLE", f"{nid}: los operandos de un nodo ambiguo usan el rol '{vocab.unresolved_child_role}'.", node_id=nid))
        return

    if node.node_type == "unresolved":
        out.append(SemanticProblem("UNRESOLVED_NOT_AMBIGUOUS", f"{nid}: 'unresolved' solo se admite con ambiguous=true.", node_id=nid))
        return
    if node.candidate_intents:
        out.append(SemanticProblem("CANDIDATES_WITHOUT_AMBIGUITY", f"{nid}: hay candidatos pero el nodo no es ambiguo.", node_id=nid))
    d = vocab.get(node.intent)
    if d is None:
        out.append(SemanticProblem("UNKNOWN_INTENT", f"{nid}: intent fuera del vocabulario: {node.intent!r}.", node_id=nid))
        return
    if d.category != node.node_type:
        out.append(SemanticProblem(
            "TYPE_INTENT_MISMATCH", f"{nid}: '{d.id}' es de categoria '{d.category}', no '{node.node_type}'.", node_id=nid))
    if node.node_type in {"element", "function"} and not node.symbol and d.id in {
            "variable", "constant", "number", "symbol", "parameter", "function"}:
        out.append(SemanticProblem("ELEMENT_WITHOUT_SYMBOL", f"{nid}: un '{d.id}' necesita simbolo.", node_id=nid))
    if node.implicit:
        if d.id not in {"multiplication", "function_application"}:
            out.append(SemanticProblem("BAD_IMPLICIT", f"{nid}: implicit solo aplica a multiplication o function_application.", node_id=nid))
        if node.symbol:
            out.append(SemanticProblem("IMPLICIT_WITH_SYMBOL", f"{nid}: un operador implicito no tiene simbolo escrito.", node_id=nid))

    _check_roles(node, d, out)
    _check_properties(node, d, out)

    def child(role: str) -> Optional[SemanticNode]:
        for op in node.operands:
            if op.role == role:
                return by_id.get(op.node_id)
        return None

    if d.id == "equation":
        c = child("statement")
        if c is not None and c.node_type != "relation":
            out.append(SemanticProblem("EQUATION_NEEDS_RELATION", f"{nid}: el enunciado de una ecuacion debe ser una relacion.", node_id=nid))
    if d.id == "function_application":
        c = child("function")
        if c is not None and c.node_type != "function":
            out.append(SemanticProblem("APPLICATION_NEEDS_FUNCTION", f"{nid}: el operando 'function' debe ser una funcion.", node_id=nid))
    if d.id == "matrix":
        for op in node.operands:
            c = by_id.get(op.node_id)
            if c is not None and c.intent != "vector":
                out.append(SemanticProblem("MATRIX_ROWS", f"{nid}: las filas de una matriz deben ser 'vector'.", node_id=nid))


def validate_semantic_output(
    output: SemanticOutput,
    vocabulary: Optional[Vocabulary] = None,
    *,
    latex: Optional[str] = None,
    mathml: Optional[str] = None,
) -> list[SemanticProblem]:
    """Devuelve la lista de problemas (errores y advertencias). Lista sin errores = estructura valida."""
    vocab = vocabulary or load_vocabulary()
    problems: list[SemanticProblem] = []

    if output.vocabulary_version != vocab.version:
        problems.append(SemanticProblem(
            "VOCABULARY_VERSION", f"Version de vocabulario {output.vocabulary_version!r}; se esperaba {vocab.version!r}."))

    by_id, dup = _index(output)
    problems += dup
    if output.root_id not in by_id:
        problems.append(SemanticProblem("ROOT_MISSING", f"root_id inexistente: {output.root_id!r}."))
        return problems

    parents: Counter = Counter()
    for node in by_id.values():
        for op in node.operands:
            if op.node_id not in by_id:
                problems.append(SemanticProblem("DANGLING_OPERAND", f"{node.id}: operando inexistente {op.node_id!r}.", node_id=node.id))
            elif op.node_id == node.id:
                problems.append(SemanticProblem("SELF_REFERENCE", f"{node.id} se referencia a si mismo.", node_id=node.id))
            else:
                parents[op.node_id] += 1
    for nid, count in parents.items():
        if count > 1:
            problems.append(SemanticProblem("MULTIPLE_PARENTS", f"{nid} tiene {count} padres (debe ser un arbol).", node_id=nid))
    if parents.get(output.root_id):
        problems.append(SemanticProblem("ROOT_HAS_PARENT", "La raiz no puede ser operando de otro nodo.", node_id=output.root_id))
    cyc = _find_cycle(by_id)
    if cyc is not None:
        problems.append(SemanticProblem("CYCLE", f"Referencia circular que pasa por {cyc}.", node_id=cyc))
    reach = _reachable(by_id, output.root_id)
    for nid in by_id:
        if nid not in reach:
            problems.append(SemanticProblem("ORPHAN_NODE", f"{nid} no es alcanzable desde la raiz.", node_id=nid))

    for node in by_id.values():
        _check_node(node, vocab, by_id, problems)

    flagged = [n.id for n in by_id.values() if n.ambiguous or n.review_required]
    if (flagged or output.ambiguities) and not output.review_required:
        problems.append(SemanticProblem(
            "REVIEW_FLAG_INCONSISTENT", "Hay nodos ambiguos, marcados para revision o ambiguedades, pero review_required es falso."))

    for link in output.links:
        s, t = by_id.get(link.source), by_id.get(link.target)
        if s is None or t is None:
            problems.append(SemanticProblem("LINK_DANGLING", f"Enlace con id inexistente: {link.source} -> {link.target}."))
            continue
        if link.source == link.target:
            problems.append(SemanticProblem("LINK_SELF", f"Enlace de {link.source} consigo mismo.", node_id=link.source))
        if link.type == "binds" and not (s.node_type == "calculus" and t.node_type == "element"):
            problems.append(SemanticProblem("LINK_BINDS", "'binds' une un nodo de calculo con un elemento.", node_id=link.source))

    if latex is not None or mathml is not None:
        tokens = mathml_tokens(mathml)
        latex_text = _norm(latex or "")
        missing = sorted({_norm(n.symbol) for n in by_id.values()
                          if n.symbol and _norm(n.symbol) not in tokens and _norm(n.symbol) not in latex_text})
        for sym in missing:
            problems.append(SemanticProblem("GROUNDING_MISMATCH", f"El simbolo {sym!r} no aparece en la formula."))
        used = {_norm(n.symbol) for n in by_id.values() if n.symbol}
        uncovered = sorted(t for t in tokens if t not in used and t not in UNCOVERED_IGNORED)
        if uncovered:
            problems.append(SemanticProblem(
                "UNCOVERED_TOKEN", "Tokens de la formula sin nodo asociado: " + ", ".join(uncovered), "warning"))
    return problems


def errors_only(problems: list[SemanticProblem]) -> list[SemanticProblem]:
    return [p for p in problems if p.severity == "error"]


def assign_evaluation_order(output: SemanticOutput) -> int:
    """Asigna orden de evaluacion en postorden (operandos antes que el padre). Devuelve cuantos nodos numero.

    El valor que traiga el LLM se sobrescribe: lo calcula el proyecto, no el modelo.
    """
    by_id = {n.id: n for n in output.nodes}
    for n in output.nodes:
        n.evaluation_order = None
    counter = 0
    seen: set[str] = set()

    def visit(nid: str) -> None:
        nonlocal counter
        if nid in seen or nid not in by_id:
            return
        seen.add(nid)
        for child in children_ids(by_id[nid]):
            visit(child)
        by_id[nid].evaluation_order = counter
        counter += 1

    visit(output.root_id)
    return counter


def tree_metrics(output: SemanticOutput) -> dict:
    by_id = {n.id: n for n in output.nodes}
    depth_cache: dict[str, int] = {}

    def depth(nid: str, trail: frozenset = frozenset()) -> int:
        if nid in trail or nid not in by_id:
            return 0
        if nid not in depth_cache:
            kids = children_ids(by_id[nid])
            depth_cache[nid] = 1 + max((depth(k, trail | {nid}) for k in kids), default=0)
        return depth_cache[nid]

    return {
        "nodes": len(output.nodes),
        "by_node_type": dict(Counter(n.node_type for n in output.nodes)),
        "ambiguous_nodes": sum(1 for n in output.nodes if n.ambiguous),
        "review_nodes": sum(1 for n in output.nodes if n.review_required),
        "context_dependent_nodes": sum(1 for n in output.nodes if n.evidence == "context_dependent"),
        "depth": depth(output.root_id),
        "evaluation_order_assigned": all(n.evaluation_order is not None for n in output.nodes),
    }


def format_tree(output: SemanticOutput) -> list[str]:
    """Lineas de texto con el arbol anidado (para la CLI)."""
    by_id = {n.id: n for n in output.nodes}
    lines: list[str] = []

    def walk(nid: str, role: str, indent: int, trail: frozenset) -> None:
        node = by_id.get(nid)
        if node is None or nid in trail:
            lines.append("  " * indent + f"- ({role}) [{nid}] <referencia invalida>")
            return
        label = node.intent or "?" + "|".join(node.candidate_intents)
        if node.ambiguous:
            label = "AMBIGUO{" + ", ".join(node.candidate_intents) + "}"
        extra = []
        if node.symbol:
            extra.append(f"'{node.symbol}'")
        if node.implicit:
            extra.append("implicito")
        if node.properties:
            extra.append(",".join(f"{k}={v}" for k, v in node.properties.items()))
        if node.review_required:
            extra.append("REVISAR")
        order = f" #{node.evaluation_order}" if node.evaluation_order is not None else ""
        lines.append("  " * indent + f"- ({role}) [{nid}] {label} {' '.join(extra)}{order}".rstrip())
        for op in node.operands:
            walk(op.node_id, op.role, indent + 1, trail | {nid})

    walk(output.root_id, "root", 0, frozenset())
    return lines
