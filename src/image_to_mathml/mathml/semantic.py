"""Semantic JSON contract for LLM intent annotation.

The language model never rewrites MathML.  It receives a clean Presentation
MathML tree with temporary ``id`` attributes and returns a JSON description of
the concepts and argument positions it recognises.  This module validates that
description and applies it deterministically to the tree.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from lxml import etree

from .concepts import ConceptRegistry, get_registry
from .dom import element_children, is_element, new_element, path

ID_ATTR = "id"

CONCEPT_ROLES: dict[str, tuple[str, ...]] = {
    "fraction": ("num", "den"),
    "power": ("base", "exp"),
    "root": ("radicand", "index"),
    "indexed-by": ("base", "sub"),
    "exponential": ("exp",),
    "sine": ("arg",),
    "cosine": ("arg",),
    "tangent": ("arg",),
    "secant": ("arg",),
    "cosecant": ("arg",),
    "cotangent": ("arg",),
    "arcsine": ("arg",),
    "arccosine": ("arg",),
    "arctangent": ("arg",),
    "arcsecant": ("arg",),
    "arccosecant": ("arg",),
    "arccotangent": ("arg",),
    "hyperbolic-sine": ("arg",),
    "hyperbolic-cosine": ("arg",),
    "hyperbolic-tangent": ("arg",),
    "hyperbolic-secant": ("arg",),
    "hyperbolic-cosecant": ("arg",),
    "hyperbolic-cotangent": ("arg",),
    "natural-logarithm": ("arg",),
    "logarithm": ("arg", "base"),
    "absolute-value": ("arg",),
    "floor": ("arg",),
    "ceiling": ("arg",),
    "factorial": ("arg",),
    "norm": ("arg",),
    "magnitude": ("arg",),
    "determinant": ("arg",),
    "transpose": ("arg",),
    "real-part": ("arg",),
    "imaginary-part": ("arg",),
    "complex-conjugate": ("arg",),
    "inverse": ("arg",),
    "gamma-function": ("arg",),
    "beta-function": ("arg",),
    "error-function": ("arg",),
    "plus": ("arg1", "arg2"),
    "minus": ("arg1", "arg2"),
    "times": ("arg1", "arg2"),
    "divided-by": ("arg1", "arg2"),
    "equals": ("arg1", "arg2"),
    "not-equal-to": ("arg1", "arg2"),
    "less-than": ("arg1", "arg2"),
    "greater-than": ("arg1", "arg2"),
    "less-than-or-equal-to": ("arg1", "arg2"),
    "greater-than-or-equal-to": ("arg1", "arg2"),
    "union": ("arg1", "arg2"),
    "intersection": ("arg1", "arg2"),
    "element-of": ("arg1", "arg2"),
    "member-of": ("arg1", "arg2"),
    "subset": ("arg1", "arg2"),
    "superset": ("arg1", "arg2"),
    "for-all": ("arg",),
    "there-exists": ("arg",),
    "there-does-not-exist": ("arg",),
    "sum": ("arg1", "arg2", "arg3"),
    "product": ("arg1", "arg2", "arg3"),
    "definite-integral": ("arg1", "arg2"),
    "derivative": ("arg1", "arg2", "arg3"),
    "partial-derivative": ("arg1", "arg2", "arg3"),
    "limit": ("arg",),
}

_GENERIC_CONCEPTS = {
    "generic",
    "generic-application",
    "generic_function",
    "$name",
    "$name($var)",
}
_SLOT_RE = re.compile(r"^(?:a|arg)(\d+)$", re.IGNORECASE)
_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_NCNAME_RE = re.compile(r"^[A-Za-z_\u0080-\uffff][A-Za-z0-9._\-\u0080-\uffff]*$")
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class SemanticError(ValueError):
    """Raised when a semantic JSON annotation cannot be applied safely."""


@dataclass
class SemanticApplyResult:
    applied: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def strip_semantic_attributes(root: etree._Element) -> None:
    """Remove intent, arg and temporary id attributes from a clean skeleton."""

    for element in root.iter():
        if not is_element(element):
            continue
        for name in ("intent", "arg", ID_ATTR):
            element.attrib.pop(name, None)


def assign_node_ids(root: etree._Element) -> dict[str, etree._Element]:
    """Assign short temporary ids and return the id-to-element map."""

    id_map: dict[str, etree._Element] = {}
    counter = 0
    for element in root.iter():
        if not is_element(element):
            continue
        counter += 1
        node_id = f"n{counter}"
        element.set(ID_ATTR, node_id)
        id_map[node_id] = element
    return id_map


def strip_node_ids(root: etree._Element) -> None:
    for element in root.iter():
        if is_element(element):
            element.attrib.pop(ID_ATTR, None)


def semantic_tree_markup(root: etree._Element) -> str:
    return etree.tostring(root, encoding="unicode", pretty_print=True).strip()


def parse_semantic_payload(raw: str) -> list[dict[str, Any]]:
    """Extract the annotations list from a model answer."""

    text = (raw or "").strip()
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    else:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            text = text[start : end + 1]

    try:
        data = json.loads(text)
    except ValueError as exc:
        raise SemanticError(f"Invalid JSON: {exc}") from exc

    if isinstance(data, dict):
        if "annotations" in data:
            annotations = data["annotations"]
        elif "concept" in data:
            annotations = [data]
        else:
            raise SemanticError("The JSON object must contain an 'annotations' list")
    elif isinstance(data, list):
        annotations = data
    else:
        raise SemanticError("The JSON answer must be an object or a list")

    if not isinstance(annotations, list):
        raise SemanticError("'annotations' must be a list")
    for index, annotation in enumerate(annotations, 1):
        if not isinstance(annotation, dict):
            raise SemanticError(f"Annotation {index} must be a JSON object")
    return annotations


def _coerce_arg(value: Any) -> tuple[str, Any]:
    if isinstance(value, str):
        return "ref", value
    if isinstance(value, list):
        return "group", _as_id_list(value)
    if isinstance(value, dict):
        if "ref" in value:
            return "ref", str(value["ref"])
        if "id" in value:
            return "ref", str(value["id"])
        if "group" in value:
            return "group", _as_id_list(value["group"])
        if "wrap" in value:
            return "group", _as_id_list(value["wrap"])
        if "literal" in value:
            return "literal", str(value["literal"])
    raise SemanticError(f"Invalid argument value {value!r}")


def _as_id_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, str) for item in value)
    ):
        return list(value)
    raise SemanticError(f"Invalid group value {value!r}")


def _concept_roles(concept: str, arity: int) -> tuple[str, ...]:
    if concept == "generic-application":
        return ("name", "var")
    roles = CONCEPT_ROLES.get(concept)
    if roles is not None and len(roles) >= arity:
        return roles[:arity]
    return tuple(f"arg{index}" for index in range(1, arity + 1))


def _slot_index(key: Any, concept: str) -> int:
    text = str(key).strip()
    match = _SLOT_RE.match(text)
    if match:
        return int(match.group(1)) - 1
    if text.isdigit():
        return int(text) - 1
    roles = _concept_roles(concept, len(CONCEPT_ROLES.get(concept, ())))
    if text in roles:
        return roles.index(text)
    raise SemanticError(f"Unknown argument slot {text!r} for concept {concept!r}")


def _collect_slots(
    annotation: dict[str, Any], concept: str
) -> dict[int, tuple[str, Any]]:
    args = annotation.get("args", annotation.get("roles"))
    groups = annotation.get("groups", annotation.get("role_groups"))
    literals = annotation.get("literals", annotation.get("role_literals"))
    slots: dict[int, tuple[str, Any]] = {}

    def add(index: int, kind: str, value: Any) -> None:
        if index < 0:
            raise SemanticError(f"Invalid argument slot index {index + 1}")
        if index in slots:
            raise SemanticError(f"Argument slot a{index + 1} is defined more than once")
        slots[index] = (kind, value)

    if args is None:
        args = {}
    if isinstance(args, list):
        for index, value in enumerate(args):
            add(index, *_coerce_arg(value))
    elif isinstance(args, dict):
        for key, value in args.items():
            add(_slot_index(key, concept), *_coerce_arg(value))
    else:
        raise SemanticError("'args' must be an object or a list")

    if groups is not None:
        if not isinstance(groups, dict):
            raise SemanticError("'groups' must be an object")
        for key, value in groups.items():
            add(_slot_index(key, concept), "group", _as_id_list(value))

    if literals is not None:
        if not isinstance(literals, dict):
            raise SemanticError("'literals' must be an object")
        for key, value in literals.items():
            add(_slot_index(key, concept), "literal", str(value))

    return slots


def _check_concept(
    concept: str,
    registry: ConceptRegistry,
    allow_open_concepts: bool,
    allowed_open_concepts: set[str] | None,
) -> None:
    if concept == "generic-application":
        return
    classification = registry.classification(concept)
    if classification == "unknown":
        raise SemanticError(f"Unknown intent concept {concept!r}")
    if classification == "open":
        if not allow_open_concepts:
            raise SemanticError(f"{concept!r} is an open concept and is not allowed")
        if allowed_open_concepts is not None and concept not in allowed_open_concepts:
            allowed = ", ".join(sorted(allowed_open_concepts)) or "none"
            raise SemanticError(
                f"{concept!r} is an open concept; allowed open concepts: {allowed}"
            )


def _resolve_target(
    annotation: dict[str, Any],
    id_map: dict[str, etree._Element],
    seen_targets: set[int],
) -> etree._Element:
    target_value = annotation.get("target", annotation.get("target_id"))
    wrap_value = annotation.get("wrap", annotation.get("target_wrap"))
    new_id = annotation.get("new_id")

    if target_value is not None and wrap_value is not None:
        raise SemanticError("Use either 'target' or 'wrap', not both")

    if target_value is not None:
        if isinstance(target_value, dict):
            if "ref" in target_value:
                target_value = target_value["ref"]
            elif "id" in target_value:
                target_value = target_value["id"]
            elif "wrap" in target_value:
                wrap_value = target_value["wrap"]
                target_value = None
            else:
                raise SemanticError(f"Invalid target value {target_value!r}")
        if target_value is not None:
            node_id = str(target_value)
            target = id_map.get(node_id)
            if target is None:
                raise SemanticError(f"Unknown target id {node_id!r}")
            if new_id is not None:
                raise SemanticError("'new_id' is only valid with a 'wrap' target")
        else:
            target = None
    else:
        target = None

    if target is None:
        if wrap_value is None:
            raise SemanticError("Each annotation needs a 'target' or a 'wrap' list")
        ids = _as_id_list(wrap_value)
        if len(ids) == 1:
            target = id_map.get(ids[0])
            if target is None:
                raise SemanticError(f"Unknown target id {ids[0]!r}")
            if new_id is not None:
                generated = str(new_id)
                if generated in id_map:
                    raise SemanticError(f"Duplicate wrapper id {generated!r}")
                id_map[generated] = target
        else:
            parent, nodes = _contiguous_nodes(ids, id_map)
            target = _wrap_nodes(parent, nodes)
            if new_id is not None:
                generated = str(new_id)
                if generated in id_map:
                    raise SemanticError(f"Duplicate wrapper id {generated!r}")
                target.set(ID_ATTR, generated)
                id_map[generated] = target

    if id(target) in seen_targets:
        raise SemanticError("The same target is annotated more than once")
    if target.get("intent") is not None:
        raise SemanticError(f"Target {path(target)} already has an intent")
    seen_targets.add(id(target))
    return target


def _contiguous_nodes(
    ids: list[str], id_map: dict[str, etree._Element]
) -> tuple[etree._Element, list[etree._Element]]:
    nodes: list[etree._Element] = []
    for node_id in ids:
        node = id_map.get(node_id)
        if node is None:
            raise SemanticError(f"Unknown node id {node_id!r}")
        nodes.append(node)

    parents = {node.getparent() for node in nodes}
    if len(parents) != 1 or None in parents:
        raise SemanticError("Grouped nodes must share the same parent")
    parent = nodes[0].getparent()
    assert parent is not None
    siblings = element_children(parent)
    try:
        indexed = sorted(
            ((siblings.index(node), node) for node in nodes),
            key=lambda item: item[0],
        )
    except ValueError as exc:
        raise SemanticError("Grouped nodes must be element siblings") from exc
    indexes = [index for index, _ in indexed]
    first = indexes[0]
    if indexes != list(range(first, first + len(indexes))):
        raise SemanticError("Grouped nodes must be contiguous siblings")
    return parent, [node for _, node in indexed]


def _wrap_nodes(
    parent: etree._Element,
    nodes: list[etree._Element],
    *,
    arg_name: str | None = None,
) -> etree._Element:
    wrapper = new_element("mrow")
    if arg_name is not None:
        wrapper.set("arg", arg_name)
    insert_at = parent.index(nodes[0])
    parent.insert(insert_at, wrapper)
    for node in nodes:
        wrapper.append(node)
    return wrapper


def _is_descendant(node: etree._Element, ancestor: etree._Element) -> bool:
    current = node.getparent()
    while current is not None:
        if current is ancestor:
            return True
        current = current.getparent()
    return False


def _set_arg(element: etree._Element, name: str) -> None:
    existing = element.get("arg")
    if existing is not None and existing != name:
        raise SemanticError(
            f"Node {path(element)} already has arg={existing!r}, cannot use {name!r}"
        )
    element.set("arg", name)


def _literal_is_valid(value: str) -> bool:
    return bool(_NUMBER_RE.match(value) or _NCNAME_RE.match(value))


def _apply_annotation(
    annotation: dict[str, Any],
    id_map: dict[str, etree._Element],
    seen_targets: set[int],
    registry: ConceptRegistry,
    allow_open_concepts: bool,
    allowed_open_concepts: set[str] | None,
) -> None:
    concept_value = annotation.get("concept")
    if concept_value is None:
        raise SemanticError("Each annotation needs a 'concept'")
    concept = str(concept_value).strip()
    if concept in _GENERIC_CONCEPTS:
        concept = "generic-application"
    _check_concept(concept, registry, allow_open_concepts, allowed_open_concepts)

    target = _resolve_target(annotation, id_map, seen_targets)
    slots = _collect_slots(annotation, concept)
    arity = len(slots)

    if concept == "generic-application":
        if arity != 2:
            raise SemanticError("generic-application needs exactly two arguments")
    elif not registry.accepts_arity(concept, arity):
        raise SemanticError(f"Concept {concept!r} does not accept {arity} argument(s)")

    if set(slots) != set(range(arity)):
        raise SemanticError("Argument slots must be contiguous starting at a1")

    role_names = _concept_roles(concept, arity)
    intent_arguments: list[str] = []
    has_literal = False

    for index in range(arity):
        kind, value = slots[index]
        role = role_names[index]
        if kind == "ref":
            node = id_map.get(str(value))
            if node is None:
                raise SemanticError(f"Unknown argument id {value!r}")
            if not _is_descendant(node, target):
                raise SemanticError(
                    f"Argument id {value!r} is not a descendant of the target"
                )
            _set_arg(node, role)
            intent_arguments.append(f"${role}")
        elif kind == "group":
            ids = _as_id_list(value)
            if len(ids) == 1:
                node = id_map.get(ids[0])
                if node is None:
                    raise SemanticError(f"Unknown argument id {ids[0]!r}")
                if not _is_descendant(node, target):
                    raise SemanticError(
                        f"Argument id {ids[0]!r} is not a descendant of the target"
                    )
                _set_arg(node, role)
            else:
                parent, nodes = _contiguous_nodes(ids, id_map)
                if any(not _is_descendant(node, target) for node in nodes):
                    raise SemanticError(
                        "Grouped argument ids must be descendants of the target"
                    )
                _wrap_nodes(parent, nodes, arg_name=role)
            intent_arguments.append(f"${role}")
        else:
            literal = str(value)
            if not _literal_is_valid(literal):
                raise SemanticError(f"Invalid intent literal {literal!r}")
            intent_arguments.append(literal)
            has_literal = True

    if concept == "generic-application":
        intent = "$name($var)"
    elif intent_arguments:
        separator = ", " if has_literal else ","
        intent = f"{concept}({separator.join(intent_arguments)})"
    else:
        intent = concept
    target.set("intent", intent)


def _annotation_target_id(annotation: dict[str, Any]) -> str | None:
    value = annotation.get("target", annotation.get("target_id"))
    if isinstance(value, dict):
        value = value.get("ref", value.get("id"))
    if isinstance(value, str):
        return value
    wrap = annotation.get("wrap", annotation.get("target_wrap"))
    if isinstance(wrap, list) and len(wrap) == 1 and isinstance(wrap[0], str):
        return wrap[0]
    return None


def apply_semantic_payload(
    root: etree._Element,
    annotations: list[dict[str, Any]],
    *,
    registry: ConceptRegistry | None = None,
    allow_open_concepts: bool = True,
    allowed_open_concepts: set[str] | None = None,
) -> SemanticApplyResult:
    """Apply a validated semantic JSON description to a tree with ids."""

    registry = registry or get_registry()
    result = SemanticApplyResult()
    id_map: dict[str, etree._Element] = {
        str(element.get(ID_ATTR)): element
        for element in root.iter()
        if is_element(element) and element.get(ID_ATTR)
    }
    target_counts: dict[str, int] = {}
    for annotation in annotations:
        target_id = _annotation_target_id(annotation)
        if target_id is not None:
            target_counts[target_id] = target_counts.get(target_id, 0) + 1
    duplicate_targets = {
        target_id
        for target_id, count in target_counts.items()
        if count > 1 and target_id in id_map
    }
    seen_targets: set[int] = set()

    for index, annotation in enumerate(annotations, 1):
        target_id = _annotation_target_id(annotation)
        if target_id in duplicate_targets:
            result.warnings.append(
                f"annotation {index}: target {target_id!r} has multiple "
                "annotations; left unannotated"
            )
            continue
        try:
            _apply_annotation(
                annotation,
                id_map,
                seen_targets,
                registry,
                allow_open_concepts,
                allowed_open_concepts,
            )
            result.applied += 1
        except SemanticError as exc:
            result.errors.append(f"annotation {index}: {exc}")

    return result
