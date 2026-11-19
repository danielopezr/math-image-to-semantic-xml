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

from .annotator import DEFAULT_INTENTS, FUNCTION_CONCEPTS, SELF_READING_CONCEPTS
from .concepts import ConceptRegistry, get_registry
from .dom import (
    drop_attribute,
    element_children,
    element_text,
    has_positional_children,
    is_element,
    localname,
    new_element,
    path,
)
from .normalizer import strip_reasoning

ID_ATTR = "id"

# Single source of truth for argument role names.  The concept cards shown to
# the language model and the deterministic JSON application both read it.
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
    "beta-function": ("arg1", "arg2"),
    "error-function": ("arg",),
    "max": ("arg",),
    "min": ("arg",),
    "matrix": ("arg",),
    "set": ("arg",),
    "inner-product": ("arg1", "arg2"),
    "dot-product": ("arg1", "arg2"),
    "cross-product": ("arg1", "arg2"),
    "outer-product": ("arg1", "arg2"),
    "cartesian-product": ("arg1", "arg2"),
    "ordered-pair": ("arg1", "arg2"),
    "coordinate": ("arg1", "arg2"),
    "polar-coordinate": ("arg1", "arg2"),
    "spherical-coordinate": ("arg1", "arg2", "arg3"),
    "cartesian-coordinate": ("arg1", "arg2"),
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
            drop_attribute(element, name)


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
            drop_attribute(element, ID_ATTR)


def semantic_tree_markup(root: etree._Element) -> str:
    return etree.tostring(root, encoding="unicode", pretty_print=True).strip()


def _is_payload(data: Any) -> bool:
    if isinstance(data, list):
        return all(isinstance(item, dict) for item in data)
    return isinstance(data, dict) and ("annotations" in data or "concept" in data)


def _decode_json_payload(text: str) -> Any:
    """Decode the JSON payload embedded in ``text``.

    The whole text is tried first.  Otherwise every ``{`` or ``[`` is tried as
    the start of a JSON value with :meth:`json.JSONDecoder.raw_decode`, so
    surrounding prose, top-level lists and stray braces do not break parsing.
    """

    try:
        return json.loads(text)
    except ValueError as exc:
        first_error = exc

    decoder = json.JSONDecoder()
    fallback: Any = None
    for index, character in enumerate(text):
        if character not in "{[":
            continue
        try:
            data, _ = decoder.raw_decode(text, index)
        except ValueError:
            continue
        if _is_payload(data):
            return data
        if fallback is None and isinstance(data, (dict, list)):
            fallback = data
    if fallback is not None:
        return fallback
    raise SemanticError(f"Invalid JSON: {first_error}") from first_error


def parse_semantic_payload(raw: str) -> list[dict[str, Any]]:
    """Extract the annotations list from a model answer."""

    text = strip_reasoning(raw or "").strip()
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    data = _decode_json_payload(text)

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
    if isinstance(args, dict) and "literals" in args:
        # Small models nest the literals inside args ({"a1": "n5", "literals":
        # {"a2": "2"}}).  A dict is moved where it belongs; anything else
        # (a bare slot name such as "a2") carries no value and is dropped.
        args = dict(args)
        nested = args.pop("literals")
        if isinstance(nested, dict):
            literals = {**nested, **(literals if isinstance(literals, dict) else {})}
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
    seen_targets: dict[int, etree._Element],
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
    seen_targets[id(target)] = target
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
    if has_positional_children(parent):
        raise SemanticError(
            f"Cannot group the children of <{localname(parent.tag)}>: each child "
            "has a fixed role (numerator, base, exponent, ...). Target the "
            f"<{localname(parent.tag)}> element itself instead"
        )
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


# Concepts whose Presentation MathML form is a specific element.  Putting them
# on another element produces a wrong reading, e.g. "power" on the mfrac of
# d/dx reads "d to the power dx".
_STRUCTURAL_TARGETS: dict[str, frozenset[str]] = {
    # An msubsup also has a subscript: annotating it as power($base,$exp)
    # would drop the index (v_1^2 read as "v squared").
    "power": frozenset({"msup"}),
    "indexed-by": frozenset({"msub", "msubsup"}),
    "fraction": frozenset({"mfrac", "mrow"}),
    "root": frozenset({"msqrt", "mroot"}),
}


def _check_target_kind(concept: str, target: etree._Element) -> None:
    allowed = _STRUCTURAL_TARGETS.get(concept)
    if allowed is None:
        return
    tag = localname(target.tag)
    if tag not in allowed:
        expected = " or ".join(f"<{name}>" for name in sorted(allowed))
        raise SemanticError(f"Concept {concept!r} must target {expected}, not <{tag}>")


def _snapshot(root: etree._Element) -> list[tuple[etree._Element, dict, list]]:
    return [
        (element, dict(element.attrib), list(element))
        for element in root.iter()
        if is_element(element)
    ]


def _restore(snapshot: list[tuple[etree._Element, dict, list]]) -> None:
    """Undo every change made after ``snapshot`` (attributes and wrappers)."""

    for element, attributes, children in snapshot:
        element.attrib.clear()
        element.attrib.update(attributes)
        element[:] = children


def _literal_is_valid(value: str) -> bool:
    return bool(_NUMBER_RE.match(value) or _NCNAME_RE.match(value))


_OPEN_FENCES = frozenset({"(", "[", "{"})
_CLOSE_FENCES = frozenset({")", "]", "}"})


def _argument_nodes(
    annotation: dict[str, Any], id_map: dict[str, etree._Element]
) -> list[etree._Element] | None:
    """Elements named as arguments, or ``None`` if any id is unknown."""

    args = annotation.get("args", annotation.get("roles"))
    values: list[Any] = []
    if isinstance(args, dict):
        values = [value for key, value in args.items() if key != "literals"]
    elif isinstance(args, list):
        values = list(args)
    groups = annotation.get("groups", annotation.get("role_groups"))
    if isinstance(groups, dict):
        values += list(groups.values())
    nodes: list[etree._Element] = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("ref", value.get("id", value.get("group")))
        ids = [value] if isinstance(value, str) else value
        if not isinstance(ids, list):
            return None
        for node_id in ids:
            node = id_map.get(str(node_id))
            if node is None:
                return None
            nodes.append(node)
    return nodes


def _fence(element: etree._Element) -> str:
    return element_text(element) if localname(element.tag) == "mo" else ""


def _repair_sibling_arguments(
    annotation: dict[str, Any], id_map: dict[str, etree._Element]
) -> dict[str, Any]:
    """Retarget a prefix function whose arguments follow it as siblings.

    The most frequent rejection (18/11/2026: 182 of 357) is an annotation on
    the function name (``sin``, ``Γ``, ``f``) with its argument taken from the
    element beside it, which is not inside the target.  When everything from
    the name to the last argument (plus the fences that close it) is only the
    name, the arguments and fences, the annotation is moved to the group of
    those siblings, which is what the LLM meant.  Anything else is returned
    unchanged and fails validation as before.
    """

    target_id = annotation.get("target", annotation.get("target_id"))
    if not isinstance(target_id, str) or "wrap" in annotation:
        return annotation
    target = id_map.get(target_id)
    nodes = _argument_nodes(annotation, id_map)
    if target is None or not nodes:
        return annotation
    outside = [node for node in nodes if not _is_descendant(node, target)]
    if not outside:
        return annotation
    parent = target.getparent()
    if parent is None or has_positional_children(parent):
        return annotation
    concept = str(annotation.get("concept") or "").strip()
    if localname(target.tag) == "mo":
        # Operators such as sup, adj or lim: the labels leave them alone.
        return annotation
    if target in outside and concept not in _GENERIC_CONCEPTS:
        # Only a generic application names its function as an argument.
        return annotation

    siblings = element_children(parent)
    start = siblings.index(target)
    covered: set[int] = {start}
    for node in outside:
        ancestor: etree._Element | None = node
        while ancestor is not None and ancestor.getparent() is not parent:
            ancestor = ancestor.getparent()
        if ancestor is None:
            return annotation
        covered.add(siblings.index(ancestor))
    if min(covered) < start:
        return annotation
    end = max(covered)

    depth = 0
    for element in siblings[start + 1 : end + 1]:
        text = _fence(element)
        depth += text in _OPEN_FENCES
        depth -= text in _CLOSE_FENCES
    while (
        depth > 0
        and end + 1 < len(siblings)
        and _fence(siblings[end + 1]) in _CLOSE_FENCES
    ):
        end += 1
        depth -= 1
    if depth != 0:
        return annotation
    for index in range(start + 1, end + 1):
        text = _fence(siblings[index])
        if index not in covered and text not in _OPEN_FENCES | _CLOSE_FENCES:
            return annotation

    span = siblings[start : end + 1]
    repaired = {
        key: value
        for key, value in annotation.items()
        if key not in {"target", "target_id"}
    }
    parent_id = parent.get(ID_ATTR)
    if len(span) == len(siblings) and parent_id and localname(parent.tag) == "mrow":
        repaired["target"] = parent_id
        return repaired
    span_ids = [element.get(ID_ATTR) for element in span]
    if not all(span_ids):
        return annotation
    repaired["wrap"] = span_ids
    return repaired


def _redundant_reason(
    annotation: dict[str, Any], id_map: dict[str, etree._Element]
) -> str | None:
    """Why an annotation would be removed from the output anyway, or ``None``.

    Intents that only repeat the default reading (``root`` on ``msqrt``) or
    that the guide leaves out (sums, integrals, limits) are stripped from the
    final output; validating them only produced rejections (18/11/2026: 81
    for ``root``, 32 for ``sum`` and ``definite-integral``).
    """

    concept = str(annotation.get("concept") or "").strip()
    if concept in SELF_READING_CONCEPTS:
        return f"{concept!r} reads well on its own; left unannotated"
    target_id = _annotation_target_id(annotation)
    target = id_map.get(target_id) if target_id else None
    if target is not None and DEFAULT_INTENTS.get(localname(target.tag)) == concept:
        return (
            f"{concept!r} is the default reading of <{localname(target.tag)}>; "
            "left unannotated"
        )
    return None


def _apply_annotation(
    annotation: dict[str, Any],
    id_map: dict[str, etree._Element],
    seen_targets: dict[int, etree._Element],
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

    annotation = _repair_sibling_arguments(annotation, id_map)
    target = _resolve_target(annotation, id_map, seen_targets)
    _check_target_kind(concept, target)
    slots = _collect_slots(annotation, concept)
    arity = len(slots)

    if concept == "generic-application":
        if arity != 2:
            raise SemanticError("generic-application needs exactly two arguments")
    elif not registry.accepts_arity(concept, arity):
        raise SemanticError(f"Concept {concept!r} does not accept {arity} argument(s)")

    if set(slots) != set(range(arity)):
        raise SemanticError("Argument slots must be contiguous starting at a1")

    if concept == "power" and slots.get(0, ("", None))[0] == "ref":
        base = id_map.get(str(slots[0][1]))
        if base is not None and element_text(base) in FUNCTION_CONCEPTS:
            raise SemanticError(
                f"{element_text(base)}^n is not a power of the symbol "
                f"{element_text(base)!r}; leave it unannotated"
            )

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


def semantic_json_schema(concept_names: list[str] | None = None) -> dict[str, Any]:
    """JSON Schema of the semantic contract, for constrained decoding.

    Local servers such as Ollama use it to force a syntactically valid answer.
    When ``concept_names`` is given, ``concept`` is restricted to those names.

    The schema enforces the parts of the contract that small models most often
    get wrong:

    * every annotation names its element, with ``target`` (an existing id) or
      ``wrap`` (sibling ids to group), never both and never neither;
    * an argument that spans several siblings is written as a list inside
      ``args`` (``"a2": ["n6", "n7"]``) instead of a separate ``groups``
      object, so the same slot cannot be defined twice;
    * no other keys are allowed.

    The parser still accepts ``groups`` for answers produced without the
    schema.
    """

    concept: dict[str, Any] = {"type": "string"}
    if concept_names:
        concept["enum"] = sorted(set(concept_names) | {"generic-application"})
    node_id = {"type": "string"}
    id_list = {"type": "array", "items": node_id, "minItems": 1}
    args = {
        "type": "object",
        "additionalProperties": {"anyOf": [node_id, id_list]},
    }
    literals = {"type": "object", "additionalProperties": {"type": "string"}}
    by_target = {
        "type": "object",
        "properties": {
            "target": node_id,
            "concept": concept,
            "args": args,
            "literals": literals,
        },
        "required": ["target", "concept"],
        "additionalProperties": False,
    }
    by_wrap = {
        "type": "object",
        "properties": {
            "wrap": {**id_list, "minItems": 2},
            "new_id": node_id,
            "concept": concept,
            "args": args,
            "literals": literals,
        },
        "required": ["wrap", "concept"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "version": {"type": "integer"},
            "annotations": {
                "type": "array",
                "items": {"anyOf": [by_target, by_wrap]},
            },
        },
        "required": ["annotations"],
        "additionalProperties": False,
    }


def apply_semantic_payload(
    root: etree._Element,
    annotations: list[dict[str, Any]],
    *,
    registry: ConceptRegistry | None = None,
    allow_open_concepts: bool = True,
    allowed_open_concepts: set[str] | None = None,
    drop_redundant: bool = False,
) -> SemanticApplyResult:
    """Apply a validated semantic JSON description to a tree with ids.

    With ``drop_redundant`` the annotations that the output would lose anyway
    (see :func:`_redundant_reason`) are skipped with a warning.
    """

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
    # id() -> element: holding the element keeps its id() from being reused.
    seen_targets: dict[int, etree._Element] = {}

    for index, annotation in enumerate(annotations, 1):
        target_id = _annotation_target_id(annotation)
        if target_id in duplicate_targets:
            result.warnings.append(
                f"annotation {index}: target {target_id!r} has multiple "
                "annotations; left unannotated"
            )
            continue
        reason = _redundant_reason(annotation, id_map) if drop_redundant else None
        if reason is not None:
            result.warnings.append(f"annotation {index}: {reason}")
            continue
        # Each annotation is applied as a transaction: a failing annotation
        # is rolled back completely, so the valid ones can still be used.
        snapshot = _snapshot(root)
        saved_ids = dict(id_map)
        saved_targets = dict(seen_targets)
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
            _restore(snapshot)
            id_map.clear()
            id_map.update(saved_ids)
            seen_targets.clear()
            seen_targets.update(saved_targets)
            result.errors.append(f"annotation {index}: {exc}")

    return result
