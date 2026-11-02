"""Deterministic MathML normalisation.

The normaliser is intentionally independent from the language model.  It is
the last line of defence that guarantees the structural contract requested by
the accessibility pipeline:

* one ``<math>`` root with the MathML namespace,
* no ``display`` attribute on the root,
* exactly one top-level ``<mrow>`` wrapping the whole expression,
* no redundant single-child ``<mrow>`` elements.
"""

from __future__ import annotations

import html.entities
import re

from lxml import etree

from .dom import (
    MATHML_NS,
    element_children,
    is_element,
    localname,
    qname,
)


class NormalizationError(ValueError):
    """Raised when the model output cannot be parsed as MathML."""


_FENCE_RE = re.compile(r"```[A-Za-z0-9_-]*\s*", re.MULTILINE)
_MATH_RE = re.compile(r"<math\b[^>]*>.*?</math>", re.DOTALL | re.IGNORECASE)
_MROW_RE = re.compile(r"<mrow\b[^>]*>.*?</mrow>", re.DOTALL | re.IGNORECASE)
_ANY_ELEMENT_RE = re.compile(r"<([A-Za-z][\w.-]*)\b[^>]*>.*?</\1>", re.DOTALL)
_NAMED_ENTITY_RE = re.compile(r"&([A-Za-z][A-Za-z0-9]+);")


def _sanitize_entities(markup: str) -> str:
    keep = {"amp", "lt", "gt", "quot", "apos"}

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in keep:
            return match.group(0)
        codepoint = html.entities.name2codepoint.get(name)
        if codepoint is None:
            return match.group(0)
        return f"&#{codepoint};"

    return _NAMED_ENTITY_RE.sub(replace, markup)


def extract_math_markup(text: str) -> str:
    """Extract a MathML document from an LLM answer.

    Models sometimes surround the markup with Markdown fences or brief prose.
    """

    cleaned = _FENCE_RE.sub("", text or "").replace("```", "").strip()
    match = _MATH_RE.search(cleaned)
    if match:
        return match.group(0)
    match = _MROW_RE.search(cleaned)
    if match:
        return f"<math>{match.group(0)}</math>"
    match = _ANY_ELEMENT_RE.search(cleaned)
    if match:
        return f"<math><mrow>{match.group(0)}</mrow></math>"
    raise NormalizationError("The response does not contain a MathML element")


def _copy_attributes(source: etree._Element, target: etree._Element) -> None:
    for key, value in source.attrib.items():
        name = localname(key)
        if (
            name.startswith("xmlns")
            or name == "display"
            and localname(source.tag) == "math"
        ):
            continue
        target.set(name, value)


def _clone_with_namespace(source: etree._Element, parent: etree._Element | None = None):
    tag = localname(source.tag)
    if parent is None:
        target = etree.Element(qname(tag), nsmap={None: MATHML_NS})
    else:
        target = etree.SubElement(parent, qname(tag))
    _copy_attributes(source, target)
    target.text = source.text
    target.tail = source.tail
    for child in source:
        if is_element(child):
            _clone_with_namespace(child, target)
    return target


def _ensure_math_root(root: etree._Element) -> etree._Element:
    if localname(root.tag) == "math":
        return root
    math = etree.Element(qname("math"), nsmap={None: MATHML_NS})
    mrow = etree.SubElement(math, qname("mrow"))
    mrow.append(root)
    return math


def _ensure_single_mrow(math: etree._Element) -> etree._Element:
    for key in list(math.attrib):
        if localname(key) == "display":
            del math.attrib[key]

    children = element_children(math)
    if len(children) == 1 and localname(children[0].tag) == "mrow":
        return children[0]

    mrow = etree.Element(qname("mrow"))
    if math.text and math.text.strip():
        mrow.text = math.text
    math.text = None
    for child in list(math):
        mrow.append(child)
    math.append(mrow)
    return mrow


def _simplify(element: etree._Element, *, is_root_mrow: bool = False) -> None:
    for child in list(element):
        if is_element(child):
            _simplify(child)

    if localname(element.tag) != "mrow" or is_root_mrow or element.attrib:
        return

    children = element_children(element)
    parent = element.getparent()
    if parent is None:
        return

    if not children:
        parent.remove(element)
        return

    if len(children) == 1 and not (element.text or "").strip():
        child = children[0]
        index = parent.index(element)
        if element.tail:
            child.tail = (child.tail or "") + element.tail
        parent.remove(element)
        parent.insert(index, child)


def parse_and_normalize(markup: str) -> etree._Element:
    """Parse model output and return the normalised MathML tree."""

    candidate = extract_math_markup(markup)
    candidate = _sanitize_entities(candidate)
    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        remove_comments=False,
        huge_tree=True,
    )
    try:
        parsed = etree.fromstring(candidate.encode("utf-8"), parser)
    except etree.XMLSyntaxError as exc:
        raise NormalizationError(f"Invalid MathML: {exc}") from exc

    root = _clone_with_namespace(parsed)
    root = _ensure_math_root(root)
    mrow = _ensure_single_mrow(root)
    for child in element_children(mrow):
        _simplify(child)
    _simplify(mrow, is_root_mrow=True)

    children = element_children(root)
    if len(children) != 1 or localname(children[0].tag) != "mrow":
        raise NormalizationError("The MathML root must contain exactly one <mrow>")
    return root


def serialize_mathml(root: etree._Element) -> str:
    return etree.tostring(root, encoding="unicode", pretty_print=True).strip()


def normalize_mathml(markup: str) -> str:
    return serialize_mathml(parse_and_normalize(markup))
