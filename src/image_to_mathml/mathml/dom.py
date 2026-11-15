"""Small helpers for namespace-aware MathML trees."""

from __future__ import annotations

from lxml import etree

MATHML_NS = "http://www.w3.org/1998/Math/MathML"

# Presentation elements whose children are positional.  Each child has a fixed
# role (numerator, base, exponent, ...), so the number of children is part of
# the structure and children must never be removed or grouped together.
FIXED_ARITY: dict[str, int] = {
    "mfrac": 2,
    "msup": 2,
    "msub": 2,
    "mroot": 2,
    "munder": 2,
    "mover": 2,
    "msubsup": 3,
    "munderover": 3,
}

# Elements whose element children must all have one specific tag.
REQUIRED_CHILD_TAGS: dict[str, frozenset[str]] = {
    "mtable": frozenset({"mtr", "mlabeledtr"}),
    "mtr": frozenset({"mtd"}),
    "mlabeledtr": frozenset({"mtd"}),
}


def has_positional_children(element: etree._Element) -> bool:
    """Return True when the children of ``element`` have fixed roles."""

    tag = localname(element.tag)
    return tag in FIXED_ARITY or tag == "mmultiscripts"


def qname(name: str) -> str:
    return f"{{{MATHML_NS}}}{name}"


def localname(tag: str) -> str:
    if not isinstance(tag, str):
        return ""
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def is_element(node: object) -> bool:
    return isinstance(getattr(node, "tag", None), str)


def element_children(element: etree._Element) -> list[etree._Element]:
    return [child for child in element if is_element(child)]


def element_text(element: etree._Element) -> str:
    return "".join(element.itertext()).strip()


def new_element(name: str) -> etree._Element:
    return etree.Element(qname(name))


def path(element: etree._Element) -> str:
    parts: list[str] = []
    current: etree._Element | None = element
    while current is not None and is_element(current):
        name = localname(current.tag)
        parent = current.getparent()
        if parent is None:
            parts.append(name)
            break
        siblings = [
            node for node in parent if is_element(node) and localname(node.tag) == name
        ]
        if len(siblings) > 1:
            name = f"{name}[{siblings.index(current) + 1}]"
        parts.append(name)
        current = parent
    return "/" + "/".join(reversed(parts))
