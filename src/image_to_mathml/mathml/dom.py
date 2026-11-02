"""Small helpers for namespace-aware MathML trees."""

from __future__ import annotations

from lxml import etree

MATHML_NS = "http://www.w3.org/1998/Math/MathML"


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
