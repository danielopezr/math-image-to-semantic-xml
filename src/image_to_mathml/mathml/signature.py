"""Structural signatures used to ensure the hybrid flow does not rewrite math."""

from __future__ import annotations

from lxml import etree

from .dom import element_children, element_text, localname

_TOKEN_ELEMENTS = {"mi", "mn", "mo", "mtext", "ms", "mglyph", "mspace"}


def structural_signature(element: etree._Element):
    tag = localname(element.tag)
    if tag in _TOKEN_ELEMENTS:
        return (tag, element_text(element))
    return (
        tag,
        tuple(structural_signature(child) for child in element_children(element)),
    )


def same_structure(left: etree._Element, right: etree._Element) -> bool:
    return structural_signature(left) == structural_signature(right)
