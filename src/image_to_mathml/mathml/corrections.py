"""Deterministic MathML corrections shared by the normaliser and annotator."""

from __future__ import annotations

from lxml import etree

from .dom import element_children, element_text, localname, qname


def _base_key(element: etree._Element) -> str:
    return "".join(element_text(element).split())


def _subscript_digits(subscript: etree._Element) -> str:
    tag = localname(subscript.tag)
    if tag == "mn":
        return (subscript.text or "").strip()
    if tag != "mrow":
        return ""
    parts: list[str] = []
    for child in element_children(subscript):
        if localname(child.tag) != "mn":
            return ""
        text = (child.text or "").strip()
        if not text or not text.isdigit():
            return ""
        parts.append(text)
    return "".join(parts)


def _subscript_cell(subscript: etree._Element) -> etree._Element | None:
    tag = localname(subscript.tag)
    if tag == "mn":
        return subscript
    if tag == "mrow":
        nested = element_children(subscript)
        if len(nested) == 1 and localname(nested[0].tag) == "mn":
            return nested[0]
    return None


def _inside_matrix(element: etree._Element) -> bool:
    parent = element.getparent()
    while parent is not None:
        if localname(parent.tag) == "mtable":
            return True
        parent = parent.getparent()
    return False


def _matrix_indexes(root: etree._Element) -> set[tuple[str, str]]:
    indexes: set[tuple[str, str]] = set()
    for table in root.iter():
        if localname(table.tag) != "mtable":
            continue
        for element in table.iter():
            if localname(element.tag) not in {"msub", "msubsup"}:
                continue
            children = element_children(element)
            if len(children) < 2:
                continue
            base = _base_key(children[0])
            digits = _subscript_digits(children[1])
            if base and len(digits) > 1 and digits.isdigit():
                indexes.add((base, digits))
    return indexes


def _split_subscript(element: etree._Element) -> None:
    children = element_children(element)
    if len(children) < 2:
        return

    subscript = children[1]
    cell = _subscript_cell(subscript)
    if cell is None:
        return

    digits = (cell.text or "").strip()
    if len(digits) < 2 or not digits.isdigit():
        return

    parent = cell.getparent()
    if parent is None:
        return

    if subscript is cell:
        wrapper = etree.Element(qname("mrow"))
        wrapper.tail = cell.tail
        for key, value in cell.attrib.items():
            wrapper.set(key, value)
        index = parent.index(cell)
        parent.remove(cell)
        for digit in digits:
            digit_cell = etree.Element(qname("mn"))
            digit_cell.text = digit
            wrapper.append(digit_cell)
        parent.insert(index, wrapper)
        return

    for key, value in cell.attrib.items():
        if key not in subscript.attrib:
            subscript.set(key, value)
    cell_tail = cell.tail
    index = parent.index(cell)
    parent.remove(cell)
    for offset, digit in enumerate(digits):
        digit_cell = etree.Element(qname("mn"))
        digit_cell.text = digit
        if offset == len(digits) - 1:
            digit_cell.tail = cell_tail
        parent.insert(index + offset, digit_cell)


def split_matrix_subscripts(root: etree._Element) -> None:
    """Split matrix indexes and matching references into individual digits.

    A matrix index such as a_{11} must be read as "a sub one one", not as the
    number eleven. The correction is applied to subscripts inside an mtable and
    to later references that reuse the same base and index, such as a_{11}
    outside the matrix.
    """

    matrix_indexes = _matrix_indexes(root)
    for element in list(root.iter()):
        if localname(element.tag) not in {"msub", "msubsup"}:
            continue
        if _inside_matrix(element):
            _split_subscript(element)
            continue
        children = element_children(element)
        if len(children) < 2:
            continue
        base = _base_key(children[0])
        digits = _subscript_digits(children[1])
        if base and digits and (base, digits) in matrix_indexes:
            _split_subscript(element)
