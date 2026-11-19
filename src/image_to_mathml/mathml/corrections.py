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


_CLOSER_TO_OPENER = {")": "(", "]": "[", "}": "{"}
_SCRIPT_TAGS = {"msup", "msub", "msubsup"}


def _matching_opener_index(
    siblings: list[etree._Element], close_index: int, opener: str, closer: str
) -> int | None:
    depth = 0
    for index in range(close_index - 1, -1, -1):
        sibling = siblings[index]
        if localname(sibling.tag) != "mo":
            continue
        text = element_text(sibling)
        if text == closer:
            depth += 1
        elif text == opener:
            if depth == 0:
                return index
            depth -= 1
    return None


def attach_parenthesized_bases(root: etree._Element) -> int:
    """Make ``(1+r)^t`` a power of the whole parenthesised group.

    ``latex2mathml`` attaches a script to the closing delimiter only:
    ``( 1 + r <msup>) t</msup>``, which reads as "close parenthesis to the
    power t".  The opening delimiter and everything up to the closing one are
    moved into an ``<mrow>`` that becomes the base of the script.  Returns the
    number of scripts fixed.
    """

    fixed = 0
    for script in list(root.iter()):
        if localname(script.tag) not in _SCRIPT_TAGS:
            continue
        children = element_children(script)
        if not children or localname(children[0].tag) != "mo":
            continue
        closer = element_text(children[0])
        opener = _CLOSER_TO_OPENER.get(closer)
        parent = script.getparent()
        if opener is None or parent is None:
            continue
        siblings = element_children(parent)
        script_index = siblings.index(script)
        open_index = _matching_opener_index(siblings, script_index, opener, closer)
        if open_index is None:
            continue

        base = etree.Element(qname("mrow"))
        for node in siblings[open_index:script_index]:
            base.append(node)  # moving keeps each node's tail
        closing = children[0]
        closing_tail = closing.tail
        closing.tail = None
        base.append(closing)
        base.tail = closing_tail
        script.insert(0, base)
        fixed += 1
    return fixed


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


# Symbols that latex2mathml and the models write as <mo> but that are
# identifiers: \infty is a quantity, not an operator.
_IDENTIFIER_SYMBOLS = frozenset({"∞"})


def fix_identifier_tokens(root: etree._Element) -> int:
    """Turn ``<mo>∞</mo>`` into ``<mi>∞</mi>``; return how many changed.

    Operator attributes (``stretchy``, ``form``...) do not apply to ``<mi>``
    and are dropped; semantic attributes (``intent``, ``arg``) are kept.
    """

    fixed = 0
    for element in list(root.iter()):
        if localname(element.tag) != "mo":
            continue
        if element_text(element).strip() not in _IDENTIFIER_SYMBOLS:
            continue
        element.tag = qname("mi")
        for name in list(element.attrib):
            if name not in {"intent", "arg", "id"}:
                del element.attrib[name]
        fixed += 1
    return fixed
