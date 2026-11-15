"""Group operators with their operands before semantic annotation.

In Presentation MathML a large operator, a pair of delimiters or a postfix
operator is a *sibling* of what it applies to::

    <msubsup><mo>∑</mo>…</msubsup> <msup>i 2</msup> <mo>+</mo> <mn>1</mn>
    <mo>|</mo> <mi>x</mi> <mo>−</mo> <mn>1</mn> <mo>|</mo>

A concept such as ``sum`` or ``absolute-value`` must target an element that
contains both the operator and its operand, so a small language model would
have to invent a ``wrap`` of the right siblings, which in practice it never
does.  This module adds those ``<mrow>`` groups to the skeleton the model
receives, so it only has to point at an existing element.

Every group contains at least two elements, so the normaliser keeps it, and
groups only add structure: no element is removed or reordered.
"""

from __future__ import annotations

from lxml import etree

from .dom import element_children, element_text, is_element, localname, new_element

_LARGE_OPERATORS = frozenset(
    {
        "∑",  # sum
        "∏",  # product
        "∐",  # coproduct
        "∫",  # integral
        "∬",
        "∭",
        "∮",
        "⋃",  # big union
        "⋂",  # big intersection
        "lim",
        "liminf",
        "limsup",
    }
)
_INTEGRALS = frozenset({"∫", "∬", "∭", "∮"})
_SCRIPTED = frozenset({"msub", "msup", "msubsup", "munder", "mover", "munderover"})

_RELATIONS = frozenset(
    {
        "=",
        "<",
        ">",
        "≤",
        "≥",
        "≠",
        "≈",
        "≡",
        "∝",
        ",",
        ";",
        "⇒",
        "⇔",
    }
)
_ADDITIVE = frozenset({"+", "-", "−", "±", "∓"})
_DIFFERENTIALS = frozenset({"d", "ⅆ", "∂"})

# Delimiter pairs; for symmetric pairs the opener and closer are equal.
_FENCES = {
    "|": "|",
    "‖": "‖",
    "∥": "∥",
    "⌊": "⌋",
    "⌈": "⌉",
    "⟨": "⟩",
    "〈": "〉",
}
_FENCE_TAGS = frozenset({"mo", "mi"})


def _text(element: etree._Element) -> str:
    return element_text(element)


def _is_token(element: etree._Element, values: frozenset[str] | set[str]) -> bool:
    return localname(element.tag) in _FENCE_TAGS and _text(element) in values


def _wrap(parent: etree._Element, nodes: list[etree._Element]) -> etree._Element:
    wrapper = new_element("mrow")
    parent.insert(parent.index(nodes[0]), wrapper)
    for node in nodes:
        wrapper.append(node)
    return wrapper


def _covers_row(children: list[etree._Element], start: int, end: int) -> bool:
    return start == 0 and end == len(children) - 1


# ---------------------------------------------------------------------------
# delimiters
# ---------------------------------------------------------------------------
def _fence_pair(children: list[etree._Element]) -> tuple[int, int] | None:
    """Return the first innermost delimiter pair that can be grouped."""

    stack: list[tuple[int, str]] = []
    for index, child in enumerate(children):
        if localname(child.tag) not in _FENCE_TAGS:
            continue
        text = _text(child)
        if stack and text == _FENCES[stack[-1][1]]:
            start, _ = stack.pop()
            if index - start >= 2 and not _covers_row(children, start, index):
                return start, index
            continue
        if text in _FENCES:
            stack.append((index, text))
    return None


def _group_fences(mrow: etree._Element) -> list[etree._Element]:
    created: list[etree._Element] = []
    while True:
        children = element_children(mrow)
        pair = _fence_pair(children)
        if pair is None:
            return created
        start, end = pair
        created.append(_wrap(mrow, children[start : end + 1]))


# ---------------------------------------------------------------------------
# postfix factorial
# ---------------------------------------------------------------------------
def _operand_start(children: list[etree._Element], end: int) -> int:
    """Start index of the operand that ends at ``end`` (a closing paren)."""

    closer = _text(children[end])
    opener = {")": "(", "]": "["}.get(closer)
    if localname(children[end].tag) != "mo" or opener is None:
        return end
    depth = 0
    for index in range(end - 1, -1, -1):
        text = _text(children[index])
        if localname(children[index].tag) != "mo":
            continue
        if text == closer:
            depth += 1
        elif text == opener:
            if depth == 0:
                return index
            depth -= 1
    return end


def _group_factorials(mrow: etree._Element) -> list[etree._Element]:
    created: list[etree._Element] = []
    index = 1
    while True:
        children = element_children(mrow)
        if index >= len(children):
            return created
        child = children[index]
        if not (localname(child.tag) == "mo" and _text(child) == "!"):
            index += 1
            continue
        start = _operand_start(children, index - 1)
        if _covers_row(children, start, index):
            return created
        created.append(_wrap(mrow, children[start : index + 1]))
        index = start + 1


# ---------------------------------------------------------------------------
# large operators, limits and derivative operators
# ---------------------------------------------------------------------------
def _operator_symbol(element: etree._Element) -> str | None:
    tag = localname(element.tag)
    if tag == "mo" and _text(element) in _LARGE_OPERATORS:
        return _text(element)
    if tag in _SCRIPTED:
        children = element_children(element)
        if children and localname(children[0].tag) == "mo":
            text = _text(children[0])
            if text in _LARGE_OPERATORS:
                return text
    return None


def _is_derivative_operator(element: etree._Element) -> bool:
    """d/dx or ∂/∂x written as a fraction with an empty operand."""

    if localname(element.tag) != "mfrac":
        return False
    children = element_children(element)
    if len(children) != 2:
        return False
    numerator, denominator = (_text(child) for child in children)
    if not denominator or denominator[0] not in _DIFFERENTIALS:
        return False
    # "d" or "d2" (d^2) alone: an operator waiting for its operand.  "df" is
    # already a complete derivative and is not grouped with what follows.
    return numerator in _DIFFERENTIALS or (
        len(numerator) == 2
        and numerator[0] in _DIFFERENTIALS
        and numerator[1].isdigit()
    )


def _body_end(
    children: list[etree._Element], start: int, *, integral: bool
) -> int | None:
    """Last index of the operand that starts at ``start``."""

    last: int | None = None
    index = start
    while index < len(children):
        child = children[index]
        text = _text(child)
        if localname(child.tag) == "mo" and text in _RELATIONS:
            break
        if (
            not integral
            and index > start
            and localname(child.tag) == "mo"
            and text in _ADDITIVE
        ):
            break
        if (
            integral
            and localname(child.tag) == "mi"
            and text in _DIFFERENTIALS
            and index + 1 < len(children)
            and localname(children[index + 1].tag) == "mi"
        ):
            return index + 1
        if localname(child.tag) != "mspace":
            last = index
        index += 1
    return last


def _group_operators(mrow: etree._Element) -> list[etree._Element]:
    created: list[etree._Element] = []
    index = 0
    while True:
        children = element_children(mrow)
        if index >= len(children):
            return created
        child = children[index]
        symbol = _operator_symbol(child)
        if symbol is None and not _is_derivative_operator(child):
            index += 1
            continue
        end = _body_end(children, index + 1, integral=symbol in _INTEGRALS)
        if end is None or _covers_row(children, index, end):
            index += 1
            continue
        created.append(_wrap(mrow, children[index : end + 1]))
        index += 1


def group_operator_scopes(root: etree._Element) -> int:
    """Add ``<mrow>`` groups around operators and their operands.

    Returns the number of groups created.
    """

    queue = [
        element
        for element in root.iter()
        if is_element(element) and localname(element.tag) == "mrow"
    ]
    created = 0
    seen: set[int] = set()
    while queue:
        mrow = queue.pop(0)
        if id(mrow) in seen:
            continue
        seen.add(id(mrow))
        new_rows: list[etree._Element] = []
        for step in (_group_fences, _group_factorials, _group_operators):
            new_rows.extend(step(mrow))
        created += len(new_rows)
        queue.extend(new_rows)
    return created
