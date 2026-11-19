"""Deterministic rules for calculus notation the general annotator misses.

* ``‖v‖`` is ``norm`` and ``|x|`` is ``absolute-value`` (``|table|`` is left to
  the determinant rule).
* ``∂f/∂x``, ``∂²f/∂x∂y`` and ``∂/∂θ (…)`` are ``partial-derivative``.
* ``∇f`` is ``gradient``, ``∇·F`` ``divergence``, ``∇×F`` and ``rot F``
  ``curl``.
* ``f(x, y)`` (a name applied to several arguments) is ``$name($v1, $v2)``.

These are the rules used to label calc070–calc220 of the dataset.  They run
before the rest of :func:`annotator.annotate_tree` and only touch elements
that have no ``intent`` or ``arg`` yet, so they can also fill the gaps of an
LLM annotation.  :func:`enforce_partial_derivatives` replaces the reading the
LLM gave to a ``∂`` fraction, which has no other reading.
"""

from __future__ import annotations

from lxml import etree

from .dom import element_children, element_text, is_element, localname, new_element

PARTIAL = frozenset({"∂", "𝜕"})
NABLA = frozenset({"∇", "𝛁"})
_PRIMES = frozenset({"′", "″", "‴", "'"})
_OPEN = {"(": ")", "[": "]", "{": "}"}
_DOTS = frozenset({"⋅", "·"})
# Names whose superscript is not a power when applied: f²(x) is rare,
# f^{-1}(x) is the inverse.
_FUNCTION_LETTERS = frozenset({"f", "g", "h", "F", "G", "H", "p", "L"})


def _tok(element: etree._Element, *texts: str) -> bool:
    return localname(element.tag) in {"mo", "mi"} and element_text(element) in texts


def _items(element: etree._Element) -> list[etree._Element]:
    return element_children(element) if localname(element.tag) == "mrow" else [element]


def _free(element: etree._Element) -> bool:
    return element.get("intent") is None and element.get("arg") is None


def _untouched(element: etree._Element) -> bool:
    """No intent or arg anywhere inside (an LLM may have annotated parts)."""

    return all(_free(node) for node in element.iter() if is_element(node))


def _wrap(parent: etree._Element, nodes: list[etree._Element]) -> etree._Element:
    wrapper = new_element("mrow")
    parent.insert(parent.index(nodes[0]), wrapper)
    for node in nodes:
        wrapper.append(node)
    return wrapper


def _as_one(nodes: list[etree._Element]) -> etree._Element:
    if len(nodes) == 1:
        return nodes[0]
    parent = nodes[0].getparent()
    assert parent is not None, "the nodes to group always have a parent"
    return _wrap(parent, nodes)


def _target(row: etree._Element, group: list[etree._Element]) -> etree._Element:
    """The row itself when the group is all of it, otherwise a new mrow.

    The row may already be an argument (``arg="num"``): it keeps that role.
    """

    if len(group) == len(element_children(row)) and row.get("intent") is None:
        return row
    return _wrap(row, group)


def _unwrap_single(row: etree._Element) -> None:
    """Drop an ``<mrow>`` left holding a single child; its ``arg`` moves down."""

    parent = row.getparent()
    children = element_children(row)
    if parent is None or len(children) != 1 or localname(parent.tag) == "math":
        return
    child = children[0]
    if set(row.attrib) - {"arg"} or (row.get("arg") and child.get("arg")):
        return
    arg = row.get("arg")
    if arg:
        child.set("arg", arg)
    parent.replace(row, child)


def _mrows(root: etree._Element) -> list[etree._Element]:
    return [
        node
        for node in root.iter()
        if is_element(node) and localname(node.tag) == "mrow"
    ]


def _match(children: list[etree._Element], index: int) -> int | None:
    """Index of the fence closing ``children[index]``."""

    opener = element_text(children[index])
    closer = _OPEN[opener]
    depth = 0
    for position in range(index, len(children)):
        child = children[position]
        if localname(child.tag) != "mo":
            continue
        text = element_text(child)
        if text == opener:
            depth += 1
        elif text == closer:
            depth -= 1
            if depth == 0:
                return position
    return None


def _contains_table(nodes: list[etree._Element]) -> bool:
    return any(
        is_element(node) and localname(node.tag) == "mtable"
        for element in nodes
        for node in element.iter()
    )


# ---------------------------------------------------------------------------
# ‖v‖ and |x|
# ---------------------------------------------------------------------------
def _annotate_fenced(root: etree._Element, fence: str, concept: str) -> int:
    changed = 0
    progress = True
    while progress:
        progress = False
        for row in _mrows(root):
            if _fenced_once(row, fence, concept):
                changed += 1
                progress = True
                break
    return changed


def _separates_top_level(nodes: list[etree._Element]) -> bool:
    """A relation or a comma outside brackets: ``|a| = |b|``, ``|x|, |y|``.

    Inside brackets they belong to the content: ``|J(u, v)|``.
    """

    depth = 0
    for node in nodes:
        if _tok(node, "(", "[", "{"):
            depth += 1
        elif _tok(node, ")", "]", "}"):
            depth = max(0, depth - 1)
        elif depth == 0 and _tok(node, "=", "<", ">", "≤", "≥", ","):
            return True
    return False


def _fenced_once(row: etree._Element, fence: str, concept: str) -> bool:
    children = element_children(row)
    for start, opener in enumerate(children):
        if not _tok(opener, fence) or not _free(opener):
            continue
        for end in range(start + 2, len(children)):
            closing = children[end]
            powered = localname(closing.tag) == "msup" and _tok(
                element_children(closing)[0], fence
            )
            if not (_tok(closing, fence) or powered):
                continue
            inner = children[start + 1 : end]
            if _separates_top_level(inner):
                return False
            if _contains_table(inner) or not all(_untouched(n) for n in inner):
                return False
            if powered:
                # |x|^2: the msup's base becomes the whole |x|.
                if not _free(closing):
                    return False
                base = element_children(closing)[0]
                group = _wrap(row, children[start:end])
                group.append(base)
                closing.insert(0, group)
                target = group
                _unwrap_single(row)
            else:
                target = _target(row, children[start : end + 1])
            _as_one(inner).set("arg", "a")
            target.set("intent", f"{concept}($a)")
            return True
        return False
    return False


# ---------------------------------------------------------------------------
# ∂f/∂x
# ---------------------------------------------------------------------------
def _partial_order(numerator_first: etree._Element) -> etree._Element | None | bool:
    """The order element of ``∂^n``; None for a plain ``∂``; False otherwise."""

    if localname(numerator_first.tag) == "msup" and _tok(
        element_children(numerator_first)[0], *PARTIAL
    ):
        return element_children(numerator_first)[1]
    if _tok(numerator_first, *PARTIAL):
        return None
    return False


def is_partial_fraction(frac: etree._Element) -> bool:
    if localname(frac.tag) != "mfrac" or len(element_children(frac)) != 2:
        return False
    numerator, denominator = element_children(frac)
    first = _items(numerator)
    return (
        bool(first)
        and _partial_order(first[0]) is not False
        and any(_tok(node, *PARTIAL) for node in _items(denominator))
    )


def _order_text(order: etree._Element | None) -> str | None:
    if order is None:
        return "1"
    if localname(order.tag) == "mn":
        return element_text(order)
    if localname(order.tag) == "mi" and len(element_text(order)) == 1:
        order.set("arg", "n")
        return "$n"
    return None


def _variables(denominator: list[etree._Element]) -> list[etree._Element] | None:
    """``∂x ∂y`` or ``∂x^2`` → the variables, without powers."""

    variables: list[etree._Element] = []
    index = 0
    while index < len(denominator):
        if not _tok(denominator[index], *PARTIAL) or index + 1 >= len(denominator):
            return None
        variable = denominator[index + 1]
        if localname(variable.tag) == "msup":
            variable = element_children(variable)[0]
        if localname(variable.tag) not in {"mi", "msub"}:
            return None
        variables.append(variable)
        index += 2
    return variables or None


def _annotate_partial(frac: etree._Element) -> bool:
    if not is_partial_fraction(frac) or frac.get("intent") is not None:
        return False
    if not _untouched_inside(frac):
        return False
    numerator, denominator = element_children(frac)
    top, bottom = _items(numerator), _items(denominator)
    order = _partial_order(top[0])
    if isinstance(order, bool):  # not a ∂ numerator (excluded above)
        return False
    rest = top[1:]
    variables = _variables(bottom)
    if variables is None:
        return False
    if not rest:
        return _annotate_partial_operator(frac, order, variables)
    if _tok(rest[0], "(") and any(_tok(node, ",") for node in rest):
        return False  # Jacobian notation ∂(f, g)/∂(x, y)
    if any(_tok(node, "/") for node in rest):
        return False
    order_text = _order_text(order)
    if order_text is None:
        return False
    function = _as_one(rest)
    function.set("arg", "f")
    references: list[str] = []
    seen: list[str] = []
    for variable in variables:
        key = etree.tostring(variable, method="text", encoding="unicode")
        if key in seen:
            continue
        seen.append(key)
        variable.set("arg", f"v{len(seen)}")
        references.append(f"$v{len(seen)}")
    frac.set("intent", f"partial-derivative($f,{order_text},{','.join(references)})")
    return True


def _untouched_inside(frac: etree._Element) -> bool:
    return all(
        _free(node) for node in frac.iter() if is_element(node) and node is not frac
    )


def _annotate_partial_operator(
    frac: etree._Element,
    order: etree._Element | None,
    variables: list[etree._Element],
) -> bool:
    """``∂/∂θ (…)``: the derivative of what the parentheses hold."""

    row = frac.getparent()
    if row is None or localname(row.tag) != "mrow" or len(variables) != 1:
        return False
    if frac.get("arg") is not None:
        return False
    siblings = element_children(row)
    index = siblings.index(frac)
    if index + 1 >= len(siblings):
        return False
    following = siblings[index + 1]
    if _tok(following, "("):
        close = _match(siblings, index + 1)
        if close is None or not all(
            _untouched(n) for n in siblings[index + 1 : close + 1]
        ):
            return False
        group = _wrap(row, siblings[index + 1 : close + 1])
    elif (
        localname(following.tag) == "mrow"
        and element_children(following)
        and _tok(element_children(following)[0], "(")
        and _untouched(following)
    ):
        group = following
    else:
        return False
    order_text = _order_text(order)
    if order_text is None:
        return False
    target = _target(row, [frac, group])
    group.set("arg", "f")
    variables[0].set("arg", "v1")
    target.set("intent", f"partial-derivative($f,{order_text},$v1)")
    return True


def annotate_partial_derivatives(root: etree._Element) -> int:
    return sum(
        1
        for frac in [
            node
            for node in root.iter()
            if is_element(node) and localname(node.tag) == "mfrac"
        ]
        if _annotate_partial(frac)
    )


def enforce_partial_derivatives(root: etree._Element) -> int:
    """Replace any other reading of a ``∂`` fraction by ``partial-derivative``.

    Intents and args inside the fraction are cleared (the fraction keeps its
    own ``arg``); the parts are then annotated again by the fill-in pass.
    Returns the number of fractions re-annotated.
    """

    changed = 0
    for frac in [
        node
        for node in root.iter()
        if is_element(node) and localname(node.tag) == "mfrac"
    ]:
        if not is_partial_fraction(frac):
            continue
        if (
            (frac.get("intent") or "")
            .replace(" ", "")
            .startswith("partial-derivative(")
        ):
            continue
        saved = [(node, dict(node.attrib)) for node in frac.iter() if is_element(node)]
        for node, _ in saved:
            if node.get("intent") is not None:
                del node.attrib["intent"]
            if node is not frac and node.get("arg") is not None:
                del node.attrib["arg"]
        if _annotate_partial(frac):
            changed += 1
        else:
            # Not a derivative the rule can read (the Jacobian
            # ∂(f_1, …, f_n)/∂(x_1, …, x_n)): keep what it had, e.g. the
            # ellipses.
            for node, attributes in saved:
                node.attrib.clear()
                node.attrib.update(attributes)
    return changed


# ---------------------------------------------------------------------------
# ∇f, ∇·F, ∇×F, rot F
# ---------------------------------------------------------------------------
def annotate_vector_operators(root: etree._Element) -> int:
    changed = 0
    progress = True
    while progress:
        progress = False
        for row in _mrows(root):
            if _operator_once(row):
                changed += 1
                progress = True
                break
    return changed


def _operator_once(row: etree._Element) -> bool:
    children = element_children(row)
    for index, operator in enumerate(children[:-1]):
        if not (_tok(operator, *NABLA) or _tok(operator, "rot")) or not _free(operator):
            continue
        following = children[index + 1]
        if _tok(operator, "rot"):
            concept, target, group = "curl", following, [operator, following]
        elif _tok(following, *_DOTS) and index + 2 < len(children):
            concept, target, group = (
                "divergence",
                children[index + 2],
                children[index : index + 3],
            )
        elif _tok(following, "×") and index + 2 < len(children):
            concept, target, group = (
                "curl",
                children[index + 2],
                children[index : index + 3],
            )
        elif _tok(following, "(", "["):
            close = _match(children, index + 1)
            if close is None or not all(
                _untouched(n) for n in children[index + 1 : close + 1]
            ):
                continue
            target = _wrap(row, children[index + 1 : close + 1])
            concept, group = "gradient", [operator, target]
        else:
            concept, target, group = "gradient", following, [operator, following]
        if localname(target.tag) == "mo" or not _free(target):
            continue
        wrapper = _target(row, group)
        target.set("arg", "f")
        wrapper.set("intent", f"{concept}($f)")
        return True
    return False


# ---------------------------------------------------------------------------
# f(x, y)
# ---------------------------------------------------------------------------
def _is_bold(element: etree._Element) -> bool:
    text = element_text(element)
    return element.get("mathvariant") == "bold" or (
        len(text) == 1 and 0x1D400 <= ord(text) <= 0x1D433
    )


def _is_primed(element: etree._Element) -> bool:
    if localname(element.tag) not in {"msup", "msubsup"}:
        return False
    children = element_children(element)
    base, sup = children[0], children[-1]
    text = element_text(sup)
    return (
        localname(base.tag) == "mi" and bool(text) and all(c in _PRIMES for c in text)
    )


def _is_name(element: etree._Element) -> bool:
    tag = localname(element.tag)
    intent = element.get("intent") or ""
    if element.get("arg") is not None:
        return False
    if tag == "mrow":
        # (∇f)(a): an operator result applied to a point.
        return intent.startswith("gradient(")
    if intent:
        return False
    if tag == "mi":
        text = element_text(element)
        return (len(text) == 1 and text.isalpha()) or _is_bold(element)
    if tag == "msub":
        return localname(element_children(element)[0].tag) == "mi"
    if tag == "msup":
        base, sup = element_children(element)
        if element_text(sup) in {"−1", "-1"}:
            return localname(base.tag) == "mi"
        if element_text(base) in _FUNCTION_LETTERS:
            return _is_primed(element) or localname(sup.tag) == "mi"
    return _is_primed(element)


def _is_d_operator(element: etree._Element) -> bool:
    """``D`` or ``D_1``, ``D_{ij}``: the derivative operator before a name."""

    if not _untouched(element):
        return False
    tag = localname(element.tag)
    if tag == "msub":
        base = element_children(element)[0]
        return localname(base.tag) == "mi" and element_text(base) == "D"
    return tag == "mi" and element_text(element) == "D"


def _split_arguments(nodes: list[etree._Element]) -> list[list[etree._Element]]:
    parts: list[list[etree._Element]] = []
    current: list[etree._Element] = []
    depth = 0
    for node in nodes:
        if _tok(node, "(", "[", "{"):
            depth += 1
        elif _tok(node, ")", "]", "}"):
            depth -= 1
        if depth == 0 and _tok(node, ",", ";"):
            parts.append(current)
            current = []
        else:
            current.append(node)
    parts.append(current)
    return parts


def annotate_multi_argument_applications(root: etree._Element) -> int:
    changed = 0
    progress = True
    while progress:
        progress = False
        for row in _mrows(root):
            if _application_once(row):
                changed += 1
                progress = True
                break
    return changed


def _application_once(row: etree._Element) -> bool:
    children = element_children(row)
    for index, name in enumerate(children[:-1]):
        if not _tok(children[index + 1], "(", "[") or not _is_name(name):
            continue
        close = _match(children, index + 1)
        if close is None or close == index + 2:
            continue
        inside = children[index + 2 : close]
        parts = _split_arguments(inside)
        # ∇f(a) applies the gradient to a; a plain f(x) is left to the
        # general annotator, which also weighs the product reading.
        minimum = 1 if localname(name.tag) == "mrow" else 2
        if len(parts) < minimum or any(not part for part in parts):
            continue
        if not all(_untouched(node) for node in children[index + 1 : close + 1]):
            continue
        if index > 0 and _is_d_operator(children[index - 1]):
            # D_1 f(x, y): the guide takes D_1 f as the name.
            name = _wrap(row, [children[index - 1], name])
        target = _target(row, [name] + children[index + 1 : close + 1])
        name.set("arg", "name")
        if len(parts) == 1:
            _as_one(parts[0]).set("arg", "var")
            target.set("intent", "$name($var)")
            return True
        references = []
        for position, part in enumerate(parts, 1):
            _as_one(part).set("arg", f"v{position}")
            references.append(f"$v{position}")
        target.set("intent", f"$name({', '.join(references)})")
        return True
    return False


def annotate_calculus(root: etree._Element) -> int:
    """All the rules of this module, in the order used for the labels."""

    return (
        _annotate_fenced(root, "‖", "norm")
        + _annotate_fenced(root, "|", "absolute-value")
        + annotate_partial_derivatives(root)
        + annotate_vector_operators(root)
        + annotate_multi_argument_applications(root)
    )
