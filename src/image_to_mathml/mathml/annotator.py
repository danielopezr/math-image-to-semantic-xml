"""Deterministic intent annotation used as a safe fallback.

This module does not try to replace the language model.  It recognises the
unambiguous structures produced by ``latex2mathml`` (fractions, powers, roots,
named function applications and generic ``f(x)`` applications) so the pipeline
can always return a useful accessible expression when the local model is not
available or its answer does not validate.
"""

from __future__ import annotations

from lxml import etree

from .dom import element_children, element_text, is_element, localname, new_element

FUNCTION_CONCEPTS: dict[str, str] = {
    "sin": "sine",
    "sen": "sine",
    "cos": "cosine",
    "tan": "tangent",
    "tg": "tangent",
    "sec": "secant",
    "csc": "cosecant",
    "cot": "cotangent",
    "arcsin": "arcsine",
    "arccos": "arccosine",
    "arctan": "arctangent",
    "sinh": "hyperbolic-sine",
    "cosh": "hyperbolic-cosine",
    "tanh": "hyperbolic-tangent",
    "sech": "hyperbolic-secant",
    "csch": "hyperbolic-cosecant",
    "coth": "hyperbolic-cotangent",
    "ln": "natural-logarithm",
    "log": "logarithm",
    "exp": "exponential",
    "det": "determinant",
    "max": "max",
    "min": "min",
    "\u0393": "gamma-function",
    "gamma": "gamma-function",
    "erf": "error-function",
    "\u03b2": "beta-function",
    "beta": "beta-function",
}

OPERATOR_INTENTS: dict[str, str] = {
    # "+": "plus",
    # "\u2212": "minus",
    # "-": "minus",
    # "=": "equals",
    "\u00d7": "times",
    "\u00b7": "times",
    "\u00f7": "divided-by",
    "/": "divided-by",
    "\u2264": "less-than-or-equal-to",
    "\u2265": "greater-than-or-equal-to",
    "\u2260": "not-equal-to",
    "<": "less-than",
    ">": "greater-than",
    "\u2208": "element-of",
    "\u2209": "does-not-belong-to",
    "\u222a": "union",
    "\u2229": "intersection",
    "\u2211": "sum",
    "\u220f": "product",
}


def _is_mo(element: etree._Element, text: str | None = None) -> bool:
    if localname(element.tag) != "mo":
        return False
    return text is None or element_text(element) == text


def _is_mi(element: etree._Element) -> bool:
    return localname(element.tag) == "mi"


def _find_matching_paren(children: list[etree._Element], open_index: int) -> int | None:
    depth = 0
    for index in range(open_index, len(children)):
        child = children[index]
        if localname(child.tag) != "mo":
            continue
        text = element_text(child)
        if text == "(":
            depth += 1
        elif text == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _wrap_content(
    target: etree._Element,
    content: list[etree._Element],
    arg_name: str,
    new_mrows: list[etree._Element],
) -> None:
    if not content:
        return
    if len(content) == 1 and content[0].get("arg") is None:
        content[0].set("arg", arg_name)
        return

    wrapper = new_element("mrow")
    wrapper.set("arg", arg_name)
    first_index = target.index(content[0])
    target.insert(first_index, wrapper)
    for node in content:
        wrapper.append(node)
    new_mrows.append(wrapper)


def _annotate_function_applications_in(mrow: etree._Element) -> list[etree._Element]:
    if mrow.get("intent") is not None:
        return []

    new_mrows: list[etree._Element] = []
    children = element_children(mrow)
    index = 0
    while index < len(children):
        candidate = children[index]
        if not _is_mi(candidate) or index + 2 >= len(children):
            index += 1
            continue
        open_paren = children[index + 1]
        if not _is_mo(open_paren, "("):
            index += 1
            continue
        close_index = _find_matching_paren(children, index + 1)
        if close_index is None or close_index <= index + 2:
            index += 1
            continue

        content = children[index + 2 : close_index]
        if not content:
            index += 1
            continue

        name = element_text(candidate)
        if not name:
            index += 1
            continue

        concept = FUNCTION_CONCEPTS.get(name)
        whole_row = index == 0 and close_index == len(children) - 1

        if whole_row:
            target = mrow
        else:
            target = new_element("mrow")
            insert_index = mrow.index(candidate)
            mrow.insert(insert_index, target)
            for node in children[index : close_index + 1]:
                target.append(node)
            children = element_children(mrow)
            index = insert_index

        if concept is None:
            target.set("intent", "$name($var)")
            candidate.set("arg", "name")
            argument_name = "var"
        else:
            target.set("intent", f"{concept}($arg)")
            argument_name = "arg"

        _wrap_content(target, content, argument_name, new_mrows)

        if whole_row:
            break
        children = element_children(mrow)
        index = mrow.index(target) + 1

    return new_mrows


def _annotate_function_applications(root: etree._Element) -> None:
    queue = [
        node
        for node in root.iter()
        if is_element(node) and localname(node.tag) == "mrow"
    ]
    seen: set[int] = set()
    while queue:
        mrow = queue.pop(0)
        marker = id(mrow)
        if marker in seen:
            continue
        seen.add(marker)
        queue.extend(_annotate_function_applications_in(mrow))


_BARE_ARGUMENT_TAGS = {
    "mi",
    "mn",
    "mrow",
    "msup",
    "msub",
    "msubsup",
    "mfrac",
    "msqrt",
    "mroot",
    "mtext",
}


def _is_bare_argument(element: etree._Element) -> bool:
    if localname(element.tag) not in _BARE_ARGUMENT_TAGS:
        return False
    if element.get("intent") is not None:
        return False
    if element.get("arg") not in (None, "arg"):
        return False
    return not (
        localname(element.tag) == "mi" and element_text(element) in FUNCTION_CONCEPTS
    )


_BARE_ARGUMENT_FOLLOWERS = {
    "+",
    "\u2212",
    "-",
    "=",
    "<",
    ">",
    "\u2264",
    "\u2265",
    "\u2260",
    "\u00d7",
    "\u00b7",
    "\u00f7",
    "/",
    ",",
    ";",
    ":",
}


def _bare_argument_is_unambiguous(
    children: list[etree._Element], argument_index: int
) -> bool:
    if argument_index == len(children) - 1:
        return True
    following = children[argument_index + 1]
    if localname(following.tag) != "mo":
        return False
    return element_text(following) in _BARE_ARGUMENT_FOLLOWERS


def _annotate_bare_functions_in(mrow: etree._Element) -> None:
    if mrow.get("intent") is not None:
        return
    children = element_children(mrow)
    index = 0
    while index + 1 < len(children):
        candidate = children[index]
        if not _is_mi(candidate):
            index += 1
            continue
        concept = FUNCTION_CONCEPTS.get(element_text(candidate))
        argument = children[index + 1]
        if (
            concept is None
            or not _is_bare_argument(argument)
            or not _bare_argument_is_unambiguous(children, index + 1)
        ):
            index += 1
            continue

        if index == 0 and len(children) == 2:
            target = mrow
        else:
            target = new_element("mrow")
            insert_index = mrow.index(candidate)
            mrow.insert(insert_index, target)
            target.append(candidate)
            target.append(argument)
            children = element_children(mrow)
            index = insert_index

        target.set("intent", f"{concept}($arg)")
        argument.set("arg", "arg")
        if target is mrow:
            break
        children = element_children(mrow)
        index = mrow.index(target) + 1


def _annotate_bare_functions(root: etree._Element) -> None:
    queue = [
        node
        for node in root.iter()
        if is_element(node) and localname(node.tag) == "mrow"
    ]
    seen: set[int] = set()
    while queue:
        mrow = queue.pop(0)
        marker = id(mrow)
        if marker in seen:
            continue
        seen.add(marker)
        _annotate_bare_functions_in(mrow)
        queue.extend(
            child for child in element_children(mrow) if localname(child.tag) == "mrow"
        )


def _set_arg(element: etree._Element, name: str) -> bool:
    existing = element.get("arg")
    if existing is not None:
        return existing == name
    element.set("arg", name)
    return True


def _set_args(children: list[etree._Element], names: list[str]) -> bool:
    if len(children) != len(names):
        return False
    for child, name in zip(children, names):
        existing = child.get("arg")
        if existing is not None and existing != name:
            return False
    for child, name in zip(children, names):
        if child.get("arg") is None:
            child.set("arg", name)
    return True


def _annotate_structures(root: etree._Element) -> None:
    for element in list(root.iter()):
        if not is_element(element):
            continue
        if element.get("intent") is not None:
            continue
        tag = localname(element.tag)
        children = element_children(element)

        if tag == "mfrac" and len(children) == 2:
            if _set_args(children, ["num", "den"]):
                element.set("intent", "fraction($num,$den)")

        elif tag == "msup" and len(children) == 2:
            base, exponent = children
            if element_text(base) in FUNCTION_CONCEPTS:
                # sin^2(x) is not the power of the symbol "sin"; leave it
                # unannotated rather than emit a misleading power intent.
                continue
            if element_text(base) == "e" and _set_arg(exponent, "exp"):
                element.set("intent", "exponential($exp)")
            elif _set_args(children, ["base", "exp"]):
                element.set("intent", "power($base,$exp)")

        elif tag == "msub" and len(children) == 2:
            if element_text(children[0]) in FUNCTION_CONCEPTS:
                # log_2(x) is a logarithm with a base, not an indexed symbol.
                continue
            if _set_args(children, ["base", "sub"]):
                element.set("intent", "indexed-by($base,$sub)")

        elif tag == "msqrt" and children:
            if len(children) == 1 and children[0].get("arg") is None:
                children[0].set("arg", "radicand")
            else:
                wrapper = new_element("mrow")
                wrapper.set("arg", "radicand")
                for child in children:
                    wrapper.append(child)
                element.append(wrapper)
            element.set("intent", "root($radicand, 2)")

        elif (
            tag == "mroot"
            and len(children) == 2
            and _set_args(children, ["radicand", "index"])
        ):
            element.set("intent", "root($radicand,$index)")


def _annotate_operators(root: etree._Element) -> None:
    for element in root.iter():
        if not is_element(element) or localname(element.tag) != "mo":
            continue
        if element.get("intent") is not None:
            continue
        concept = OPERATOR_INTENTS.get(element_text(element))
        if concept:
            element.set("intent", concept)


def annotate_tree(root: etree._Element) -> int:
    """Add deterministic intents to a normalised MathML tree.

    Returns the number of ``intent`` attributes present after annotation.
    """

    _annotate_function_applications(root)
    _annotate_bare_functions(root)
    _annotate_structures(root)
    _annotate_operators(root)
    return sum(
        1
        for element in root.iter()
        if is_element(element) and element.get("intent") is not None
    )
