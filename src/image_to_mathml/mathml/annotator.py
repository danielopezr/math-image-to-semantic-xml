"""Deterministic intent annotation used as a safe fallback.

This module does not try to replace the language model.  It recognises the
unambiguous structures produced by ``latex2mathml`` (fractions, powers, roots,
named function applications and generic ``f(x)`` applications) so the pipeline
can always return a useful accessible expression when the local model is not
available or its answer does not validate. It deliberately leaves bare
operators unannotated unless a structural rule gives them a non-default
meaning, matching the agent prompts.

The guiding rule is: when a structure has more than one common reading, leave
it unannotated.  A screen reader reads plain Presentation MathML correctly,
while a wrong ``intent`` makes it say something false.
"""

from __future__ import annotations

from lxml import etree

from .corrections import split_matrix_subscripts
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
    "Γ": "gamma-function",
    "gamma": "gamma-function",
    "erf": "error-function",
}

# Named functions that are only recognised when followed by parentheses.  A
# bare "Γ x" is far more often a product with the symbol Γ than Γ(x).
_PARENTHESIS_ONLY_FUNCTIONS = frozenset({"Γ", "gamma"})

# Single-letter names that conventionally denote functions in f(x).  Any other
# single letter followed by parentheses, as in a(b+c) or 2x(x+1), is far more
# often a product than a function application.
GENERIC_FUNCTION_NAMES = frozenset({"f", "g", "h", "F", "G", "H"})

# Superscripts that are not exponents: primes, conjugates, adjoints, degrees.
_NON_EXPONENT_SUPERSCRIPTS = frozenset(
    {
        "′",  # prime
        "″",  # double prime
        "‴",  # triple prime
        "⁗",  # quadruple prime
        "'",
        "''",
        "*",
        "∗",  # asterisk operator
        "†",  # dagger
        "‡",  # double dagger
        "∘",  # ring operator (degree)
        "°",  # degree sign
        "T",  # transpose or a variable T: ambiguous
        "⊤",  # down tack (transpose)
        "ᵀ",  # modifier letter capital T
        "c",  # complement
        "C",  # complement
    }
)
_TRANSPOSE_SUPERSCRIPTS = frozenset({"⊤", "ᵀ"})


def _is_mo(element: etree._Element, text: str | None = None) -> bool:
    if localname(element.tag) != "mo":
        return False
    return text is None or element_text(element) == text


def _is_mi(element: etree._Element) -> bool:
    return localname(element.tag) == "mi"


def _named_function(element: etree._Element) -> str | None:
    """Return the concept of a named function token such as ``sin`` or ``det``.

    ``latex2mathml`` writes some operator names (``\\det``, ``\\max``,
    ``\\lim``) as ``<mo>`` and others (``\\sin``) as ``<mi>``; both are
    accepted.
    """

    if localname(element.tag) not in {"mi", "mo"}:
        return None
    return FUNCTION_CONCEPTS.get(element_text(element))


def _is_operator_name(element: etree._Element) -> bool:
    """True for multi-letter operator names such as ``lim``, ``sup``, ``det``."""

    text = element_text(element)
    return localname(element.tag) == "mo" and len(text) > 1 and text.isalpha()


_LIMIT_OPERATORS = frozenset({"lim", "liminf", "limsup"})


def _is_generic_name_token(element: etree._Element) -> bool:
    text = element_text(element)
    if not text or text in FUNCTION_CONCEPTS:
        return False
    if localname(element.tag) == "mo":
        # \operatorname{sgn} is written as <mo>sgn</mo>.
        return len(text) > 1 and text.isalpha() and text not in _LIMIT_OPERATORS
    if not _is_mi(element):
        return False
    if text in GENERIC_FUNCTION_NAMES:
        return True
    # Multi-letter identifiers come from \operatorname{...} or \mathrm{...}.
    return len(text) > 1 and text.isalpha()


def _is_prime_superscript(element: etree._Element) -> bool:
    text = element_text(element)
    return bool(text) and all(char in "′″‴⁗'" for char in text)


def is_generic_function_name(element: etree._Element) -> bool:
    """Return True when ``element`` can be the name in a generic ``f(x)``.

    Accepted: ``f``, ``g``, ``h``, ``F``, ``G``, ``H``, multi-letter operator
    names, their indexed forms (``f_i``) and their primed forms (``f'``).
    """

    tag = localname(element.tag)
    if tag in {"mi", "mo"}:
        return _is_generic_name_token(element)
    children = element_children(element)
    if tag == "msub" and len(children) == 2:
        return _is_generic_name_token(children[0])
    if tag == "msup" and len(children) == 2:
        return _is_generic_name_token(children[0]) and _is_prime_superscript(
            children[1]
        )
    return False


def _log_base(element: etree._Element) -> etree._Element | None:
    """Return the base of ``\\log_b`` written as ``<msub><mi>log</mi>b</msub>``."""

    if localname(element.tag) != "msub":
        return None
    children = element_children(element)
    if len(children) != 2:
        return None
    if localname(children[0].tag) not in {"mi", "mo"}:
        return None
    if element_text(children[0]) != "log":
        return None
    return children[1]


def _function_name(
    element: etree._Element,
) -> tuple[etree._Element, str | None] | None:
    """Classify a possible function name.

    Returns ``(name_element, concept)``: ``concept`` is a core concept name
    for named functions, ``"logarithm-base"`` for ``\\log_b`` and ``None`` for
    a generic application.  Returns ``None`` when ``element`` is not a
    function name.
    """

    concept = _named_function(element)
    if concept is not None:
        return element, concept
    if _log_base(element) is not None:
        return element, "logarithm-base"
    if is_generic_function_name(element):
        return element, None
    return None


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


def _set_function_intent(
    target: etree._Element,
    name_element: etree._Element,
    concept: str | None,
) -> str:
    """Set the intent of a function application and return the argument role."""

    if concept is None:
        target.set("intent", "$name($var)")
        name_element.set("arg", "name")
        return "var"
    if concept == "logarithm-base":
        base = _log_base(name_element)
        assert base is not None
        base.set("arg", "base")
        target.set("intent", "logarithm($arg,$base)")
        return "arg"
    target.set("intent", f"{concept}($arg)")
    return "arg"


def _annotate_function_applications_in(mrow: etree._Element) -> list[etree._Element]:
    if mrow.get("intent") is not None:
        return []

    new_mrows: list[etree._Element] = []
    children = element_children(mrow)
    index = 0
    while index < len(children):
        candidate = children[index]
        if index + 2 >= len(children):
            index += 1
            continue
        name_info = _function_name(candidate)
        if name_info is None:
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

        name_element, concept = name_info
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

        argument_name = _set_function_intent(target, name_element, concept)
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
    return _named_function(element) is None


_BARE_ARGUMENT_FOLLOWERS = {
    "+",
    "−",
    "-",
    "=",
    "<",
    ">",
    "≤",
    "≥",
    "≠",
    "×",
    "·",
    "÷",
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


def _bare_function(element: etree._Element) -> str | None:
    """Concept of a function that may be applied without parentheses."""

    if _log_base(element) is not None:
        return "logarithm-base"
    concept = _named_function(element)
    if concept is None or element_text(element) in _PARENTHESIS_ONLY_FUNCTIONS:
        return None
    return concept


def _annotate_bare_functions_in(mrow: etree._Element) -> None:
    if mrow.get("intent") is not None:
        return
    children = element_children(mrow)
    index = 0
    while index + 1 < len(children):
        candidate = children[index]
        concept = _bare_function(candidate)
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

        _set_function_intent(target, candidate, concept)
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


def _annotate_linear_fractions(root: etree._Element) -> None:
    for element in list(root.iter()):
        if not is_element(element) or localname(element.tag) != "mrow":
            continue
        if element.get("intent") is not None:
            continue
        children = element_children(element)
        if len(children) != 3:
            continue
        numerator, operator, denominator = children
        if not _is_mo(operator, "/"):
            continue
        if localname(numerator.tag) == "mo" or localname(denominator.tag) == "mo":
            continue
        if numerator.get("arg") not in (None, "num"):
            continue
        if denominator.get("arg") not in (None, "den"):
            continue
        numerator.set("arg", "num")
        denominator.set("arg", "den")
        element.set("intent", "fraction($num,$den)")


def _not_a_power_base(base: etree._Element) -> bool:
    """Bases whose superscript is not an exponent of an ordinary power."""

    if localname(base.tag) == "mo":
        # Large operators and operator names: \sum^n, \int^b, \lim, \max.
        return True
    return element_text(base) in FUNCTION_CONCEPTS


def _annotate_msup(element: etree._Element, base, exponent) -> None:
    if _not_a_power_base(base):
        # sin^2(x) is not the power of the symbol "sin"; leave it unannotated
        # rather than emit a misleading power intent.
        return
    exponent_text = element_text(exponent)
    if exponent_text in _TRANSPOSE_SUPERSCRIPTS:
        if _set_arg(base, "arg"):
            element.set("intent", "transpose($arg)")
        return
    if exponent_text in _NON_EXPONENT_SUPERSCRIPTS or _is_prime_superscript(exponent):
        # f', z*, A^T, A^c: not powers.  The plain presentation reads better
        # than a wrong "to the power" reading.
        return
    if element_text(base) == "e" and _set_arg(exponent, "exp"):
        element.set("intent", "exponential($exp)")
    elif _set_args([base, exponent], ["base", "exp"]):
        element.set("intent", "power($base,$exp)")


def _annotate_msubsup(element: etree._Element, children) -> None:
    base, _subscript, superscript = children
    if _not_a_power_base(base):
        # \sum_{i=1}^{n}, \int_0^1: the scripts are limits, not indexes.
        return
    if element_text(superscript) in _NON_EXPONENT_SUPERSCRIPTS or (
        _is_prime_superscript(superscript)
    ):
        return
    if _set_args(children, ["base", "sub", "exp"]):
        element.set("intent", "power(indexed-by($base,$sub),$exp)")


def _annotate_structures(root: etree._Element) -> None:
    for element in list(root.iter()):
        if not is_element(element):
            continue
        if element.get("intent") is not None:
            continue
        if element.get("arg") == "name":
            continue
        tag = localname(element.tag)
        children = element_children(element)

        if tag == "mfrac" and len(children) == 2:
            if _set_args(children, ["num", "den"]):
                element.set("intent", "fraction($num,$den)")

        elif tag == "msup" and len(children) == 2:
            _annotate_msup(element, children[0], children[1])

        elif tag == "msubsup" and len(children) == 3:
            _annotate_msubsup(element, children)

        elif tag == "msub" and len(children) == 2:
            base = children[0]
            if element_text(base) in FUNCTION_CONCEPTS or _is_operator_name(base):
                # log_2(x) is a logarithm with a base and lim_{x->0} a limit,
                # not indexed symbols.
                continue
            if localname(base.tag) == "mo":
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


def annotate_tree(root: etree._Element) -> int:
    """Add deterministic intents to a normalised MathML tree.

    Returns the number of ``intent`` attributes present after annotation.
    """

    split_matrix_subscripts(root)
    _annotate_function_applications(root)
    _annotate_bare_functions(root)
    _annotate_linear_fractions(root)
    _annotate_structures(root)
    split_matrix_subscripts(root)
    return sum(
        1
        for element in root.iter()
        if is_element(element) and element.get("intent") is not None
    )
