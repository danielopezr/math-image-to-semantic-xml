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


def _is_symbol(element: etree._Element) -> bool:
    """A one-character identifier, optionally with a subscript (y, T, y_0)."""

    if localname(element.tag) == "msub":
        children = element_children(element)
        return len(children) == 2 and _is_symbol(children[0])
    text = element_text(element)
    return _is_mi(element) and len(text) == 1 and text not in FUNCTION_CONCEPTS


def _is_single_variable_call(children: list[etree._Element], index: int) -> bool:
    """``T(x)``, ``y_0(t)``, ``Q(x)``: a symbol applied to exactly one variable.

    With a single variable between the parentheses the function reading is
    by far the most common one in textbooks, whereas ``a(b+c)`` or ``n(n+1)``
    (an expression inside) are products.
    """

    if index + 3 >= len(children):
        return False
    name, opener, variable, closer = children[index : index + 4]
    return (
        _is_symbol(name)
        and _is_mo(opener, "(")
        and _is_mo(closer, ")")
        and _is_mi(variable)
        and len(element_text(variable)) == 1
    )


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
        if name_info is None and _is_single_variable_call(children, index):
            name_info = (candidate, None)
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


_PRIME_ORDER = {"′": 1, "″": 2, "‴": 3, "⁗": 4, "'": 1}


def _prime_order(exponent: etree._Element) -> int | None:
    text = element_text(exponent)
    if not text or any(char not in _PRIME_ORDER for char in text):
        return None
    return sum(_PRIME_ORDER[char] for char in text)


def _parenthesised_order(exponent: etree._Element) -> etree._Element | None:
    """The ``n`` element of a ``y^{(n)}`` exponent."""

    children = element_children(exponent)
    if (
        localname(exponent.tag) == "mrow"
        and len(children) == 3
        and _is_mo(children[0], "(")
        and _is_mo(children[2], ")")
        and localname(children[1].tag) in {"mn", "mi"}
    ):
        return children[1]
    return None


def _annotate_derivative(element, base, exponent, variable: str) -> bool:
    """y', y'', y''' and y^{(n)} as derivative($y, _x, n)."""

    if localname(base.tag) != "mi" or not _is_generic_or_variable(base):
        return False
    literal = "_" + variable
    order = _prime_order(exponent)
    if order is not None:
        if _set_arg(base, "y"):
            element.set("intent", f"derivative($y,{literal},{order})")
            return True
        return False
    order_element = _parenthesised_order(exponent)
    if (
        order_element is not None
        and _set_arg(base, "y")
        and _set_arg(order_element, "n")
    ):
        element.set("intent", f"derivative($y,{literal},$n)")
        return True
    return False


def _is_generic_or_variable(element: etree._Element) -> bool:
    text = element_text(element)
    return bool(text) and text not in FUNCTION_CONCEPTS and len(text) == 1


def _annotate_msup(
    element: etree._Element,
    base,
    exponent,
    derivative_variable: str | None = None,
) -> None:
    if _not_a_power_base(base):
        # sin^2(x) is not the power of the symbol "sin"; leave it unannotated
        # rather than emit a misleading power intent.
        return
    if derivative_variable and _annotate_derivative(
        element, base, exponent, derivative_variable
    ):
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


def _annotate_structures(
    root: etree._Element,
    *,
    default_structures: bool = True,
    derivative_variable: str | None = None,
) -> None:
    """Annotate fractions, scripts and roots.

    With ``default_structures=False`` a plain fraction, subscript or root is
    left unannotated: its default reading is already right, and the intent
    would only repeat it (the W3C principle and the dataset convention).
    """

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
            if default_structures and _set_args(children, ["num", "den"]):
                element.set("intent", "fraction($num,$den)")

        elif tag == "msup" and len(children) == 2:
            _annotate_msup(element, children[0], children[1], derivative_variable)

        elif tag == "msubsup" and len(children) == 3:
            _annotate_msubsup(element, children)

        elif tag == "msub" and len(children) == 2:
            base = children[0]
            if element_text(base) in FUNCTION_CONCEPTS or _is_operator_name(base):
                # log_2(x) is a logarithm with a base and lim_{x->0} a limit,
                # not indexed symbols.
                continue
            if localname(base.tag) == "mo" or not default_structures:
                continue
            if _set_args(children, ["base", "sub"]):
                element.set("intent", "indexed-by($base,$sub)")

        elif tag in {"msqrt", "mroot"} and not default_structures:
            continue

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


# ---------------------------------------------------------------------------
# inner products
# ---------------------------------------------------------------------------
_ANGLE_PAIRS = {"\u27e8": "\u27e9", "\u2329": "\u232a"}


def _pair_at(
    children: list[etree._Element], index: int, round_brackets: bool
) -> tuple[etree._Element, etree._Element] | None:
    """Match ``open X , Y close`` starting at ``index``."""

    if index + 4 >= len(children):
        return None
    opener, first, comma, second, closer = children[index : index + 5]
    open_text, close_text = element_text(opener), element_text(closer)
    angle = _ANGLE_PAIRS.get(open_text) == close_text
    round_pair = round_brackets and open_text == "(" and close_text == ")"
    if not (angle or round_pair):
        return None
    if localname(opener.tag) not in {"mo", "mi"}:
        return None
    if localname(closer.tag) not in {"mo", "mi"}:
        return None
    if not _is_mo(comma, ","):
        return None
    for operand in (first, second):
        if localname(operand.tag) == "mo" or operand.get("arg") is not None:
            return None
    if round_pair and index > 0:
        previous = children[index - 1]
        if localname(previous.tag) in {"mi", "msub", "msup", "msubsup"}:
            return None  # T(x, y), f_i(a, b): an application, not a pair
    return first, second


def _annotate_inner_products(root: etree._Element, round_brackets: bool) -> None:
    """<u, v> always, and (u, v) when the context is about inner products."""

    rows = [
        node
        for node in root.iter()
        if is_element(node) and localname(node.tag) == "mrow"
    ]
    for mrow in rows:
        if mrow.get("intent") is not None:
            continue
        index = 0
        while True:
            children = element_children(mrow)
            if index >= len(children):
                break
            pair = _pair_at(children, index, round_brackets)
            if pair is None:
                index += 1
                continue
            nodes = children[index : index + 5]
            if index == 0 and len(children) == 5:
                target = mrow
            else:
                target = new_element("mrow")
                mrow.insert(mrow.index(nodes[0]), target)
                for node in nodes:
                    target.append(node)
            pair[0].set("arg", "arg1")
            pair[1].set("arg", "arg2")
            target.set("intent", "inner-product($arg1,$arg2)")
            if target is mrow:
                break
            index += 1


# ---------------------------------------------------------------------------
# implicit multiplication
# ---------------------------------------------------------------------------
_FACTOR_TAGS = frozenset(
    {"mi", "mn", "msup", "msub", "msubsup", "mfrac", "msqrt", "mroot", "mover", "mrow"}
)
_INVISIBLE_TIMES = "⁢"
_OPENERS = {"(": ")", "[": "]"}


def _is_factor(element: etree._Element) -> bool:
    tag = localname(element.tag)
    if tag not in _FACTOR_TAGS or element.get("arg") is not None:
        return False
    text = element_text(element)
    if tag == "mi" and (not text or text in FUNCTION_CONCEPTS):
        return False
    if tag == "mi" and len(text) > 1 and text.isalpha():
        # dim, ker, sgn...: an operator name applied to what follows, not a
        # factor of a product.
        return False
    if tag in {"msup", "msub", "msubsup", "mover"}:
        children = element_children(element)
        if children and _not_a_power_base(children[0]):
            return False  # sin^2, lim_x, sum^n: not plain factors
    if tag == "mrow":
        # Only groups that already stand for one value (an annotated
        # application such as f(x)); other rows are scanned on their own.
        return element.get("intent") is not None
    return True


def _paren_group_end(children: list[etree._Element], start: int) -> int | None:
    opener = element_text(children[start])
    closer = _OPENERS.get(opener)
    if localname(children[start].tag) != "mo" or closer is None:
        return None
    depth = 0
    for index in range(start, len(children)):
        child = children[index]
        if localname(child.tag) != "mo":
            continue
        text = element_text(child)
        if text == opener:
            depth += 1
        elif text == closer:
            depth -= 1
            if depth == 0:
                return index if index > start + 1 else None
        elif text == "," and depth == 1:
            # (x, y): an argument list, a pair or an interval, not a factor.
            return None
    return None


def _is_differential(children: list[etree._Element], index: int) -> bool:
    child = children[index]
    return (
        localname(child.tag) == "mi"
        and element_text(child) in {"d", "ⅆ"}
        and index + 1 < len(children)
        and localname(children[index + 1].tag) == "mi"
    )


def _factor_runs(children: list[etree._Element]) -> list[list[tuple[int, int]]]:
    """Runs of adjacent factors as lists of (start, end) index spans."""

    runs: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    index = 0
    while index < len(children):
        child = children[index]
        if localname(child.tag) == "mo" and element_text(child) == _INVISIBLE_TIMES:
            index += 1
            continue
        end = _paren_group_end(children, index)
        if end is not None and index == 0 and end == len(children) - 1:
            # The row *is* one parenthesised group, e.g. (2x+1): scan inside.
            end = None
        if end is not None and not any(c.get("arg") for c in children[index : end + 1]):
            current.append((index, end))
            index = end + 1
            continue
        if _is_differential(children, index) or not _is_factor(child):
            if len(current) >= 2:
                runs.append(current)
            current = []
            index += 1
            continue
        if (
            current
            and localname(child.tag) == "mn"
            and current[-1][0] == current[-1][1]
            and localname(children[current[-1][0]].tag) == "mn"
        ):
            # Two adjacent numbers are never a product: they are separate
            # digits or indexes (the "1 1" of a matrix entry a_11).
            if len(current) >= 2:
                runs.append(current)
            current = []
        current.append((index, index))
        index += 1
    if len(current) >= 2:
        runs.append(current)
    return runs


_SUBSCRIPT_PARENTS = frozenset({"msub", "msubsup", "munder", "munderover"})


def _in_subscript(element: etree._Element) -> bool:
    """True inside the subscript (second child) of a scripted element."""

    current = element
    parent = current.getparent()
    while parent is not None:
        if localname(parent.tag) in _SUBSCRIPT_PARENTS:
            siblings = element_children(parent)
            if len(siblings) > 1 and siblings[1] is current:
                return True
        current, parent = parent, parent.getparent()
    return False


def _times_intent(names: list[str], nary: bool) -> str:
    if nary or len(names) == 2:
        return "times(" + ",".join(f"${name}" for name in names) + ")"
    return f"times(${names[0]},{_times_intent(names[1:], nary)})"


def _annotate_implicit_times_in(mrow: etree._Element, nary: bool) -> list:
    children = element_children(mrow)
    runs = _factor_runs(children)
    if _in_subscript(mrow):
        # In a subscript, adjacent letters are an index list (a_ij), not a
        # product; only a numeric coefficient (a_{2n}) marks a product.
        runs = [
            run
            for run in runs
            if run[0][0] == run[0][1] and localname(children[run[0][0]].tag) == "mn"
        ]
    created: list[etree._Element] = []
    # Process from the right so earlier indexes stay valid.
    for run in reversed(runs):
        factors: list[etree._Element] = []
        for start, end in run:
            if start == end:
                factors.append(children[start])
            else:
                group = new_element("mrow")
                mrow.insert(mrow.index(children[start]), group)
                for node in children[start : end + 1]:
                    group.append(node)
                factors.append(group)
                created.append(group)
        whole = (
            run[0][0] == 0
            and run[-1][1] == len(children) - 1
            and mrow.get("intent") is None
        )
        if whole:
            target = mrow
        else:
            target = new_element("mrow")
            mrow.insert(mrow.index(factors[0]), target)
            for factor in factors:
                target.append(factor)
        names = [f"arg{position}" for position in range(1, len(factors) + 1)]
        if nary or len(factors) == 2:
            for factor, name in zip(factors, names):
                factor.set("arg", name)
            target.set("intent", _times_intent(names, True))
        else:
            # Without n-ary times: times(a, times(b, c)) as nested rows.
            current = target
            remaining = factors
            while len(remaining) > 2:
                remaining[0].set("arg", "arg1")
                rest = new_element("mrow")
                current.insert(current.index(remaining[1]), rest)
                for factor in remaining[1:]:
                    rest.append(factor)
                rest.set("arg", "arg2")
                current.set("intent", "times($arg1,$arg2)")
                current, remaining = rest, remaining[1:]
            remaining[0].set("arg", "arg1")
            remaining[1].set("arg", "arg2")
            current.set("intent", "times($arg1,$arg2)")
    return created


def _annotate_implicit_times(root: etree._Element, nary: bool) -> None:
    queue = [
        node
        for node in root.iter()
        if is_element(node) and localname(node.tag) == "mrow"
    ]
    seen: set[int] = set()
    while queue:
        mrow = queue.pop(0)
        if id(mrow) in seen:
            continue
        seen.add(id(mrow))
        # Parenthesised groups created here are scanned too: (2x+1) has its
        # own product inside.
        queue.extend(_annotate_implicit_times_in(mrow, nary))


_TENDS_TO_ARROWS = frozenset({"→", "⟶"})
_SIGNS = frozenset({"+", "-", "−"})


def _is_limit_variable(element: etree._Element) -> bool:
    text = element_text(element)
    return _is_mi(element) and len(text) == 1 and text.isalpha() and text.islower()


def _is_limit_value(element: etree._Element) -> bool:
    """A number, infinity or a single lowercase letter (x → a)."""

    tag = localname(element.tag)
    if tag == "mn":
        return True
    return tag == "mi" and (element_text(element) == "∞" or _is_limit_variable(element))


def _signed_value(children: list[etree._Element]) -> bool:
    return (
        len(children) == 2
        and _is_mo(children[0])
        and element_text(children[0]) in _SIGNS
        and _is_limit_value(children[1])
    )


def _annotate_tends_to(root: etree._Element) -> None:
    """Read ``x → a``, ``x → 0`` and ``x → +∞`` as ``tends-to($var, $value)``.

    Only a whole row of the form *letter, arrow, value* is annotated; maps
    such as ``ℝ² → ℝ²`` or ``f: A → B`` (capital letters) keep their reading.
    """

    for mrow in list(root.iter()):
        if not is_element(mrow) or localname(mrow.tag) != "mrow":
            continue
        if mrow.get("intent") is not None:
            continue
        children = element_children(mrow)
        if len(children) < 3:
            continue
        variable, arrow, *value = children
        if not _is_mo(arrow) or element_text(arrow) not in _TENDS_TO_ARROWS:
            continue
        if not _is_limit_variable(variable) or variable.get("arg") is not None:
            continue
        if any(child.get("arg") is not None for child in value):
            continue
        if len(value) == 1 and localname(value[0].tag) == "mrow":
            if not _signed_value(element_children(value[0])):
                continue
            target = value[0]
        elif len(value) == 1 and _is_limit_value(value[0]):
            target = value[0]
        elif _signed_value(value):
            target = new_element("mrow")
            value[0].addprevious(target)
            for child in value:
                target.append(child)
        else:
            continue
        variable.set("arg", "var")
        target.set("arg", "value")
        mrow.set("intent", "tends-to($var,$value)")


def annotate_tree(
    root: etree._Element,
    *,
    implicit_times: bool = False,
    nary_associative: bool = True,
    default_structures: bool = True,
    derivative_variable: str | None = None,
    inner_product_pairs: bool = False,
) -> int:
    """Add deterministic intents to a normalised MathML tree.

    ``implicit_times`` also annotates implicit multiplication (``2x``,
    ``i(1+r)``) as ``times(...)``: flat with all factors when
    ``nary_associative`` is true, nested pairs otherwise.

    ``default_structures=False`` leaves plain fractions, subscripts and roots
    unannotated.  ``derivative_variable`` (e.g. ``"x"``) reads primes and
    ``y^{(n)}`` as ``derivative($y, _x, n)``.  ``inner_product_pairs`` also
    reads a round ``(u, v)`` as an inner product (``<u, v>`` always is).

    Returns the number of ``intent`` attributes present after annotation.
    """

    split_matrix_subscripts(root)
    _annotate_function_applications(root)
    _annotate_bare_functions(root)
    _annotate_inner_products(root, inner_product_pairs)
    _annotate_tends_to(root)
    if implicit_times:
        _annotate_implicit_times(root, nary_associative)
    _annotate_linear_fractions(root)
    _annotate_structures(
        root,
        default_structures=default_structures,
        derivative_variable=derivative_variable,
    )
    split_matrix_subscripts(root)
    return sum(
        1
        for element in root.iter()
        if is_element(element) and element.get("intent") is not None
    )


_DEFAULT_INTENTS = {
    "mfrac": "fraction",
    "msub": "indexed-by",
    "msqrt": "root",
    "mroot": "root",
}


def strip_default_intents(root: etree._Element) -> int:
    """Remove intents that only repeat the default reading of an element.

    A ``fraction`` on an ``mfrac``, ``indexed-by`` on an ``msub`` and ``root``
    on ``msqrt``/``mroot`` add nothing a screen reader does not already say.
    Linear fractions such as ``a/b`` (an ``mrow``) are kept.  Returns the
    number of intents removed; orphan ``arg`` attributes should be pruned
    afterwards.
    """

    removed = 0
    for element in root.iter():
        if not is_element(element):
            continue
        concept = _DEFAULT_INTENTS.get(localname(element.tag))
        intent = element.get("intent") or ""
        if concept and intent.split("(", 1)[0].strip() == concept:
            del element.attrib["intent"]
            removed += 1
    return removed
