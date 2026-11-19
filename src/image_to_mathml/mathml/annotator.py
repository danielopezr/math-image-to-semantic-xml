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

from .calculus import annotate_calculus
from .corrections import ELLIPSIS_SYMBOLS, split_matrix_subscripts
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
    "𝛤": "gamma-function",  # mathematical italic capital gamma
    "gamma": "gamma-function",
    "erf": "error-function",
}

# Named functions that are only recognised when followed by parentheses.  A
# bare "Γ x" is far more often a product with the symbol Γ than Γ(x).
_PARENTHESIS_ONLY_FUNCTIONS = frozenset({"Γ", "𝛤", "gamma"})

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
    if _bare_function(following) is not None or _is_differential(
        children, argument_index + 1
    ):
        # sin θ cos θ, cos x dx: the next function or the differential ends it.
        return True
    if localname(following.tag) != "mo":
        return False
    return element_text(following) in _BARE_ARGUMENT_FOLLOWERS


_PRODUCT_FACTOR_TAGS = frozenset({"mi", "mn", "msup", "msub"})


def _product_argument_end(children: list[etree._Element], start: int) -> int | None:
    """Last index of an implicit product such as ``kx`` in ``cos kx``.

    The factors are plain tokens or scripts, and the product must end where an
    unparenthesised argument ends (an operator such as + or =, another
    function, a differential or the end of the row); otherwise ``None``.
    """

    end = start
    while (
        end + 1 < len(children)
        and localname(children[end + 1].tag) in _PRODUCT_FACTOR_TAGS
        and children[end + 1].get("intent") is None
        and children[end + 1].get("arg") is None
        and _bare_function(children[end + 1]) is None
        and not _is_differential(children, end + 1)
    ):
        end += 1
    if end == start or not all(
        localname(child.tag) in _PRODUCT_FACTOR_TAGS
        for child in children[start : end + 1]
    ):
        return None
    if not _bare_argument_is_unambiguous(children, end):
        return None
    return end


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
        if concept is None or not _is_bare_argument(argument):
            index += 1
            continue
        if not _bare_argument_is_unambiguous(children, index + 1):
            # cos kx: the argument is the whole implicit product.
            end = _product_argument_end(children, index + 1)
            if end is None:
                index += 1
                continue
            argument = new_element("mrow")
            mrow.insert(mrow.index(children[index + 1]), argument)
            for factor in children[index + 1 : end + 1]:
                argument.append(factor)
            children = element_children(mrow)

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


def group_function_products(mrow: etree._Element) -> list[etree._Element]:
    """Group ``cos kx`` as ``<mrow>cos <mrow>k x</mrow></mrow>``.

    Used before the LLM so it sees the function and its whole argument as one
    element instead of a flat cos·k·x.  Returns the ``<mrow>`` elements
    created.
    """

    created: list[etree._Element] = []
    children = element_children(mrow)
    index = 0
    while index + 2 < len(children):
        candidate = children[index]
        if (
            _bare_function(candidate) is None
            or not _is_bare_argument(children[index + 1])
            or _bare_argument_is_unambiguous(children, index + 1)
        ):
            index += 1
            continue
        end = _product_argument_end(children, index + 1)
        if end is None:
            index += 1
            continue
        whole_row = index == 0 and end == len(children) - 1
        product = new_element("mrow")
        mrow.insert(mrow.index(children[index + 1]), product)
        for factor in children[index + 1 : end + 1]:
            product.append(factor)
        created.append(product)
        if whole_row:
            # The row already is the application: cos kx alone in a numerator.
            return created
        application = new_element("mrow")
        mrow.insert(mrow.index(candidate), application)
        application.append(candidate)
        application.append(product)
        created.append(application)
        children = element_children(mrow)
        index = mrow.index(application) + 1
    return created


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


# ---------------------------------------------------------------------------
# matrices, column vectors and |matrix| determinants
# ---------------------------------------------------------------------------
_MATRIX_FENCES = {"[": "]", "(": ")"}


def _table_shape(table: etree._Element) -> tuple[int, int]:
    rows = [row for row in element_children(table) if localname(row.tag) == "mtr"]
    columns = max((len(element_children(row)) for row in rows), default=0)
    return len(rows), columns


def _fenced_wrapper(table: etree._Element) -> etree._Element | None:
    """The ``<mrow>`` holding exactly ``fence table fence``, created if needed."""

    parent = table.getparent()
    if parent is None:
        return None
    siblings = element_children(parent)
    index = siblings.index(table)
    if index == 0 or index == len(siblings) - 1:
        return None
    opener, closer = siblings[index - 1], siblings[index + 1]
    if localname(parent.tag) == "mrow" and len(siblings) == 3:
        return parent
    wrapper = new_element("mrow")
    parent.insert(parent.index(opener), wrapper)
    for node in (opener, table, closer):
        wrapper.append(node)
    return wrapper


def _fences_of(table: etree._Element) -> tuple[str, str] | None:
    parent = table.getparent()
    if parent is None:
        return None
    siblings = element_children(parent)
    index = siblings.index(table)
    if index == 0 or index == len(siblings) - 1:
        return None
    opener, closer = siblings[index - 1], siblings[index + 1]
    if localname(opener.tag) not in {"mo", "mi"} or localname(closer.tag) not in {
        "mo",
        "mi",
    }:
        return None
    return element_text(opener), element_text(closer)


def _clear(element: etree._Element, *names: str) -> None:
    for name in names:
        if element.get(name) is not None:
            del element.attrib[name]


def _annotate_tables(root: etree._Element, *, override: bool) -> int:
    """``[table]`` is a matrix (``:matrix``) or, with one column, a vector;
    ``|table|`` is the determinant of the matrix.

    With ``override`` other intents on the table or its fenced group (an LLM
    reading) are replaced; otherwise only unannotated elements are touched.
    Returns the number of tables annotated.
    """

    changed = 0
    for table in [
        node
        for node in root.iter()
        if is_element(node) and localname(node.tag) == "mtable"
    ]:
        fences = _fences_of(table)
        if fences is None:
            continue
        rows, columns = _table_shape(table)
        if fences == ("|", "|") and rows >= 2 and columns >= 2:
            concept, role, matrix = "determinant($mat)", "mat", True
        elif _MATRIX_FENCES.get(fences[0]) == fences[1] and columns >= 2:
            concept, role, matrix = None, None, True
        elif _MATRIX_FENCES.get(fences[0]) == fences[1] and columns == 1 and rows >= 2:
            concept, role, matrix = "vector($vec)", "vec", False
        else:
            continue
        table_intent = (table.get("intent") or "").replace(" ", "")
        if (
            not override
            and table.get("intent") is not None
            and table_intent != ":matrix"
        ):
            continue
        if role is not None and concept is not None:
            if table.get("arg") not in (None, role):
                continue
            wrapper = _fenced_wrapper(table)
            if wrapper is None:
                continue
            current = (wrapper.get("intent") or "").replace(" ", "")
            if current != concept.replace(" ", ""):
                if current and not override:
                    continue
                wrapper.set("intent", concept)
            table.set("arg", role)
        elif override:
            # A matrix(...) reading of the fenced group repeats :matrix.
            parent = table.getparent()
            if (
                parent is not None
                and len(element_children(parent)) == 3
                and (parent.get("intent") or "").startswith("matrix")
            ):
                _clear(parent, "intent")
        if matrix:
            table.set("intent", ":matrix")
        else:
            _clear(table, "intent")
        changed += 1
    return changed


# ---------------------------------------------------------------------------
# mixed fractions: 3 1/2
# ---------------------------------------------------------------------------
def _integer(element: etree._Element) -> int | None:
    text = element_text(element)
    if localname(element.tag) != "mn" or not text.isdigit():
        return None
    return int(text)


def _is_proper_fraction(element: etree._Element) -> bool:
    if localname(element.tag) != "mfrac":
        return False
    parts = element_children(element)
    if len(parts) != 2:
        return False
    numerator, denominator = (_integer(part) for part in parts)
    return numerator is not None and denominator is not None and numerator < denominator


def _annotate_mixed_fractions(root: etree._Element, *, override: bool) -> int:
    """``3\\frac{1}{2}`` is the mixed number three and a half, not a product."""

    changed = 0
    for mrow in [
        node
        for node in root.iter()
        if is_element(node) and localname(node.tag) == "mrow"
    ]:
        children = element_children(mrow)
        index = 0
        while index + 1 < len(children):
            whole, fraction = children[index], children[index + 1]
            if _integer(whole) is None or not _is_proper_fraction(fraction):
                index += 1
                continue
            if index > 0 and localname(children[index - 1].tag) in {"mn", "mi"}:
                index += 1
                continue
            parts_free = all(
                part.get("intent") is None and part.get("arg") is None
                for part in (whole, fraction)
            )
            pair_row = len(children) == 2
            if pair_row and (mrow.get("intent") is None or override):
                target = mrow
            elif parts_free:
                target = new_element("mrow")
                mrow.insert(mrow.index(whole), target)
                target.append(whole)
                target.append(fraction)
            else:
                index += 1
                continue
            if not parts_free:
                if not override:
                    index += 1
                    continue
                for part in (whole, fraction):
                    _clear(part, "intent", "arg")
            target.set("intent", "mixed-fraction($int,$frac)")
            whole.set("arg", "int")
            fraction.set("arg", "frac")
            changed += 1
            children = element_children(mrow)
            index += 1
    return changed


def enforce_structures(root: etree._Element) -> int:
    """Readings that win over the LLM: matrices, vectors, ``|A|`` and mixed
    numbers.  Returns the number of elements re-annotated."""

    return _annotate_tables(root, override=True) + _annotate_mixed_fractions(
        root, override=True
    )


def enforce_derivatives(root: etree._Element, variable: str) -> int:
    """Read every y', y'', y''' and y^{(n)} as a derivative, whatever it had.

    In a calculus context these forms are unambiguous, but the LLM tends to
    read ``y^{(4)}`` (and then every other derivative of the equation) as a
    power.  The intent another reading left on the ``msup`` and the ``arg``
    and ``intent`` attributes on its parts are replaced by the deterministic
    derivative.  Returns the number of elements re-annotated.
    """

    changed = 0
    for element in list(root.iter()):
        if not is_element(element) or localname(element.tag) != "msup":
            continue
        children = element_children(element)
        if len(children) != 2:
            continue
        base, exponent = children
        if localname(base.tag) != "mi" or not _is_generic_or_variable(base):
            continue
        if _prime_order(exponent) is None and _parenthesised_order(exponent) is None:
            continue
        current = (element.get("intent") or "").replace(" ", "")
        if current.startswith("derivative("):
            continue
        for part in (element, base, *exponent.iter()):
            if not is_element(part):
                continue
            for name in ("intent", "arg") if part is not element else ("intent",):
                if part.get(name) is not None:
                    del part.attrib[name]
        if _annotate_derivative(element, base, exponent, variable):
            changed += 1
    return changed


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
    """``d`` before a variable: ``dx``, and ``dt^n`` in ``d^n/dt^n``."""

    child = children[index]
    if (
        localname(child.tag) != "mi"
        or element_text(child) not in {"d", "ⅆ"}
        or index + 1 >= len(children)
    ):
        return False
    following = children[index + 1]
    tag = localname(following.tag)
    if tag == "mi":
        return True
    if tag in {"msup", "msub"}:
        parts = element_children(following)
        return bool(parts) and localname(parts[0].tag) == "mi"
    return False


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


def _annotate_ellipses(root: etree._Element) -> None:
    """``…`` and ``⋯`` always read as an ellipsis (annotation guide)."""

    for element in root.iter():
        if not is_element(element) or localname(element.tag) != "mo":
            continue
        if element.get("intent") is None and element_text(element) in ELLIPSIS_SYMBOLS:
            element.set("intent", "ellipsis")


def _factorial_operand(
    children: list[etree._Element], bang: int
) -> tuple[int, list[etree._Element]] | None:
    """Return ``(start, argument nodes)`` of the operand before ``children[bang]``.

    The operand is a number, a letter (``n!``, ``5!``), an indexed letter
    (``n_1!``) or a parenthesised group (``(n-k)!``); for a group the argument
    is its content, without the parentheses.  Anything else (``x^2!``) is
    ambiguous and gives None.
    """

    previous = children[bang - 1]
    tag = localname(previous.tag)
    if tag in {"mi", "mn", "msub"} and not (tag == "mi" and not element_text(previous)):
        return bang - 1, [previous]
    if tag == "mrow" and previous.get("intent") is None:
        inner = element_children(previous)
        if (
            len(inner) >= 3
            and _is_mo(inner[0])
            and _is_mo(inner[-1])
            and _OPENERS.get(element_text(inner[0])) == element_text(inner[-1])
        ):
            return bang - 1, [previous]  # an already grouped (n+p)
    if tag != "mo" or element_text(previous) not in {")", "]"}:
        return None
    closer = element_text(previous)
    opener = "(" if closer == ")" else "["
    depth = 0
    for index in range(bang - 1, -1, -1):
        child = children[index]
        if localname(child.tag) != "mo":
            continue
        text = element_text(child)
        if text == closer:
            depth += 1
        elif text == opener:
            depth -= 1
            if depth == 0:
                content = children[index + 1 : bang - 1]
                return (index, content) if content else None
    return None


def _annotate_factorials(root: etree._Element) -> None:
    """Read ``n!`` and ``(n-k)!`` as ``factorial($var)`` (annotation guide).

    The intent goes on the group formed by the operand and ``!``. Double
    factorials (``n!!``) and ``!`` after anything but a number, a letter or a
    parenthesised group are left alone.
    """

    for mrow in list(root.iter()):
        if not is_element(mrow) or localname(mrow.tag) not in {"mrow", "math"}:
            continue
        children = element_children(mrow)
        bangs = [
            index
            for index, child in enumerate(children)
            if index > 0 and _is_mo(child) and element_text(child) == "!"
        ]
        for bang in reversed(bangs):
            children = element_children(mrow)
            mark = children[bang]
            neighbours = children[bang - 1 : bang] + children[bang + 1 : bang + 2]
            if any(_is_mo(n) and element_text(n) == "!" for n in neighbours):
                continue  # n!! or !!
            found = _factorial_operand(children, bang)
            if found is None:
                continue
            start, content = found
            nodes = children[start:bang]
            if any(node.get("arg") is not None for node in [*nodes, mark]):
                continue
            if start == 0 and bang == len(children) - 1 and mrow.get("intent") is None:
                target = mrow
            else:
                target = new_element("mrow")
                nodes[0].addprevious(target)
                for node in [*nodes, mark]:
                    target.append(node)
            if len(content) == 1:
                content[0].set("arg", "var")
            else:
                wrapper = new_element("mrow")
                wrapper.set("arg", "var")
                content[0].addprevious(wrapper)
                for node in content:
                    wrapper.append(node)
            target.set("intent", "factorial($var)")


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
    # Norms, |x|, ∂f/∂x, ∇f and f(x, y) first: they group what the generic
    # rules below would read as products.
    annotate_calculus(root)
    _annotate_function_applications(root)
    _annotate_bare_functions(root)
    _annotate_inner_products(root, inner_product_pairs)
    _annotate_tends_to(root)
    _annotate_ellipses(root)
    _annotate_tables(root, override=False)
    # Before implicit times, so 3 1/2 is not times(3, 1/2) and 2n! is
    # times(2, factorial(n)).
    _annotate_mixed_fractions(root, override=False)
    _annotate_factorials(root)
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
        head = intent.split("(", 1)[0].strip()
        if (
            (concept and head == concept)
            or head in _SELF_READING_CONCEPTS
            or head.replace(" ", "").startswith("vector:prefix")
            or (head == "times" and _has_visible_product(element))
        ):
            del element.attrib["intent"]
            removed += 1
    return removed


# Annotation guide, section 4: large operators with limits read well on their
# own, and so do products with a visible × or ·.  Subscripted symbols such as
# e_k are not annotated as vectors.
_SELF_READING_CONCEPTS = frozenset(
    {
        "integral",
        "definite-integral",
        "indefinite-integral",
        "sum",
        "product",
        "limit",
    }
)
_VISIBLE_PRODUCTS = frozenset({"×", "⋅", "·"})


def _has_visible_product(element: etree._Element) -> bool:
    return any(
        _is_mo(child) and element_text(child) in _VISIBLE_PRODUCTS
        for child in element_children(element)
    )
