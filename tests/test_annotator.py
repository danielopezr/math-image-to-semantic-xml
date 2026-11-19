from lxml import etree

from image_to_mathml.mathml.annotator import annotate_tree
from image_to_mathml.mathml.converter import LatexToMathML
from image_to_mathml.mathml.dom import localname
from image_to_mathml.mathml.normalizer import (
    normalize_mathml,
    parse_and_normalize,
    serialize_mathml,
)


def _annotated(latex: str) -> str:
    base = LatexToMathML().convert(latex)
    root = parse_and_normalize(base)
    annotate_tree(root)
    return serialize_mathml(root)


def test_annotates_fraction():
    result = _annotated(r"\frac{a}{b}")
    assert 'intent="fraction($num,$den)"' in result
    assert 'arg="num"' in result
    assert 'arg="den"' in result


def test_annotates_power_and_root():
    assert 'intent="power($base,$exp)"' in _annotated("x^2")
    result = _annotated(r"\sqrt{x}")
    assert 'intent="root($radicand, 2)"' in result
    assert 'arg="radicand"' in result


def test_annotates_generic_function_exactly_as_requested():
    result = _annotated("f(x)")
    assert 'intent="$name($var)"' in result
    assert '<mi arg="name">f</mi>' in result
    assert '<mi arg="var">x</mi>' in result


def test_annotates_named_and_gamma_functions():
    assert 'intent="sine($arg)"' in _annotated(r"\sin(x)")
    assert 'intent="gamma-function($arg)"' in _annotated(r"\Gamma(x)")


def test_greek_letters_are_not_guessed_as_special_functions():
    # beta is usually a variable, and the beta function B(x,y) takes two
    # arguments; a bare Gamma x is a product.
    assert "beta-function" not in _annotated(r"\beta(x)")
    assert "beta-function" not in _annotated(r"\beta x")
    assert "gamma-function" not in _annotated(r"\Gamma x")


def test_products_with_parentheses_are_not_function_applications():
    for latex in (r"a(b+c)", r"2x(x+1)", r"k(x-1)"):
        assert "$name($var)" not in _annotated(latex), latex


def test_conventional_and_operator_names_are_function_applications():
    assert 'intent="$name($var)"' in _annotated(r"g(t)")
    assert 'intent="$name($var)"' in _annotated(r"\operatorname{sgn}(x)")


def test_primed_function_name_is_a_generic_application_not_a_power():
    result = _annotated(r"f'(x)")
    assert "power(" not in result
    assert 'intent="$name($var)"' in result
    assert '<msup arg="name">' in result


def test_non_exponent_superscripts_are_not_powers():
    for latex in (r"A^{T}", r"z^{*}", r"A^{\dagger}", r"y''"):
        assert "power(" not in _annotated(latex), latex
    assert 'intent="transpose($arg)"' in _annotated(r"A^{\top}")


def test_operator_names_written_as_mo_are_recognised():
    assert 'intent="determinant($arg)"' in _annotated(r"\det(A)")
    assert 'intent="determinant($arg)"' in _annotated(r"\det A")
    assert 'intent="max($arg)"' in _annotated(r"\max(a,b)")


def test_limits_are_not_indexed_symbols():
    result = _annotated(r"\lim_{x\to 0} f(x)")
    assert "indexed-by" not in result


def test_logarithm_with_base():
    for latex in (r"\log_2(x)", r"\log_{2} x"):
        result = _annotated(latex)
        assert 'intent="logarithm($arg,$base)"' in result, latex
        assert 'arg="base"' in result


def test_indexed_power_and_large_operators():
    assert 'intent="power(indexed-by($base,$sub),$exp)"' in _annotated(r"x_i^2")
    assert "intent=" not in _annotated(r"\sum_{i=1}^{n} i")
    assert "power(" not in _annotated(r"\int_0^1 t\,dt")


def test_annotates_bare_named_functions_when_unambiguous():
    assert 'intent="sine($arg)"' in _annotated(r"\sin x")
    assert 'intent="logarithm($arg)"' in _annotated(r"\log x")
    assert 'intent="sine($arg)"' in _annotated(r"a+\sin x")


def test_does_not_guess_ambiguous_bare_function_argument():
    result = _annotated(r"\sin 2x")
    assert "sine($arg)" not in result


def test_does_not_mislabel_function_powers_or_log_bases():
    assert "power($base,$exp)" not in _annotated(r"\sin^2 x")
    assert "indexed-by($base,$sub)" not in _annotated(r"\log_2 x")


def test_annotates_exponential_not_power_for_e_base():
    result = _annotated("e^{-x}")
    assert 'intent="exponential($exp)"' in result


def test_does_not_add_unnecessary_mrow_for_plain_fraction():
    base = normalize_mathml(LatexToMathML().convert(r"\frac{a}{b}"))
    root = parse_and_normalize(base)
    annotate_tree(root)
    result = serialize_mathml(root)
    assert result.count("<mrow") == 1


def test_annotates_indexed_generic_function_names():
    subscript = _annotated(r"f_{i}(x)")
    assert 'intent="$name($var)"' in subscript
    assert '<msub arg="name">' in subscript
    assert '<mi arg="var">x</mi>' in subscript

    expression = _annotated(r"f_{i+1}(x)")
    assert 'intent="$name($var)"' in expression
    assert '<msub arg="name">' in expression
    assert '<mi arg="var">x</mi>' in expression


def test_does_not_annotate_basic_operators_with_default_meaning():
    result = _annotated("a+b=c")
    assert 'intent="plus"' not in result
    assert 'intent="equals"' not in result
    assert "intent=" not in result

    product = _annotated(r"a \times b")
    assert 'intent="times"' not in product
    assert "intent=" not in product

    division = _annotated("a/b")
    assert 'intent="divided-by"' not in division
    assert 'intent="fraction($num,$den)"' in division
    assert 'arg="num"' in division
    assert 'arg="den"' in division

    compound = _annotated("a/b+c")
    assert "intent=" not in compound


def test_annotator_preserves_matrix_subscript_digit_separation():
    result = _annotated(
        r"\begin{matrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{matrix}"
    )

    assert "<mn>11</mn>" not in result
    assert "<mn>12</mn>" not in result
    assert result.count('<mrow arg="sub">') == 4
    assert result.count('intent="indexed-by($base,$sub)"') == 4


def test_annotator_splits_matrix_subscripts_before_annotating():
    root = etree.fromstring(
        b"<math><mrow><mtable><mtr><mtd>"
        b"<msub><mi>a</mi><mn>11</mn></msub>"
        b"</mtd></mtr></mtable></mrow></math>"
    )

    annotate_tree(root)

    msub = next(element for element in root.iter() if localname(element.tag) == "msub")
    assert msub.get("intent") == "indexed-by($base,$sub)"
    subscript = list(msub)[1]
    assert localname(subscript.tag) == "mrow"
    assert [localname(child.tag) for child in subscript] == ["mn", "mn"]
    assert [child.text for child in subscript] == ["1", "1"]


def test_annotates_a_variable_tending_to_a_value():
    result = " ".join(_annotated(r"x \to +\infty").split())
    assert '<mrow intent="tends-to($var,$value)"> <mi arg="var">x</mi> <mo>→</mo>' in (
        result
    )
    assert '<mrow arg="value"> <mo>+</mo> <mi>∞</mi> </mrow>' in result
    assert 'intent="tends-to($var,$value)"' in _annotated(r"x \to 0")
    assert 'intent="tends-to($var,$value)"' in _annotated(r"\lim_{n \to \infty} a_n")


def test_maps_between_sets_are_not_read_as_tends_to():
    assert "tends-to" not in _annotated(r"T: \mathbb{R}^2 \to \mathbb{R}^2")
    assert "tends-to" not in _annotated(r"f: A \to B")
