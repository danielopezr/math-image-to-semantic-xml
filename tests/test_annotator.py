from lxml import etree

from image_to_mathml.mathml.annotator import (
    annotate_tree,
    enforce_derivatives,
    enforce_structures,
)
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


def test_bare_function_takes_a_whole_implicit_product():
    # The labels read cos kx as cosine(kx), as is usual in mathematics.
    assert "sine($arg)" in _annotated(r"\sin 2x")
    assert "cosine($arg)" in _annotated(r"\frac{\cos kx}{\sqrt{\pi}}")


def test_bare_function_product_stops_at_the_next_function_or_differential():
    result = _annotated(r"\sin \theta \cos \theta")
    assert "sine($arg)" in result and "cosine($arg)" in result
    assert "cosine($arg)" in _annotated(r"\int f(x) \cos kx \, dx")


def test_does_not_guess_a_bare_argument_followed_by_a_parenthesis():
    assert "sine($arg)" not in _annotated(r"\sin x (a+b)")


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


def _flat(latex: str, **options) -> str:
    base = LatexToMathML().convert(latex)
    root = parse_and_normalize(base)
    annotate_tree(root, **options)
    return " ".join(serialize_mathml(root).split())


def test_annotates_a_factorial_on_the_group_with_its_mark():
    result = _flat(r"n!")
    assert '<mrow intent="factorial($var)"> <mi arg="var">n</mi> <mo>!</mo>' in result
    assert 'intent="factorial($var)"> <mn arg="var">5</mn> <mo>!</mo>' in _flat(r"5!")


def test_a_parenthesised_factorial_takes_the_content_as_argument():
    result = _flat(r"\frac{n!}{(n-k)!}")
    assert result.count('intent="factorial($var)"') == 2
    assert (
        '<mrow arg="var"> <mi>n</mi> <mo>−</mo> <mi>k</mi> </mrow> '
        '<mo stretchy="false">)</mo> <mo>!</mo>'
    ) in result


def test_a_factorial_is_one_factor_of_an_implicit_product():
    result = _flat(r"2n!", implicit_times=True)
    assert '<mrow intent="times($arg1,$arg2)"> <mn arg="arg1">2</mn>' in result
    assert '<mrow intent="factorial($var)" arg="arg2">' in result


def test_ambiguous_exclamation_marks_are_not_factorials():
    assert "factorial" not in _flat(r"n!!")
    assert "factorial" not in _flat(r"x^2!")


def test_ellipses_are_annotated_and_are_operators():
    result = _flat(r"a_1, a_2, \ldots, a_n")
    assert '<mo intent="ellipsis">…</mo>' in result
    assert '<mo intent="ellipsis">⋯</mo>' in _flat(r"1 + 2 + \cdots + n")


def test_the_d_of_a_higher_order_differential_is_not_a_factor():
    result = _flat(r"\frac{d^n}{dt^n}", implicit_times=True)
    assert '<mrow arg="den"> <mi>d</mi> <msup intent="power($base,$exp)">' in result
    assert "times" not in result


def test_enforce_derivatives_replaces_another_reading():
    root = parse_and_normalize(
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>'
        '<msup intent="power($base,$exp)"><mi arg="base">y</mi>'
        '<mrow arg="exp"><mo>(</mo><mn>4</mn><mo>)</mo></mrow></msup>'
        '<mo>+</mo><msup intent="power($base,$exp)"><mi arg="base">y</mi>'
        '<mi arg="exp">″</mi></msup><mo>+</mo>'
        '<msup intent="derivative($y,_t,1)"><mi arg="y">y</mi><mi>′</mi></msup>'
        '<mo>+</mo><msup intent="power($base,$exp)"><mi arg="base">x</mi>'
        '<mn arg="exp">2</mn></msup></mrow></math>'
    )

    assert enforce_derivatives(root, "x") == 2
    markup = serialize_mathml(root)
    assert 'intent="derivative($y,_x,$n)"' in markup
    assert 'intent="derivative($y,_x,2)"' in markup
    # An existing derivative and a real power are left alone.
    assert 'intent="derivative($y,_t,1)"' in markup
    assert markup.count("power(") == 1
    assert markup.count('arg="base"') == 1


def test_bracketed_tables_are_matrices_vectors_or_determinants():
    assert 'intent=":matrix"' in _annotated(
        r"\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}"
    )
    vector = _annotated(r"T\left(\begin{bmatrix} x \\ y \end{bmatrix}\right)")
    assert 'intent="vector($vec)"' in vector
    assert 'arg="vec"' in vector
    determinant = _annotated(r"\begin{vmatrix} a & b \\ c & d \end{vmatrix}")
    assert 'intent="determinant($mat)"' in determinant
    assert 'intent=":matrix"' in determinant
    # A table without brackets (a system of equations) is left alone.
    assert "intent" not in _annotated(r"\begin{matrix} a & b \end{matrix}")


def test_mixed_numbers_are_not_products():
    result = _annotated(r"\frac{47}{12} = 3\frac{11}{12}")
    assert 'intent="mixed-fraction($int,$frac)"' in result
    assert "mixed-fraction" not in _annotated(r"3\frac{12}{11}")
    assert "mixed-fraction" not in _annotated(r"x 3\frac{1}{2}")


def test_enforce_structures_replaces_llm_readings():
    root = parse_and_normalize(
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>'
        '<mo>|</mo><mtable intent="determinant($arg)"><mtr arg="arg">'
        "<mtd><mi>a</mi></mtd><mtd><mi>b</mi></mtd></mtr>"
        "<mtr><mtd><mi>c</mi></mtd><mtd><mi>d</mi></mtd></mtr></mtable>"
        '<mo>|</mo><mo>=</mo><mrow intent="times($a,$b)"><mn arg="a">3</mn>'
        '<mfrac arg="b"><mn>1</mn><mn>2</mn></mfrac></mrow></mrow></math>'
    )

    assert enforce_structures(root) == 2
    markup = serialize_mathml(root)
    assert 'intent="determinant($mat)"' in markup
    assert 'intent=":matrix"' in markup
    assert 'intent="mixed-fraction($int,$frac)"' in markup
    assert "times(" not in markup
