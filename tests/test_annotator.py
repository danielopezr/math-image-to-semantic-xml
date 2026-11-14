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
    assert 'intent="beta-function($arg)"' in _annotated(r"\beta(x)")


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
