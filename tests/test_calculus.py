from image_to_mathml.mathml.annotator import annotate_tree, strip_default_intents
from image_to_mathml.mathml.calculus import enforce_partial_derivatives
from image_to_mathml.mathml.converter import LatexToMathML
from image_to_mathml.mathml.normalizer import parse_and_normalize, serialize_mathml
from image_to_mathml.mathml.validator import prune_orphan_args, validate_mathml


def _annotated(latex: str) -> str:
    root = parse_and_normalize(LatexToMathML().convert(latex))
    annotate_tree(root, implicit_times=True, default_structures=False)
    strip_default_intents(root)
    prune_orphan_args(root)
    return serialize_mathml(root)


def _valid(markup: str) -> bool:
    return validate_mathml(markup, allow_open_concepts=True).ok


def test_partial_derivative_of_a_function_of_two_variables():
    result = _annotated(r"\frac{\partial f(x,y)}{\partial x} = 0")
    assert 'intent="partial-derivative($f,1,$v1)"' in result
    assert 'intent="$name($v1, $v2)"' in result
    assert _valid(result)


def test_partial_derivative_orders_and_mixed_variables():
    second = _annotated(r"\frac{\partial^{2} u}{\partial x^{2}}")
    assert 'intent="partial-derivative($f,2,$v1)"' in second
    mixed = _annotated(r"\frac{\partial^{2} u}{\partial x \partial y}")
    assert 'intent="partial-derivative($f,2,$v1,$v2)"' in mixed
    symbolic = _annotated(r"\frac{\partial^{n} u}{\partial t^{n}}")
    assert 'intent="partial-derivative($f,$n,$v1)"' in symbolic


def test_partial_operator_applied_to_parentheses():
    result = _annotated(r"\frac{\partial}{\partial \theta}(r \cos \theta)")
    assert 'intent="partial-derivative($f,1,$v1)"' in result
    assert _valid(result)


def test_jacobian_notation_is_left_alone():
    assert "partial-derivative" not in _annotated(
        r"\frac{\partial(f, g)}{\partial(x, y)}"
    )


def test_gradient_divergence_and_curl():
    assert 'intent="gradient($f)"' in _annotated(r"\nabla f = 0")
    assert 'intent="divergence($f)"' in _annotated(r"\nabla \cdot F")
    assert 'intent="curl($f)"' in _annotated(r"\nabla \times F")


def test_gradient_applied_to_a_point():
    result = _annotated(r"\nabla f(a) \cdot y")
    assert 'intent="gradient($f)"' in result
    assert 'intent="$name($var)"' in result
    assert _valid(result)


def test_norms_and_absolute_values():
    norm = _annotated(r"\|B\|(e^{x}-1)")
    assert 'intent="norm($a)"' in norm
    squared = _annotated(r"\frac{1}{\|\mathbf{a}\|^2}")
    assert 'intent="norm($a)"' in squared
    assert _valid(squared)
    absolute = _annotated(r"D = \frac{|ax_0 + c|}{\sqrt{a^2 + b^2}}")
    assert 'intent="absolute-value($a)"' in absolute
    assert _valid(absolute)
    # |A| with a table is a determinant, not an absolute value.
    determinant = _annotated(r"\begin{vmatrix} a & b \\ c & d \end{vmatrix}")
    assert "absolute-value" not in determinant


def test_multi_argument_application_but_not_single_variable():
    assert 'intent="$name($v1, $v2)"' in _annotated(r"g(x, y) + 1")
    # A single argument is decided by the general annotator as before.
    assert "$v1" not in _annotated(r"a(b + c)")


def test_enforce_partial_derivatives_replaces_llm_reading():
    root = parse_and_normalize(
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>'
        '<mfrac intent="fraction($num,$den)"><mrow arg="num"><mo>∂</mo><mi>u</mi></mrow>'
        '<mrow arg="den"><mo>∂</mo><mi>x</mi></mrow></mfrac>'
        "<mo>=</mo><mn>0</mn></mrow></math>"
    )
    assert enforce_partial_derivatives(root) == 1
    markup = serialize_mathml(root)
    assert 'intent="partial-derivative($f,1,$v1)"' in markup
    assert "fraction(" not in markup
    assert enforce_partial_derivatives(root) == 0
