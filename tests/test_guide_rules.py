"""Rules agreed in docs/annotation-guide.md, checked end to end (offline)."""

import re

import pytest

from image_to_mathml.config import Settings
from image_to_mathml.mathml.annotator import annotate_tree, strip_default_intents
from image_to_mathml.mathml.normalizer import parse_and_normalize, serialize_mathml
from image_to_mathml.models import EquationRequest, Flow, MathContext
from image_to_mathml.pipeline import EquationPipeline

NS = 'xmlns="http://www.w3.org/1998/Math/MathML"'


class NoLLM:
    model = "none"
    base_url = ""

    def is_available(self):
        return False

    def has_model(self):
        return False

    def list_models(self):
        return []

    def chat(self, *args, **kwargs):
        raise AssertionError("not called")


def _intents(latex, *, area="", topic="", settings=None):
    pipeline = EquationPipeline(settings or Settings(), llm=NoLLM())
    result = pipeline.run(
        EquationRequest(
            latex=latex,
            flow=Flow.OFFLINE,
            context=MathContext(area=area, topic=topic),
        )
    )
    assert result.validation.ok, result.validation.error_messages()
    return re.findall(r'intent="([^"]+)"', result.mathml)


# ---------------------------------------------------------------------------
# 4. default readings are not annotated
# ---------------------------------------------------------------------------
def test_plain_fractions_subscripts_and_roots_keep_their_default_reading():
    assert _intents(r"\frac{a}{b} + x_{1} + \sqrt{2} + \sqrt[3]{y}") == []


def test_linear_fractions_and_non_default_scripts_are_still_annotated():
    assert _intents("a/b") == ["fraction($num,$den)"]
    assert _intents("x^{2}") == ["power($base,$exp)"]
    assert _intents("e^{-x}") == ["exponential($exp)"]


def test_default_structures_can_be_switched_back_on():
    legacy = Settings(annotate_default_structures=True, annotate_implicit_times=False)
    assert _intents(r"\frac{a}{b}", settings=legacy) == ["fraction($num,$den)"]
    assert _intents("x_{1}", settings=legacy) == ["indexed-by($base,$sub)"]


def test_strip_default_intents_keeps_meaningful_ones():
    root = parse_and_normalize(
        f'<math {NS}><mrow><mfrac intent="fraction($n,$d)"><mi arg="n">a</mi>'
        '<mi arg="d">b</mi></mfrac><mo>+</mo>'
        '<mrow intent="fraction($n,$d)"><mi arg="n">a</mi><mo>/</mo>'
        '<mi arg="d">b</mi></mrow><mo>+</mo>'
        '<mfrac intent="binomial-coefficient($n,$k)"><mi arg="n">n</mi>'
        '<mi arg="k">k</mi></mfrac></mrow></math>'
    )
    assert strip_default_intents(root) == 1
    markup = serialize_mathml(root)
    assert markup.count("fraction(") == 1  # the linear a/b
    assert "binomial-coefficient" in markup


# ---------------------------------------------------------------------------
# 6. derivatives written with primes
# ---------------------------------------------------------------------------
CALCULUS = "Cálculo y ecuaciones diferenciales"


def test_primes_are_derivatives_in_a_calculus_context():
    intents = _intents(r"y''' - 2y'' + 4y' - 8y = 0", area=CALCULUS)
    assert intents.count("derivative($y,_x,3)") == 1
    assert intents.count("derivative($y,_x,2)") == 1
    assert intents.count("derivative($y,_x,1)") == 1
    # 2y'', 4y' and 8y are products whose second factor is the derivative.
    assert intents.count("times($arg1,$arg2)") == 3


def test_parenthesised_order_is_a_derivative():
    assert "derivative($y,_x,$n)" in _intents(r"y^{(4)} + y = 0", area=CALCULUS)


def test_primes_are_left_alone_outside_calculus():
    assert _intents(r"y' = 3", area="Álgebra") == []


def test_derivative_primes_can_be_forced_or_disabled():
    assert _intents("y'", settings=Settings(derivative_primes="on")) == [
        "derivative($y,_x,1)"
    ]
    assert (
        _intents("y'", area=CALCULUS, settings=Settings(derivative_primes="off")) == []
    )
    assert _intents(
        "y'", settings=Settings(derivative_primes="on", derivative_variable="t")
    ) == ["derivative($y,_t,1)"]


# ---------------------------------------------------------------------------
# 5. inner product decided by the topic
# ---------------------------------------------------------------------------
def test_round_pairs_are_inner_products_when_the_topic_says_so():
    intents = _intents(
        r"(x_{2}, y_{0}) = \int_{-1}^{1} t^{2}\,dt",
        area="Álgebra lineal",
        topic="ortogonalización, bases ortogonales",
    )
    assert "inner-product($arg1,$arg2)" in intents


def test_round_pairs_are_not_guessed_without_that_topic():
    assert (
        _intents(r"(x_{2}, y_{0}) = 0", area="Álgebra lineal", topic="matrices") == []
    )


def test_angle_brackets_are_always_inner_products():
    assert _intents(r"\langle u, v \rangle = 0") == ["inner-product($arg1,$arg2)"]


def test_an_argument_list_is_neither_a_pair_nor_a_product():
    assert _intents(r"T(x, y)", area="Álgebra lineal", topic="producto interno") == []


# ---------------------------------------------------------------------------
# functions applied to one variable vs products
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("latex", [r"T(x)", r"Q(x)", r"y_{0}(t)"])
def test_a_symbol_applied_to_one_variable_is_a_function(latex):
    assert _intents(latex) == ["$name($var)"]


@pytest.mark.parametrize("latex", [r"a(b+c)", r"n(n+1)"])
def test_a_symbol_before_an_expression_is_a_product(latex):
    assert _intents(latex) == ["times($arg1,$arg2)"]


def test_operator_names_are_not_factors():
    root = parse_and_normalize(f"<math {NS}><mrow><mi>dim</mi><mi>V</mi></mrow></math>")
    annotate_tree(root, implicit_times=True)
    assert "times" not in serialize_mathml(root)


# ---------------------------------------------------------------------------
# metric: a prime next to its base equals a prime superscript
# ---------------------------------------------------------------------------
def test_metric_treats_sibling_and_superscript_primes_as_equal():
    from image_to_mathml.metrics.bleu import exact_match, intent_f1, mathml_bleu

    label = (
        f'<math {NS}><mrow><mrow intent="derivative($y, _x, 3)">'
        '<mi arg="y">y</mi><mo>&#x2034;</mo></mrow><mo>-</mo><mi>y</mi>'
        "<mo>&#x2032;</mo></mrow></math>"
    )
    generated = (
        f'<math {NS}><mrow><msup intent="derivative($y,_x,3)">'
        '<mi arg="y">y</mi><mi>&#x2034;</mi></msup><mo>-</mo>'
        "<msup><mi>y</mi><mi>&#x2032;</mi></msup></mrow></math>"
    )

    assert exact_match(label, generated, include_intent=False)
    assert exact_match(label, generated, include_intent=True)
    assert intent_f1(label, generated).f1 == 1.0
    assert mathml_bleu(label, generated).structure.score == 1.0
