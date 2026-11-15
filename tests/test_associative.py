"""N-ary associative operators, flattening in the metric and implicit times."""

import re

import pytest

from image_to_mathml.config import Settings
from image_to_mathml.mathml.annotator import annotate_tree
from image_to_mathml.mathml.concept_cards import build_concept_cards
from image_to_mathml.mathml.concepts import ConceptRegistry
from image_to_mathml.mathml.converter import LatexToMathML
from image_to_mathml.mathml.normalizer import parse_and_normalize, serialize_mathml
from image_to_mathml.mathml.validator import validate_mathml
from image_to_mathml.metrics.bleu import exact_match, intent_f1, mathml_bleu
from image_to_mathml.models import EquationRequest, Flow
from image_to_mathml.pipeline import EquationPipeline

NS = 'xmlns="http://www.w3.org/1998/Math/MathML"'


def _product(intent: str, *factors: tuple[str, str]) -> str:
    inner = "".join(f'<mi arg="{arg}">{text}</mi>' for arg, text in factors)
    return f'<math {NS}><mrow intent="{intent}">{inner}</mrow></math>'


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def test_associative_operators_accept_two_or_more_arguments():
    registry = ConceptRegistry(nary_associative=True)
    for name in ("times", "plus", "and", "or", "union", "intersection"):
        assert registry.accepts_arity(name, 2), name
        assert registry.accepts_arity(name, 4), name
        assert not registry.accepts_arity(name, 1), name
    for name in ("minus", "divided-by", "power"):
        assert not registry.accepts_arity(name, 3), name


def test_nary_can_be_switched_off(monkeypatch):
    strict = ConceptRegistry(nary_associative=False)
    assert not strict.accepts_arity("times", 3)
    monkeypatch.setenv("IMAGE_TO_MATHML_NARY_ASSOCIATIVE", "false")
    assert not ConceptRegistry().accepts_arity("times", 3)
    assert not Settings.from_env().nary_associative


def test_validator_follows_the_registry_option():
    markup = _product("times($a,$b,$c)", ("a", "x"), ("b", "y"), ("c", "z"))
    assert validate_mathml(markup, registry=ConceptRegistry(nary_associative=True)).ok
    report = validate_mathml(markup, registry=ConceptRegistry(nary_associative=False))
    assert any(issue.code == "intent.arity" for issue in report.errors)


def test_cards_show_the_nary_arity():
    def label(registry):
        card = next(c for c in build_concept_cards(registry) if c.name == "times")
        return [(s.min_arity, s.max_arity) for s in card.signatures]

    assert label(ConceptRegistry(nary_associative=True)) == [(2, None)]
    assert label(ConceptRegistry(nary_associative=False)) == [(2, 2)]


# ---------------------------------------------------------------------------
# metric
# ---------------------------------------------------------------------------
FLAT = _product("times($a,$b,$c)", ("a", "x"), ("b", "y"), ("c", "z"))
NESTED_ACROSS = (
    f'<math {NS}><mrow intent="times($a,$b)"><mi arg="a">x</mi>'
    '<mrow arg="b" intent="times($c,$d)"><mi arg="c">y</mi><mi arg="d">z</mi></mrow>'
    "</mrow></math>"
)
NESTED_INSIDE = _product("times($a,times($b,$c))", ("a", "x"), ("b", "y"), ("c", "z"))


@pytest.mark.parametrize("nested", [NESTED_ACROSS, NESTED_INSIDE])
def test_metric_treats_nested_and_flat_products_as_equal(nested):
    assert intent_f1(FLAT, nested).f1 == 1.0
    assert intent_f1(nested, FLAT).f1 == 1.0
    assert mathml_bleu(FLAT, nested).intent.score == pytest.approx(
        mathml_bleu(FLAT, FLAT).intent.score
    )


def test_flattening_does_not_merge_different_operators_or_minus():
    plus_inside_times = (
        f'<math {NS}><mrow intent="times($a,$b)"><mi arg="a">x</mi>'
        '<mrow arg="b" intent="plus($c,$d)"><mi arg="c">y</mi><mi arg="d">z</mi></mrow>'
        "</mrow></math>"
    )
    assert intent_f1(FLAT, plus_inside_times).f1 < 1.0
    nested_minus = _product(
        "minus($a,minus($b,$c))", ("a", "x"), ("b", "y"), ("c", "z")
    )
    flat_minus = _product("minus($a,$b,$c)", ("a", "x"), ("b", "y"), ("c", "z"))
    assert not exact_match(flat_minus, nested_minus, include_intent=True)


# ---------------------------------------------------------------------------
# implicit times rule
# ---------------------------------------------------------------------------
def _times(latex: str, *, nary: bool = True) -> list[str]:
    root = parse_and_normalize(LatexToMathML().convert(latex))
    annotate_tree(root, implicit_times=True, nary_associative=nary)
    markup = serialize_mathml(root)
    assert validate_mathml(markup, registry=ConceptRegistry(nary_associative=nary)).ok
    return re.findall(r'intent="(times[^"]*)"', markup)


@pytest.mark.parametrize(
    ("latex", "expected"),
    [
        (r"2x", ["times($arg1,$arg2)"]),
        (r"2x^{2}y", ["times($arg1,$arg2,$arg3)"]),
        (r"i(1+r)^{t}(1+r)", ["times($arg1,$arg2,$arg3)"]),
        (r"a(b+c)", ["times($arg1,$arg2)"]),
        (r"2f(x)", ["times($arg1,$arg2)"]),
        (r"(2x+1)(x-3)", ["times($arg1,$arg2)", "times($arg1,$arg2)"]),
    ],
)
def test_implicit_products_are_annotated_flat(latex, expected):
    assert sorted(_times(latex)) == sorted(expected)


@pytest.mark.parametrize("latex", [r"x + y", r"\sin^{2} x", r"a/b"])
def test_no_implicit_product_where_there_is_none(latex):
    assert _times(latex) == []


def test_the_differential_is_not_a_factor():
    root = parse_and_normalize(LatexToMathML().convert(r"\int_0^1 2x\,dx"))
    annotate_tree(root, implicit_times=True)
    markup = serialize_mathml(root)
    assert markup.count("times(") == 1
    assert '<mi arg="arg3">' not in markup


def test_without_nary_products_are_nested_pairs():
    assert _times(r"2x^{2}y", nary=False) == [
        "times($arg1,$arg2)",
        "times($arg1,$arg2)",
    ]


def test_implicit_times_is_off_by_default_and_configurable():
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
            raise AssertionError

    request = EquationRequest(latex="2x", flow=Flow.OFFLINE)
    default = EquationPipeline(Settings(), llm=NoLLM()).run(request)
    assert "times" not in default.mathml
    enabled = EquationPipeline(Settings(annotate_implicit_times=True), llm=NoLLM()).run(
        request
    )
    assert 'intent="times($arg1,$arg2)"' in enabled.mathml
