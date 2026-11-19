"""Operator grouping, parenthesised script bases and partial LLM answers."""

import json

import pytest

from image_to_mathml.config import Settings
from image_to_mathml.mathml.annotator import annotate_tree
from image_to_mathml.mathml.converter import LatexToMathML
from image_to_mathml.mathml.dom import element_children, element_text, localname
from image_to_mathml.mathml.grouping import group_operator_scopes
from image_to_mathml.mathml.normalizer import parse_and_normalize, serialize_mathml
from image_to_mathml.mathml.semantic import (
    apply_semantic_payload,
    assign_node_ids,
    strip_node_ids,
)
from image_to_mathml.mathml.validator import validate_mathml
from image_to_mathml.models import EquationRequest, Flow
from image_to_mathml.pipeline import EquationPipeline

NS = 'xmlns="http://www.w3.org/1998/Math/MathML"'


def _tree(latex: str):
    return parse_and_normalize(LatexToMathML().convert(latex))


def _top(root):
    return element_children(element_children(root)[0])


def _texts(element) -> list[str]:
    return [element_text(child) for child in element_children(element)]


# ---------------------------------------------------------------------------
# parenthesised bases
# ---------------------------------------------------------------------------
def test_script_on_a_closing_parenthesis_takes_the_whole_group_as_base():
    root = _tree(r"(1+r)^{t}")
    msup = next(el for el in root.iter() if localname(el.tag) == "msup")
    base, exponent = element_children(msup)
    assert localname(base.tag) == "mrow"
    assert _texts(base) == ["(", "1", "+", "r", ")"]
    assert element_text(exponent) == "t"
    assert validate_mathml(serialize_mathml(root)).ok


def test_parenthesised_power_is_annotated_by_the_fallback():
    root = _tree(r"c = \frac{i(1+r)^{t}(1+r)-i(1+r)}{(1+r)-1}")
    annotate_tree(root)
    markup = serialize_mathml(root)
    assert 'intent="power($base,$exp)"' in markup
    assert validate_mathml(markup).ok


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("latex", "groups", "first_group"),
    [
        (r"|x-1| < 3", 1, ["|", "x", "−", "1", "|"]),
        (r"\lfloor x \rfloor \le x", 1, ["⌊", "x", "⌋"]),
        (r"\langle u, v \rangle = 0", 1, ["⟨", "u", ",", "v", "⟩"]),
        (r"\| v \| = 1", 1, ["‖", "v", "‖"]),
        (r"n! = n (n-1)!", 2, ["n", "!"]),
        (r"\sum_{i=1}^{n} i^{2}+1", 1, ["∑i=1n", "i2"]),
        # \, is dropped when the LaTeX is cleaned.
        (r"\int_{0}^{1} x^{2}\,dx = \frac{1}{3}", 1, ["∫01", "x2", "d", "x"]),
        (r"\lim_{x \to 0} \frac{\sin x}{x} = 1", 1, ["limx→0", "sinxx"]),
        (r"\frac{d}{dx} x^{3} = 3x^{2}", 1, ["ddx", "x3"]),
    ],
)
def test_operators_are_grouped_with_their_operands(latex, groups, first_group):
    root = _tree(latex)
    assert group_operator_scopes(root) == groups
    first = next(child for child in _top(root) if localname(child.tag) == "mrow")
    assert _texts(first) == first_group
    assert validate_mathml(serialize_mathml(root)).ok


def test_each_absolute_value_gets_its_own_group():
    root = _tree(r"|a|+|b|")
    assert group_operator_scopes(root) == 2
    assert [localname(child.tag) for child in _top(root)] == ["mrow", "mo", "mrow"]


@pytest.mark.parametrize(
    "latex", [r"\frac{df}{dx} = 2x", r"\{x \mid x > 0\}", r"a+b=c", r"|x|"]
)
def test_grouping_leaves_complete_or_unpaired_notation_alone(latex):
    root = _tree(latex)
    before = serialize_mathml(root)
    assert group_operator_scopes(root) == 0
    assert serialize_mathml(root) == before


def test_nested_sums_group_the_inner_operator_with_its_body():
    root = _tree(r"\sum_{i}\sum_{j} a_{ij}")
    assert group_operator_scopes(root) == 1
    top = _top(root)
    assert localname(top[0].tag) == "msub"
    assert localname(top[1].tag) == "mrow"


# ---------------------------------------------------------------------------
# transactional application and target kinds
# ---------------------------------------------------------------------------
def _with_ids(markup: str):
    root = parse_and_normalize(markup)
    assign_node_ids(root)
    return root


def test_a_failing_annotation_is_rolled_back_completely():
    root = _with_ids(
        f"<math {NS}><mrow><mi>a</mi><mo>+</mo><mi>b</mi><mo>=</mo><mi>c</mi>"
        "</mrow></math>"
    )
    before = serialize_mathml(root)
    # The wrap succeeds, then the unknown argument id fails the annotation.
    result = apply_semantic_payload(
        root,
        [{"wrap": ["n3", "n4", "n5"], "concept": "plus", "args": ["n3", "n99"]}],
    )
    assert result.applied == 0
    assert result.errors
    assert serialize_mathml(root) == before


def test_valid_annotations_survive_an_invalid_sibling():
    root = _with_ids(
        f"<math {NS}><mrow><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo>"
        "<msup><mi>y</mi><mn>3</mn></msup></mrow></math>"
    )
    result = apply_semantic_payload(
        root,
        [
            {"target": "n3", "concept": "power", "args": {"a1": "n4", "a2": "n5"}},
            {"target": "n7", "concept": "power", "args": {"a1": "n8", "a2": "n4"}},
        ],
    )
    assert result.applied == 1
    assert len(result.errors) == 1
    strip_node_ids(root)
    markup = serialize_mathml(root)
    assert markup.count('intent="power($base,$exp)"') == 1
    assert validate_mathml(markup).ok


def test_concepts_must_target_their_presentation_element():
    root = _with_ids(
        f"<math {NS}><mrow><mfrac><mi>d</mi><mrow><mi>d</mi><mi>x</mi></mrow>"
        "</mfrac></mrow></math>"
    )
    result = apply_semantic_payload(
        root,
        [{"target": "n3", "concept": "power", "args": {"a1": "n4", "a2": "n5"}}],
    )
    assert result.applied == 0
    assert "must target <msup>, not <mfrac>" in result.errors[0]


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------
class _TreeAwareLLM:
    """Answers by looking at the id tree it receives, like a real model."""

    model = "fake"
    base_url = "http://fake"

    def __init__(self, answer):
        self.answer = answer
        self.trees: list[str] = []

    def is_available(self):
        return True

    def has_model(self):
        return True

    def list_models(self):
        return ["fake"]

    def chat(self, messages, **kwargs):
        tree = messages[-1]["content"].split("Clean MathML tree with temporary ids:\n")[
            1
        ]
        self.trees.append(tree)
        return json.dumps({"annotations": self.answer(tree)})


def test_sum_can_target_the_grouped_operator_and_body():
    def answer(tree):
        root = parse_and_normalize(tree)
        for element in root.iter():
            children = element_children(element)
            if (
                localname(element.tag) == "mrow"
                and len(children) == 2
                and localname(children[0].tag) == "msubsup"
            ):
                return [
                    {
                        "target": element.get("id"),
                        "concept": "sum",
                        "args": {"a1": children[1].get("id")},
                    }
                ]
        return []

    llm = _TreeAwareLLM(answer)
    pipeline = EquationPipeline(Settings(max_repair_attempts=0), llm=llm)

    result = pipeline.run(
        EquationRequest(latex=r"\sum_{i=1}^{n} i^{2}+1", flow=Flow.HYBRID)
    )

    assert result.llm_used
    assert any("operator-grouping" in step for step in result.trace)
    assert any("accepted" in step for step in result.trace)
    # The guide does not annotate sums with limits: the intent is removed.
    assert "sum(" not in result.mathml
    assert any("reads well on its own" in item for item in result.warnings)
    assert result.validation.ok


def test_partial_answers_are_kept_when_no_attempt_is_fully_valid():
    def answer_with_ids(tree):
        root = parse_and_normalize(tree)
        msup = next(el for el in root.iter() if localname(el.tag) == "msup")
        base, exponent = element_children(msup)
        return [
            {
                "target": msup.get("id"),
                "concept": "power",
                "args": {"a1": base.get("id"), "a2": exponent.get("id")},
            },
            # Not a concept the rules decide, so it is not filtered out first.
            {"target": "n2", "concept": "logarithm", "args": {"a1": "n999"}},
        ]

    llm = _TreeAwareLLM(answer_with_ids)
    pipeline = EquationPipeline(Settings(max_repair_attempts=1), llm=llm)

    result = pipeline.run(EquationRequest(latex=r"x^{2}+y", flow=Flow.HYBRID))

    assert result.llm_used
    assert len(llm.trees) == 2  # the partial answer did not stop the repair
    assert 'intent="power($base,$exp)"' in result.mathml
    assert any("invalid LLM annotation(s) were discarded" in w for w in result.warnings)
    assert any("accepted the valid annotations" in step for step in result.trace)


def test_power_cannot_drop_a_subscript_or_use_a_function_name_as_base():
    root = _with_ids(
        f"<math {NS}><mrow><msubsup><mi>v</mi><mn>1</mn><mn>2</mn></msubsup>"
        "<mo>+</mo><msup><mi>sin</mi><mn>2</mn></msup><mi>x</mi></mrow></math>"
    )
    result = apply_semantic_payload(
        root,
        [
            {"target": "n3", "concept": "power", "args": {"a1": "n4", "a2": "n6"}},
            {"target": "n8", "concept": "power", "args": {"a1": "n9", "a2": "n10"}},
        ],
    )
    assert result.applied == 0
    assert "must target <msup>, not <msubsup>" in result.errors[0]
    assert "not a power of the symbol 'sin'" in result.errors[1]


def test_cards_never_suggest_a_token_as_target_for_concepts_with_arguments():
    from image_to_mathml.mathml.concept_cards import get_concept_card

    for name in ("cross-product", "inner-product", "factorial", "plus"):
        card = get_concept_card(name)
        assert card is not None, name
        assert not set(card.targets) & {"mo", "mi", "mn"}, (name, card.targets)
    # Concepts without arguments (constants) can still sit on a token.
    pi = get_concept_card("pi")
    assert pi is None or "mi" in pi.targets or "mo" in pi.targets


def test_prompt_follows_the_w3c_placement_of_large_operators():
    from image_to_mathml.agents.prompts import semantic_messages
    from image_to_mathml.models import MathContext

    system = semantic_messages("x", "<math/>", MathContext())[0]["content"]
    assert "can never reference its operands" in system
    assert "definite-integral: a1=lower limit" in system
    assert "never an <msubsup>" in system


def test_functions_are_grouped_with_an_implicit_product_argument():
    root = _tree(r"\frac{\cos kx}{2} + \sin 2x = \sin x")
    assert group_operator_scopes(root) == 3
    markup = serialize_mathml(root).replace(" ", "").replace("\n", "")
    assert "<mrow><mi>k</mi><mi>x</mi></mrow>" in markup
    assert "<mrow><mi>sin</mi><mrow><mn>2</mn><mi>x</mi></mrow></mrow>" in markup
