import json
from importlib.resources import files

from image_to_mathml.agents.prompts import semantic_messages
from image_to_mathml.config import Settings
from image_to_mathml.mathml.concept_cards import (
    get_concept_card,
    get_concept_cards,
    render_concept_cards,
)
from image_to_mathml.mathml.concept_rag import ConceptRAG
from image_to_mathml.mathml.concepts import get_registry
from image_to_mathml.mathml.converter import LatexToMathML
from image_to_mathml.mathml.normalizer import parse_and_normalize
from image_to_mathml.mathml.open_concept_rag import (
    OpenConceptRAG,
    get_open_concept_cards,
)
from image_to_mathml.models import EquationRequest, Flow, MathContext
from image_to_mathml.pipeline import EquationPipeline


def _tree(latex: str):
    return parse_and_normalize(LatexToMathML().convert(latex))


def _names(latex: str, **kwargs) -> list[str]:
    return [
        card.name
        for card in ConceptRAG().retrieve(latex, _tree(latex), top_k=8, **kwargs)
    ]


def test_fraction_card_contains_retrieval_fields_only():
    card = get_concept_card("fraction")

    assert card is not None
    assert card.kind == "core"
    assert card.roles == ("num", "den")
    assert "\\frac" in card.latex
    assert "mfrac" in card.mathml_tags
    assert "/" in card.symbols
    assert card.targets == ("mfrac", "mrow")
    assert card.examples

    rendered = render_concept_cards([card])
    assert "definition" not in rendered
    assert "speech" not in rendered
    assert "fraction" in rendered
    assert "targets mfrac, mrow" in rendered
    assert "\\frac{a}{b}" in rendered


def test_every_core_concept_has_a_card_with_targets():
    registry = get_registry()
    cards = get_concept_cards()

    for name in registry.core:
        card = get_concept_card(name, cards)
        assert card is not None, name
        assert card.targets, name


def test_operator_and_fenced_concepts_offer_target_alternatives():
    inner = get_concept_card("inner-product")
    ordered = get_concept_card("ordered-pair")

    assert inner is not None
    assert ordered is not None
    assert inner.targets == ("mo", "mrow")
    assert ordered.targets == ("mrow", "mo")
    assert "\\cdot" in inner.latex
    assert "\u27e8" in ordered.symbols


def test_retriever_selects_fraction_for_a_simple_fraction():
    assert _names(r"\frac{a}{b}") == ["fraction"]


def test_retriever_selects_structural_anchors_for_trigonometry():
    names = set(_names(r"\sin^2(x)+\cos^2(x)=1"))

    assert {"sine", "cosine", "power"} <= names
    assert "definite-integral" not in names
    assert "matrix" not in names


def test_retriever_detects_generic_function_application():
    names = _names("f(x)")

    assert names == ["generic-application"]


def test_retriever_does_not_confuse_log_base_with_generic_application():
    names = _names(r"\log_{2}(x)")

    assert "logarithm" in names
    assert "generic-application" not in names


def test_retriever_applies_the_open_concept_policy():
    assert "gamma-function" in _names(r"\Gamma(x)", allow_open=True)
    assert "gamma-function" not in _names(r"\Gamma(x)", allow_open=False)
    assert "gamma-function" not in _names(
        r"\Gamma(x)",
        allow_open=True,
        allowed_open_concepts=set(),
    )
    assert "gamma-function" in _names(
        r"\Gamma(x)",
        allow_open=True,
        allowed_open_concepts={"gamma-function"},
    )


def test_semantic_prompt_uses_only_retrieved_cards():
    card = get_concept_card("fraction")
    assert card is not None
    messages = semantic_messages(
        r"\frac{a}{b}",
        "<math><mrow><mfrac><mi>a</mi><mi>b</mi></mfrac></mrow></math>",
        MathContext(),
        concept_cards=[card],
    )

    system = messages[0]["content"]
    assert "candidate concepts" in system
    assert "fraction; arity 2; roles num, den" in system
    assert "Core names are:" not in system


def test_semantic_prompt_handles_an_empty_candidate_set():
    messages = semantic_messages(
        "x",
        "<math><mrow><mi>x</mi></mrow></math>",
        MathContext(),
        concept_cards=[],
    )

    assert "No candidate concept applies" in messages[0]["content"]


class FakeSemanticLLM:
    model = "fake"

    def __init__(self, response: str) -> None:
        self.response = response
        self.messages = []

    def is_available(self) -> bool:
        return True

    def has_model(self) -> bool:
        return True

    def chat(self, messages, **kwargs) -> str:
        self.messages.append(list(messages))
        return self.response


def test_pipeline_sends_retrieved_cards_to_the_semantic_prompt():
    response = (
        '{"version": 1, "annotations": ['
        '{"target": "n3", "concept": "fraction", "args": {"a1": "n4", "a2": "n5"}}'
        "]}"
    )
    fake = FakeSemanticLLM(response)
    pipeline = EquationPipeline(llm=fake)
    result = pipeline.run(EquationRequest(latex=r"\frac{a}{b}", flow=Flow.HYBRID))

    assert result.validation.acceptable
    system = fake.messages[0][0]["content"]
    assert "candidate concepts" in system
    assert "fraction; arity 2; roles num, den" in system
    assert "Core names are:" not in system


def test_pipeline_can_disable_concept_rag():
    response = (
        '{"version": 1, "annotations": ['
        '{"target": "n3", "concept": "fraction", "args": {"a1": "n4", "a2": "n5"}}'
        "]}"
    )
    fake = FakeSemanticLLM(response)
    pipeline = EquationPipeline(
        settings=Settings(concept_rag=False),
        llm=fake,
    )
    result = pipeline.run(EquationRequest(latex=r"\frac{a}{b}", flow=Flow.HYBRID))

    assert result.validation.acceptable
    system = fake.messages[0][0]["content"]
    assert "Core names are:" in system


def test_all_cards_come_from_the_registry():
    registry_names = {card.name for card in get_concept_cards()}

    assert "fraction" in registry_names
    assert "generic-application" in registry_names


def test_open_rag_covers_every_open_only_concept():
    registry = get_registry()
    open_only = set(registry.open) - set(registry.core)
    card_names = {card.name for card in get_open_concept_cards()}

    assert card_names == open_only
    assert "fourier-transform" in card_names
    assert "gamma-function" in card_names


def test_open_hints_only_reference_open_concepts():
    payload = json.loads(
        files("image_to_mathml")
        .joinpath("data", "open_concept_hints.json")
        .read_text(encoding="utf-8")
    )
    registry = get_registry()
    open_only = set(registry.open) - set(registry.core)
    referenced = set(payload["cues"])

    for section in ("areas", "topics"):
        for names in payload[section].values():
            referenced.update(names)

    assert referenced <= open_only


def test_open_rag_uses_area_and_topic_to_select_candidates():
    rag = OpenConceptRAG()
    context = MathContext(area="Análisis", topic="transformada de Fourier")

    names = [
        card.name
        for card in rag.retrieve("f", _tree("f"), context, allow_open=True, top_k=8)
    ]

    assert names[0] == "fourier-transform"
    assert "mobius-function" not in names


def test_open_rag_applies_the_open_concept_policy():
    rag = OpenConceptRAG()
    latex = r"\Gamma(x)"
    root = _tree(latex)

    assert rag.retrieve(latex, root, allow_open=False) == []
    assert "gamma-function" in {
        card.name for card in rag.retrieve(latex, root, allow_open=True)
    }
    assert (
        rag.retrieve(
            latex,
            root,
            allow_open=True,
            allowed_open_concepts={"beta-function"},
        )
        == []
    )
    assert "gamma-function" in {
        card.name
        for card in rag.retrieve(
            latex,
            root,
            allow_open=True,
            allowed_open_concepts={"gamma-function"},
        )
    }


def test_pipeline_adds_open_rag_cards_to_the_semantic_prompt():
    response = '{"version": 1, "annotations": []}'
    fake = FakeSemanticLLM(response)
    pipeline = EquationPipeline(
        settings=Settings(allow_all_open_concepts=True),
        llm=fake,
    )
    pipeline.run(
        EquationRequest(
            latex="f",
            flow=Flow.HYBRID,
            context=MathContext(area="Análisis", topic="transformada de Fourier"),
        )
    )

    system = fake.messages[0][0]["content"]
    assert "fourier-transform" in system
