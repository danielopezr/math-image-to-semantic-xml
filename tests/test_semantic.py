from image_to_mathml.mathml.normalizer import (
    parse_and_normalize,
    serialize_mathml,
)
from image_to_mathml.mathml.semantic import (
    apply_semantic_payload,
    assign_node_ids,
    parse_semantic_payload,
    strip_node_ids,
    strip_semantic_attributes,
)
from image_to_mathml.mathml.validator import validate_mathml
from image_to_mathml.models import EquationRequest, Flow
from image_to_mathml.pipeline import EquationPipeline

NS = 'xmlns="http://www.w3.org/1998/Math/MathML"'


def _with_ids(markup: str):
    root = parse_and_normalize(markup)
    strip_semantic_attributes(root)
    assign_node_ids(root)
    return root


def _final(root) -> str:
    strip_node_ids(root)
    return serialize_mathml(root)


def test_assign_and_strip_node_ids():
    root = parse_and_normalize(f"<math {NS}><mrow><mi>x</mi></mrow></math>")
    id_map = assign_node_ids(root)
    assert list(id_map) == ["n1", "n2", "n3"]
    assert id_map["n2"].tag.endswith("mrow")

    strip_node_ids(root)
    assert 'id="' not in serialize_mathml(root)


def test_apply_fraction_payload():
    root = _with_ids(
        f"<math {NS}><mrow><mfrac><mi>a</mi><mi>b</mi></mfrac></mrow></math>"
    )
    result = apply_semantic_payload(
        root,
        [
            {
                "target": "n3",
                "concept": "fraction",
                "args": {"a1": "n4", "a2": "n5"},
            }
        ],
    )
    assert result.ok
    assert result.applied == 1
    markup = _final(root)
    assert 'intent="fraction($num,$den)"' in markup
    assert 'arg="num"' in markup
    assert 'arg="den"' in markup
    assert validate_mathml(markup).acceptable


def test_apply_group_argument_for_generic_application():
    root = _with_ids(
        f"<math {NS}><mrow><msub><mi>f</mi><mi>i</mi></msub>"
        "<mo>(</mo><mi>x</mi><mo>+</mo><mn>1</mn><mo>)</mo>"
        "</mrow></math>"
    )
    result = apply_semantic_payload(
        root,
        [
            {
                "target": "n2",
                "concept": "generic-application",
                "args": {"a1": "n3"},
                "groups": {"a2": ["n7", "n8", "n9"]},
            }
        ],
    )
    assert result.ok
    markup = _final(root)
    assert 'intent="$name($var)"' in markup
    assert '<msub arg="name">' in markup
    assert '<mrow arg="var">' in markup
    assert validate_mathml(markup).acceptable


def test_apply_literal_argument_for_root_index():
    root = _with_ids(f"<math {NS}><mrow><msqrt><mi>x</mi></msqrt></mrow></math>")
    result = apply_semantic_payload(
        root,
        [
            {
                "target": "n3",
                "concept": "root",
                "args": {"a1": "n4"},
                "literals": {"a2": "2"},
            }
        ],
    )
    assert result.ok
    markup = _final(root)
    assert 'intent="root($radicand, 2)"' in markup
    assert 'arg="radicand"' in markup
    assert validate_mathml(markup).acceptable


def test_apply_wrap_target_for_generic_application_inside_expression():
    root = _with_ids(
        f"<math {NS}><mrow><mn>2</mn><mi>f</mi><mo>(</mo><mi>x</mi>"
        "<mo>)</mo><mo>+</mo><mn>1</mn></mrow></math>"
    )
    result = apply_semantic_payload(
        root,
        [
            {
                "wrap": ["n4", "n5", "n6", "n7"],
                "new_id": "w1",
                "concept": "generic-application",
                "args": {"a1": "n4", "a2": "n6"},
            }
        ],
    )
    assert result.ok
    markup = _final(root)
    assert 'intent="$name($var)"' in markup
    assert '<mrow intent="$name($var)">' in markup
    assert '<mi arg="name">f</mi>' in markup
    assert '<mi arg="var">x</mi>' in markup
    assert validate_mathml(markup).acceptable


def test_group_ids_are_sorted_in_document_order():
    root = _with_ids(
        f"<math {NS}><mrow><mi>f</mi><mo>(</mo><mi>x</mi><mo>+</mo>"
        "<mn>1</mn><mo>)</mo></mrow></math>"
    )
    result = apply_semantic_payload(
        root,
        [
            {
                "target": "n2",
                "concept": "generic-application",
                "args": {"a1": "n3"},
                "groups": {"a2": ["n7", "n5", "n6"]},
            }
        ],
    )
    assert result.ok
    markup = _final(root)
    assert '<mrow arg="var">' in markup
    assert validate_mathml(markup).acceptable


def test_duplicate_target_annotations_are_skipped_and_left_empty():
    root = _with_ids(
        f"<math {NS}><mrow><mfrac><mi>a</mi><mi>b</mi></mfrac></mrow></math>"
    )
    result = apply_semantic_payload(
        root,
        [
            {
                "target": "n3",
                "concept": "fraction",
                "args": {"a1": "n4", "a2": "n5"},
            },
            {
                "target": "n3",
                "concept": "power",
                "args": {"a1": "n4", "a2": "n5"},
            },
        ],
    )
    assert result.ok
    assert result.applied == 0
    assert len(result.warnings) == 2
    assert 'intent="' not in _final(root)


def test_nested_intent_node_can_also_be_an_argument():
    root = _with_ids(
        f"<math {NS}><mrow><mi>sin</mi><mo>(</mo><mi>f</mi><mo>(</mo>"
        "<mi>x</mi><mo>)</mo><mo>)</mo></mrow></math>"
    )
    result = apply_semantic_payload(
        root,
        [
            {
                "wrap": ["n5", "n6", "n7", "n8"],
                "new_id": "w1",
                "concept": "generic-application",
                "args": {"a1": "n5", "a2": "n7"},
            },
            {
                "target": "n2",
                "concept": "sine",
                "args": {"a1": "w1"},
            },
        ],
    )
    assert result.ok
    assert result.applied == 2
    markup = _final(root)
    assert 'intent="$name($var)"' in markup
    assert 'intent="sine($arg)"' in markup
    assert '<mrow intent="$name($var)" arg="arg">' in markup
    assert validate_mathml(markup).acceptable


def test_wrap_target_has_no_temporary_id_unless_requested():
    root = _with_ids(
        f"<math {NS}><mrow><mn>2</mn><mi>f</mi><mo>(</mo><mi>x</mi>"
        "<mo>)</mo></mrow></math>"
    )
    result = apply_semantic_payload(
        root,
        [
            {
                "wrap": ["n4", "n5", "n6", "n7"],
                "concept": "generic-application",
                "args": {"a1": "n4", "a2": "n6"},
            }
        ],
    )
    assert result.ok
    wrappers = [
        element
        for element in root.iter()
        if element.tag.endswith("mrow") and element.get("intent") is not None
    ]
    assert wrappers
    assert wrappers[0].get("id") is None


def test_invented_role_is_reported():
    root = _with_ids(
        f"<math {NS}><mrow><mfrac><mi>a</mi><mi>b</mi></mfrac></mrow></math>"
    )
    result = apply_semantic_payload(
        root,
        [
            {
                "target": "n3",
                "concept": "fraction",
                "args": {"b1": "n4", "a2": "n5"},
            }
        ],
    )
    assert not result.ok
    assert any("Unknown argument slot" in error for error in result.errors)


def test_unknown_id_is_reported():
    root = _with_ids(
        f"<math {NS}><mrow><mfrac><mi>a</mi><mi>b</mi></mfrac></mrow></math>"
    )
    result = apply_semantic_payload(
        root,
        [
            {
                "target": "missing",
                "concept": "fraction",
                "args": {"a1": "n4", "a2": "n5"},
            }
        ],
    )
    assert not result.ok
    assert any("Unknown target id" in error for error in result.errors)


def test_non_contiguous_group_is_reported():
    root = _with_ids(
        f"<math {NS}><mrow><mi>f</mi><mo>(</mo><mi>x</mi><mo>+</mo>"
        "<mn>1</mn><mo>)</mo></mrow></math>"
    )
    result = apply_semantic_payload(
        root,
        [
            {
                "target": "n2",
                "concept": "generic-application",
                "args": {"a1": "n3"},
                "groups": {"a2": ["n4", "n6"]},
            }
        ],
    )
    assert not result.ok
    assert any("contiguous" in error for error in result.errors)


def test_parse_semantic_payload_accepts_markdown_fences():
    annotations = parse_semantic_payload(
        '```json\n{"annotations": [{"target": "n2", "concept": "sine"}]}\n```'
    )
    assert annotations == [{"target": "n2", "concept": "sine"}]


class FakeSemanticLLM:
    model = "fake"

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0
        self.messages: list[list[dict[str, str]]] = []

    def is_available(self) -> bool:
        return True

    def has_model(self) -> bool:
        return True

    def chat(self, messages, **kwargs) -> str:
        self.messages.append(list(messages))
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


FRACTION_JSON = (
    '{"version": 1, "annotations": [{"target": "n3", "concept": "fraction", '
    '"args": {"a1": "n4", "a2": "n5"}}]}'
)


def test_pipeline_accepts_semantic_json_and_strips_ids():
    fake = FakeSemanticLLM([FRACTION_JSON])
    pipeline = EquationPipeline(llm=fake)
    result = pipeline.run(EquationRequest(latex=r"\frac{a}{b}", flow=Flow.HYBRID))

    assert result.validation.acceptable
    assert result.llm_used
    assert result.attempts == 1
    assert 'intent="fraction($num,$den)"' in result.mathml
    assert 'id="' not in result.mathml
    assert "JSON contract" in fake.messages[0][0]["content"]
    assert "temporary ids" in fake.messages[0][1]["content"]
    assert result.llm_outputs[0].input_mathml in fake.messages[0][1]["content"]
    assert 'id="n3"' in result.llm_outputs[0].input_mathml
    assert result.as_dict()["llm_outputs"][0]["input_mathml"] == (
        result.llm_outputs[0].input_mathml
    )


def test_pipeline_retries_after_invalid_json():
    fake = FakeSemanticLLM(["not json", FRACTION_JSON])
    pipeline = EquationPipeline(llm=fake)
    result = pipeline.run(
        EquationRequest(
            latex=r"\frac{a}{b}",
            flow=Flow.HYBRID,
            max_repair_attempts=1,
        )
    )

    assert result.validation.acceptable
    assert result.llm_used
    assert result.attempts == 2
    assert fake.calls == 2
    assert "Problems:" in fake.messages[1][1]["content"]


def test_pipeline_retries_after_semantic_error():
    bad = (
        '{"version": 1, "annotations": [{"target": "missing", '
        '"concept": "fraction", "args": {"a1": "n4", "a2": "n5"}}]}'
    )
    fake = FakeSemanticLLM([bad, FRACTION_JSON])
    pipeline = EquationPipeline(llm=fake)
    result = pipeline.run(
        EquationRequest(
            latex=r"\frac{a}{b}",
            flow=Flow.HYBRID,
            max_repair_attempts=1,
        )
    )

    assert result.validation.acceptable
    assert result.attempts == 2
    assert "Unknown target id" in fake.messages[1][1]["content"]


def test_pipeline_falls_back_when_semantic_json_is_never_valid():
    fake = FakeSemanticLLM(["not json"])
    pipeline = EquationPipeline(llm=fake)
    result = pipeline.run(
        EquationRequest(
            latex=r"\frac{a}{b}",
            flow=Flow.HYBRID,
            max_repair_attempts=1,
        )
    )

    assert result.validation.ok
    assert not result.llm_used
    assert result.attempts == 2
    assert 'intent="fraction($num,$den)"' in result.mathml
    assert any("semantic JSON" in warning for warning in result.warnings)
    assert 'id="' not in result.mathml
