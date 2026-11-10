from image_to_mathml.mathml.validator import validate_mathml
from image_to_mathml.models import EquationRequest, Flow, MathContext
from image_to_mathml.pipeline import EquationPipeline, compare_flows

NS = 'xmlns="http://www.w3.org/1998/Math/MathML"'


class FakeLLM:
    model = "fake"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def has_model(self) -> bool:
        return True

    def chat(self, messages, **kwargs) -> str:
        self.calls += 1
        return self.response


def test_offline_flow_produces_valid_mathml():
    pipeline = EquationPipeline()
    result = pipeline.run(
        EquationRequest(
            latex=r"f(x)=\frac{1}{1+e^{-x}}",
            flow=Flow.OFFLINE,
            context=MathContext(area="análisis", topic="funciones"),
        )
    )
    assert result.validation.ok
    assert result.validation.acceptable
    assert 'intent="$name($var)"' in result.mathml
    assert 'intent="fraction($num,$den)"' in result.mathml
    assert not result.llm_used


def test_default_policy_allows_gamma_but_rejects_other_open_concepts():
    pipeline = EquationPipeline()
    gamma = pipeline.run(EquationRequest(latex=r"\Gamma(x)", flow=Flow.OFFLINE))
    assert gamma.validation.ok
    assert any(
        issue.code == "intent.open-concept" for issue in gamma.validation.warnings
    )

    airy = (
        f'<math {NS}><mrow intent="airy-function($arg)">'
        '<mi arg="arg">x</mi></mrow></math>'
    )
    report = validate_mathml(
        airy,
        allowed_open_concepts=set(pipeline.settings.allowed_open_concepts),
    )
    assert not report.ok
    assert any(
        issue.code == "intent.open-concept-not-allowed" for issue in report.errors
    )


def test_hybrid_flow_uses_fake_llm_annotation():
    response = (
        f'<math {NS}><mrow><mfrac intent="fraction($num,$den)">'
        '<mi arg="num">a</mi><mi arg="den">b</mi></mfrac></mrow></math>'
    )
    fake = FakeLLM(response)
    pipeline = EquationPipeline(llm=fake)
    result = pipeline.run(EquationRequest(latex=r"\frac{a}{b}", flow=Flow.HYBRID))
    assert result.validation.acceptable
    assert result.llm_used
    assert result.attempts == 1
    assert "fraction($num,$den)" in result.mathml
    assert fake.calls == 1


def test_hybrid_flow_removes_orphan_args():
    response = (
        f'<math {NS}><mrow arg="unused"><mfrac intent="fraction($num,$den)">'
        '<mi arg="num">a</mi><mi arg="den">b</mi></mfrac></mrow></math>'
    )
    fake = FakeLLM(response)
    pipeline = EquationPipeline(llm=fake)
    result = pipeline.run(EquationRequest(latex=r"\frac{a}{b}", flow=Flow.HYBRID))
    assert result.validation.acceptable
    assert 'arg="unused"' not in result.mathml
    assert not any("arg.unused" in issue.code for issue in result.validation.issues)


def test_hybrid_flow_survives_invalid_llm_intents():
    response = (
        f'<math {NS}><mrow><mfrac intent="fraction($num)">'
        '<mi arg="num">a</mi><mi>b</mi></mfrac></mrow></math>'
    )
    fake = FakeLLM(response)
    pipeline = EquationPipeline(llm=fake)
    result = pipeline.run(EquationRequest(latex=r"\frac{a}{b}", flow=Flow.HYBRID))
    assert result.validation.acceptable
    assert result.llm_used
    assert "fraction($num,$den)" in result.mathml


def test_direct_flow_falls_back_when_llm_returns_invalid_markup():
    fake = FakeLLM("this is not MathML")
    pipeline = EquationPipeline(llm=fake)
    result = pipeline.run(
        EquationRequest(
            latex=r"\frac{a}{b}",
            flow=Flow.DIRECT,
            max_repair_attempts=1,
        )
    )
    assert result.validation.ok
    assert not result.llm_used
    assert any("deterministic fallback" in warning for warning in result.warnings)


def test_direct_flow_accepts_valid_generic_function():
    response = (
        f'<math {NS}><mrow intent="$name($var)">'
        '<mi arg="name">f</mi><mo>(</mo><mi arg="var">x</mi><mo>)</mo>'
        "</mrow></math>"
    )
    fake = FakeLLM(response)
    pipeline = EquationPipeline(llm=fake)
    result = pipeline.run(EquationRequest(latex="f(x)", flow=Flow.DIRECT))
    assert result.validation.acceptable
    assert result.llm_used
    assert 'intent="$name($var)"' in result.mathml


def test_image_input_uses_configured_ocr_engine():
    class FakeOCR:
        name = "fake-ocr"

        def image_to_latex(self, image_path: str) -> str:
            assert image_path == "equation.png"
            return r"\frac{a}{b}"

    pipeline = EquationPipeline(llm=FakeLLM("unused"), ocr=FakeOCR())
    result = pipeline.run(EquationRequest(image_path="equation.png", flow=Flow.OFFLINE))
    assert result.ocr_backend == "fake-ocr"
    assert result.validation.acceptable
    assert "fraction($num,$den)" in result.mathml


def test_auto_flow_skips_direct_when_hybrid_is_acceptable():
    response = (
        f'<math {NS}><mrow><mfrac intent="fraction($num,$den)">'
        '<mi arg="num">a</mi><mi arg="den">b</mi></mfrac></mrow></math>'
    )
    fake = FakeLLM(response)
    pipeline = EquationPipeline(llm=fake)
    result = pipeline.run(EquationRequest(latex=r"\frac{a}{b}", flow=Flow.AUTO))
    assert result.flow is Flow.HYBRID
    assert result.validation.acceptable
    assert fake.calls == 1
    assert any("skipped the slower direct flow" in item for item in result.trace)


def test_compare_flows_returns_three_results():
    fake = FakeLLM("<math><mrow><mi>x</mi></mrow></math>")
    pipeline = EquationPipeline(llm=fake)
    results = compare_flows(
        EquationRequest(latex="x", flow=Flow.OFFLINE),
        pipeline,
    )
    assert set(results) == {"hybrid", "direct", "offline"}
    assert results["offline"].validation.ok


def test_hybrid_structure_source_is_latex2mathml():
    response = (
        f'<math {NS}><mrow><mfrac intent="fraction($num,$den)">'
        '<mi arg="num">a</mi><mi arg="den">b</mi></mfrac></mrow></math>'
    )
    pipeline = EquationPipeline(llm=FakeLLM(response))
    result = pipeline.run(EquationRequest(latex=r"\frac{a}{b}", flow=Flow.HYBRID))

    assert result.structure_source == "latex2mathml"
    assert result.as_dict()["structure_source"] == "latex2mathml"


def test_direct_structure_source_is_llm_when_accepted():
    response = (
        f'<math {NS}><mrow intent="$name($var)">'
        '<mi arg="name">f</mi><mo>(</mo><mi arg="var">x</mi><mo>)</mo>'
        "</mrow></math>"
    )
    pipeline = EquationPipeline(llm=FakeLLM(response))
    result = pipeline.run(EquationRequest(latex="f(x)", flow=Flow.DIRECT))

    assert result.structure_source == "llm"
    assert result.as_dict()["structure_source"] == "llm"


def test_direct_fallback_structure_source_is_latex2mathml():
    pipeline = EquationPipeline(llm=FakeLLM("not MathML"))
    result = pipeline.run(
        EquationRequest(latex=r"\frac{a}{b}", flow=Flow.DIRECT, max_repair_attempts=0)
    )

    assert result.structure_source == "latex2mathml"
