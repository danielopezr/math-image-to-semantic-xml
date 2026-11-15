"""Regression tests for the pipeline, OCR and LLM client robustness fixes."""

import json

import pytest
import requests

from image_to_mathml.config import Settings
from image_to_mathml.llm.base import LLMError, model_matches
from image_to_mathml.llm.ollama import OllamaClient
from image_to_mathml.llm.openai_compat import OpenAICompatibleClient
from image_to_mathml.models import EquationRequest, Flow
from image_to_mathml.pipeline import EquationPipeline


class _UnavailableLLM:
    model = "fake"
    base_url = "http://fake"

    def is_available(self):
        return False

    def has_model(self):
        return False

    def list_models(self):
        return []

    def chat(self, messages, **kwargs):
        raise AssertionError("not called")


class _CountingOCR:
    name = "counting"

    def __init__(self):
        self.calls = 0

    def image_to_latex(self, image_path):
        self.calls += 1
        return "x^{2}"


def _image(tmp_path):
    path = tmp_path / "eq.png"
    path.write_bytes(b"not used by the fake OCR")
    return str(path)


def test_ocr_engine_is_built_once_per_backend(tmp_path, monkeypatch):
    built = []

    def fake_build(backend, **kwargs):
        built.append((backend, kwargs))
        return _CountingOCR()

    monkeypatch.setattr("image_to_mathml.pipeline.build_ocr", fake_build)
    pipeline = EquationPipeline(Settings(), llm=_UnavailableLLM())
    image = _image(tmp_path)
    other = tmp_path / "other.png"
    other.write_bytes(b"x")

    for path in (image, str(other)):
        pipeline.run(
            EquationRequest(image_path=path, flow=Flow.OFFLINE, ocr_backend="p2t")
        )

    assert len(built) == 1
    options = built[0][1]["ollama_options"]
    assert options["model"] == Settings().ollama_vision_model
    assert options["timeout"] == Settings().ollama_vision_timeout


def test_ocr_result_is_reused_for_the_same_image(tmp_path):
    ocr = _CountingOCR()
    pipeline = EquationPipeline(Settings(), llm=_UnavailableLLM(), ocr=ocr)
    image = _image(tmp_path)

    first = pipeline.run(EquationRequest(image_path=image, flow=Flow.OFFLINE))
    second = pipeline.run(EquationRequest(image_path=image, flow=Flow.OFFLINE))

    assert ocr.calls == 1
    assert first.mathml == second.mathml
    assert any("cached" in timing.note for timing in second.timings)


def test_direct_vision_falls_back_when_the_vision_server_is_down(tmp_path):
    ocr = _CountingOCR()
    pipeline = EquationPipeline(Settings(), llm=_UnavailableLLM(), ocr=ocr)
    pipeline.vision_agent.is_available = lambda: False

    result = pipeline.run(
        EquationRequest(
            image_path=_image(tmp_path), flow=Flow.DIRECT, direct_vision=True
        )
    )

    assert result.flow is Flow.DIRECT
    assert not result.llm_used
    assert result.validation.ok
    assert any("not reachable" in warning for warning in result.warnings)


def test_vision_flow_without_fallback_still_raises(tmp_path):
    pipeline = EquationPipeline(Settings(), llm=_UnavailableLLM())
    pipeline.vision_agent.is_available = lambda: False

    with pytest.raises(RuntimeError, match="not reachable"):
        pipeline.run(EquationRequest(image_path=_image(tmp_path), flow=Flow.VISION))


class _JsonErrorLLM:
    model = "fake"
    base_url = "http://fake"

    def __init__(self):
        self.kwargs = []

    def is_available(self):
        return True

    def has_model(self):
        return True

    def list_models(self):
        return ["fake"]

    def chat(self, messages, **kwargs):
        self.kwargs.append(kwargs)
        return '{"annotations": [{"target": "n999", "concept": "power"}]}'


def test_hybrid_retrieves_concepts_once_and_sends_a_schema(monkeypatch):
    llm = _JsonErrorLLM()
    pipeline = EquationPipeline(Settings(max_repair_attempts=2), llm=llm)
    calls = []
    original = pipeline.concept_rag.retrieve

    def counting_retrieve(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline.concept_rag, "retrieve", counting_retrieve)

    result = pipeline.run(EquationRequest(latex="x^{2}", flow=Flow.HYBRID))

    assert len(calls) == 1
    assert len(llm.kwargs) == 3
    schema = llm.kwargs[0]["json_schema"]
    for variant in schema["properties"]["annotations"]["items"]["anyOf"]:
        assert "power" in variant["properties"]["concept"]["enum"]
    assert len(result.warnings) == len(set(result.warnings))


def test_structured_output_can_be_disabled():
    llm = _JsonErrorLLM()
    pipeline = EquationPipeline(
        Settings(structured_output=False, max_repair_attempts=0), llm=llm
    )
    pipeline.run(EquationRequest(latex="x^{2}", flow=Flow.HYBRID))
    assert llm.kwargs[0]["json_schema"] is None


# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------
class _Response:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {"message": {"content": "ok"}}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


def test_ollama_client_sends_format_and_think(monkeypatch):
    sent = []

    def fake_post(url, data=None, headers=None, timeout=None, stream=False):
        sent.append(json.loads(data))
        return _Response()

    monkeypatch.setattr("image_to_mathml.llm.ollama.requests.post", fake_post)
    client = OllamaClient(Settings(ollama_think=False))
    client.chat([{"role": "user", "content": "x"}], json_schema={"type": "object"})
    client.chat([{"role": "user", "content": "x"}])

    assert sent[0]["format"] == {"type": "object"}
    assert sent[0]["think"] is False
    assert "format" not in sent[1]


def test_ollama_client_retries_connection_errors_but_not_timeouts(monkeypatch):
    monkeypatch.setattr("image_to_mathml.llm.base.time.sleep", lambda seconds: None)
    attempts = []

    def flaky_post(url, data=None, headers=None, timeout=None, stream=False):
        attempts.append(1)
        if len(attempts) == 1:
            raise requests.ConnectionError("refused")
        return _Response()

    monkeypatch.setattr("image_to_mathml.llm.ollama.requests.post", flaky_post)
    assert OllamaClient(Settings(llm_retries=1)).chat(
        [{"role": "user", "content": "x"}]
    )
    assert len(attempts) == 2

    timeouts = []

    def slow_post(url, data=None, headers=None, timeout=None, stream=False):
        timeouts.append(1)
        raise requests.ReadTimeout("Read timed out")

    monkeypatch.setattr("image_to_mathml.llm.ollama.requests.post", slow_post)
    with pytest.raises(LLMError, match="timed out"):
        OllamaClient(Settings(llm_retries=3)).chat([{"role": "user", "content": "x"}])
    assert len(timeouts) == 1


def test_openai_client_drops_response_format_when_unsupported(monkeypatch):
    sent = []

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.append(dict(json))
        if "response_format" in json:
            return _Response(400, {"error": "unknown field response_format"})
        return _Response(200, {"choices": [{"message": {"content": "{}"}}]})

    monkeypatch.setattr("image_to_mathml.llm.openai_compat.requests.post", fake_post)
    client = OpenAICompatibleClient(Settings(llm_backend="openai"))

    assert client.chat([{"role": "user", "content": "x"}], json_schema={}) == "{}"
    client.chat([{"role": "user", "content": "x"}], json_schema={})

    assert ["response_format" in payload for payload in sent] == [True, False, False]


@pytest.mark.parametrize(
    ("installed", "wanted", "expected"),
    [
        ("qwen2.5:3b", "qwen2.5:3b", True),
        ("qwen2.5:latest", "qwen2.5", True),
        ("qwen2.5:3b", "qwen2.5", True),
        ("qwen3-vl:2b-instruct-q8_0", "qwen3-vl:2b", False),
        ("my-qwen2.5:3b", "qwen2.5:3b", False),
        ("qwen2.5", "qwen2.5:latest", True),
    ],
)
def test_model_matches_is_strict(installed, wanted, expected):
    assert model_matches(installed, wanted) is expected


class _ScriptedLLM(_JsonErrorLLM):
    def __init__(self, answer):
        super().__init__()
        self.answer = answer

    def chat(self, messages, **kwargs):
        self.kwargs.append(kwargs)
        return self.answer


def test_deterministic_annotator_fills_what_the_llm_left_unannotated():
    # The LLM only annotates the fraction of f(x)=1/(1+e^{-x}).
    llm = _ScriptedLLM(
        '{"annotations": [{"target": "n8", "concept": "fraction",'
        ' "args": {"a1": "n9", "a2": "n10"}}]}'
    )
    pipeline = EquationPipeline(Settings(max_repair_attempts=0), llm=llm)

    result = pipeline.run(
        EquationRequest(latex=r"f(x)=\frac{1}{1+e^{-x}}", flow=Flow.HYBRID)
    )

    assert result.llm_used
    assert result.validation.acceptable
    # The LLM's fraction only repeats the default reading and is dropped.
    assert "fraction(" not in result.mathml
    assert 'intent="$name($var)"' in result.mathml
    assert 'intent="exponential($exp)"' in result.mathml
    assert any("deterministic fill (+2" in step for step in result.trace)


def test_fill_is_skipped_when_it_would_break_the_llm_annotation():
    # The LLM points $arg1 at the bare "f"; wrapping f(x) in a generic
    # application would hide that reference, so the fill must be discarded.
    llm = _ScriptedLLM(
        '{"annotations": [{"target": "n2", "concept": "equals",'
        ' "args": {"a1": "n3", "a2": "n8"}}]}'
    )
    pipeline = EquationPipeline(Settings(max_repair_attempts=0), llm=llm)

    result = pipeline.run(EquationRequest(latex="f(x)=y", flow=Flow.HYBRID))

    assert result.llm_used
    assert result.validation.ok
    assert 'intent="equals($arg1,$arg2)"' in result.mathml
    assert "$name($var)" not in result.mathml
    assert not any("deterministic fill" in step for step in result.trace)
