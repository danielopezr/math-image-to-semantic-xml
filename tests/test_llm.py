import json

import pytest

from image_to_mathml.config import Settings
from image_to_mathml.llm.base import LLMError
from image_to_mathml.llm.factory import build_llm
from image_to_mathml.llm.ollama import OllamaClient
from image_to_mathml.llm.openai_compat import OpenAICompatibleClient


def test_factory_defaults_to_ollama():
    client = build_llm(Settings())
    assert isinstance(client, OllamaClient)


def test_factory_builds_openai_compatible_client_for_aliases():
    for backend in ("openai", "openai-compatible", "vllm", "vert", "lmstudio"):
        client = build_llm(Settings(llm_backend=backend))
        assert isinstance(client, OpenAICompatibleClient)
        assert client.base_url == "http://localhost:8000/v1"


def test_openai_settings_can_come_from_environment(monkeypatch):
    monkeypatch.setenv("IMAGE_TO_MATHML_LLM_BACKEND", "vllm")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    settings = Settings.from_env()
    client = build_llm(settings)
    assert isinstance(client, OpenAICompatibleClient)
    assert client.base_url == "http://localhost:9000/v1"
    assert client.api_key == "secret"


def test_ollama_stream_setting_comes_from_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_STREAM", "true")
    assert Settings.from_env().ollama_stream is True


def test_ollama_client_streams_and_joins_chunks(monkeypatch):
    client = OllamaClient(Settings(ollama_stream=True))
    captured = {}
    chunks = [
        b'{"message":{"content":"<math>"},"done":false}',
        b'{"message":{"content":"</math>"},"done":true}',
    ]

    class FakeResponse:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_lines(self, decode_unicode=False):
            return iter(chunks)

    def fake_post(url, data=None, headers=None, timeout=None, stream=False):
        captured["data"] = data
        captured["stream"] = stream
        return FakeResponse()

    monkeypatch.setattr("image_to_mathml.llm.ollama.requests.post", fake_post)

    assert client.chat([{"role": "user", "content": "x"}]) == "<math></math>"
    assert json.loads(captured["data"])["stream"] is True
    assert captured["stream"] is True


def test_ollama_client_rejects_invalid_streaming_chunks(monkeypatch):
    client = OllamaClient(Settings(ollama_stream=True))

    class FakeResponse:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_lines(self, decode_unicode=False):
            return iter([b"not-json"])

    monkeypatch.setattr(
        "image_to_mathml.llm.ollama.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(LLMError, match="invalid streaming JSON"):
        client.chat([{"role": "user", "content": "x"}])


def test_cli_stream_flags_override_the_default(monkeypatch):
    from image_to_mathml.cli import build_parser
    from image_to_mathml.cli_commands import _settings_from_args

    monkeypatch.delenv("OLLAMA_STREAM", raising=False)
    parser = build_parser()

    assert (
        _settings_from_args(
            parser.parse_args(["convert", "--latex", "x"])
        ).ollama_stream
        is False
    )
    assert (
        _settings_from_args(
            parser.parse_args(["convert", "--latex", "x", "--stream"])
        ).ollama_stream
        is True
    )
    assert (
        _settings_from_args(
            parser.parse_args(["convert", "--latex", "x", "--no-stream"])
        ).ollama_stream
        is False
    )


def test_model_calls_are_logged_with_ollama_stats(monkeypatch):
    from image_to_mathml.llm.base import record_model_calls

    client = OllamaClient(Settings(ollama_stream=True))
    chunks = [
        b'{"message":{"content":"<math>"},"done":false}',
        (
            b'{"message":{"content":"</math>"},"done":true,'
            b'"load_duration":3000000000,"prompt_eval_count":50,'
            b'"prompt_eval_duration":1000000000,"eval_count":30,'
            b'"eval_duration":6000000000}'
        ),
    ]

    class FakeResponse:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_lines(self, decode_unicode=False):
            return iter(chunks)

    monkeypatch.setattr(
        "image_to_mathml.llm.ollama.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )

    with record_model_calls() as calls:
        client.chat([{"role": "user", "content": "x"}])

    assert len(calls) == 1
    call = calls[0]
    assert (call.stage, call.model, call.completed) == ("llm", client.model, True)
    assert (call.load_seconds, call.prompt_seconds, call.generate_seconds) == (
        3.0,
        1.0,
        6.0,
    )
    assert (call.prompt_tokens, call.output_tokens) == (50, 30)


def test_a_failed_model_call_is_still_logged(monkeypatch):
    import requests

    from image_to_mathml.llm.base import record_model_calls
    from image_to_mathml.llm.ollama_vision import OllamaVisionClient

    def timeout(*args, **kwargs):
        raise requests.Timeout("Read timed out")

    monkeypatch.setattr("image_to_mathml.llm.ollama_vision.requests.post", timeout)
    client = OllamaVisionClient(model="vl")

    from PIL import Image

    with record_model_calls() as calls, pytest.raises(LLMError):
        client.chat([{"role": "user", "content": "x"}], Image.new("RGB", (4, 4)))

    assert [(call.stage, call.model, call.completed) for call in calls] == [
        ("vision", "vl", False)
    ]


class _StreamResponse:
    """A streamed answer that fails with ``error`` after its chunks."""

    status_code = 200

    def __init__(self, chunks, error=None):
        self.chunks = chunks
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_lines(self, decode_unicode=False):
        yield from self.chunks
        if self.error is not None:
            raise self.error


def test_a_stream_past_its_deadline_is_cut_and_keeps_the_partial_answer():
    from image_to_mathml.llm.base import ModelCall
    from image_to_mathml.llm.ollama_stream import collect_stream_content

    call = ModelCall(stage="llm", model="m")
    response = _StreamResponse(
        [
            b'{"message":{"content":"{\\"a\\": "},"done":false}',
            b'{"message":{"content":"[1, 1, 1"},"done":false}',
        ]
    )

    with pytest.raises(LLMError) as raised:
        collect_stream_content(
            response, source="Ollama", deadline=0.0, timeout=600, call=call
        )

    # The message is the one the evaluation reads to retry with a longer timeout.
    assert "timed out" in str(raised.value)
    assert "read timeout=600" in str(raised.value)
    assert call.partial_output == '{"a": '


def test_a_stream_that_finishes_in_time_is_not_cut():
    from image_to_mathml.llm.ollama_stream import collect_stream_content

    response = _StreamResponse(
        [
            b'{"message":{"content":"ok"},"done":false}',
            b'{"message":{"content":"!"},"done":true}',
        ]
    )

    assert collect_stream_content(response, source="Ollama", deadline=1e18) == "ok!"


def test_a_broken_stream_logs_the_partial_answer_of_the_call(monkeypatch):
    import requests

    from image_to_mathml.llm.base import record_model_calls

    client = OllamaClient(Settings(ollama_stream=True))
    response = _StreamResponse(
        [b'{"message":{"content":"times(times("},"done":false}'],
        error=requests.ConnectionError("Read timed out"),
    )
    monkeypatch.setattr(
        "image_to_mathml.llm.ollama.requests.post", lambda *a, **k: response
    )

    with record_model_calls() as calls, pytest.raises(LLMError):
        client.chat([{"role": "user", "content": "x"}])

    assert [(call.completed, call.partial_output) for call in calls] == [
        (False, "times(times(")
    ]


def test_every_model_call_records_the_free_memory(monkeypatch):
    from image_to_mathml.llm.base import record_model_calls, track_model_call

    monkeypatch.setattr("image_to_mathml.llm.base.free_memory_mb", lambda: 812.0)
    with record_model_calls() as calls, track_model_call("llm", "m"):
        pass

    assert calls[0].free_memory_mb == 812.0


def test_free_memory_is_a_positive_number_or_unknown():
    from image_to_mathml.llm.base import free_memory_mb

    value = free_memory_mb()
    assert value is None or value > 0
