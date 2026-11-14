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
