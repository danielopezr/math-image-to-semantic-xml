from image_to_mathml.config import Settings
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
