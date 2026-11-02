"""Select the local LLM backend."""

from __future__ import annotations

from ..config import Settings
from .base import LLMClient, LLMError
from .ollama import OllamaClient
from .openai_compat import OpenAICompatibleClient


def build_llm(settings: Settings | None = None) -> LLMClient:
    settings = settings or Settings.from_env()
    backend = settings.llm_backend.strip().lower()
    if backend == "ollama":
        return OllamaClient(settings)
    if backend in {"openai", "openai-compatible", "vllm", "vert", "lmstudio"}:
        return OpenAICompatibleClient(settings)
    raise LLMError(
        f"Unknown LLM backend {settings.llm_backend!r}. Use 'ollama' or 'openai'."
    )
