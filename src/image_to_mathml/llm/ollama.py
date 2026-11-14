"""Ollama HTTP client with no third-party SDK dependency."""

from __future__ import annotations

import json
from collections.abc import Sequence

import requests

from ..config import Settings
from .base import LLMError
from .ollama_stream import collect_stream_content


class OllamaClient:
    """Small client for Ollama's ``/api/chat`` endpoint."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.model = self.settings.ollama_model
        self.base_url = self.settings.ollama_base_url.rstrip("/")

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def is_available(self) -> bool:
        try:
            response = requests.get(self._url("/api/tags"), timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        try:
            response = requests.get(self._url("/api/tags"), timeout=10)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise LLMError(f"Could not list Ollama models: {exc}") from exc
        return [entry.get("name", "") for entry in data.get("models", [])]

    def has_model(self) -> bool:
        try:
            models = self.list_models()
        except LLMError:
            return False
        return any(
            name == self.model or name.startswith(self.model + ":") for name in models
        )

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        num_predict: int | None = None,
    ) -> str:
        stream = self.settings.ollama_stream
        payload = {
            "model": self.model,
            "messages": list(messages),
            "stream": stream,
            "options": {
                "temperature": (
                    self.settings.temperature if temperature is None else temperature
                ),
                "num_predict": (
                    self.settings.num_predict if num_predict is None else num_predict
                ),
                "num_ctx": self.settings.num_ctx,
            },
        }
        try:
            response = requests.post(
                self._url("/api/chat"),
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=self.settings.ollama_timeout,
                stream=stream,
            )
        except requests.RequestException as exc:
            raise LLMError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc

        if response.status_code != 200:
            detail = response.text.strip()[:500]
            raise LLMError(f"Ollama returned HTTP {response.status_code}: {detail}")

        if stream:
            with response:
                content = collect_stream_content(
                    response, source=f"Ollama at {self.base_url}"
                )
        else:
            try:
                data = response.json()
            except ValueError as exc:
                raise LLMError("Ollama returned a non-JSON response") from exc

            message = data.get("message") or {}
            content = message.get("content") or data.get("response") or ""

        if not content.strip():
            raise LLMError("Ollama returned an empty response")
        return content
