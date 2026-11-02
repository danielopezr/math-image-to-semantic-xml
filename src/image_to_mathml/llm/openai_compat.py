"""OpenAI-compatible local server client.

Most local runners (vLLM, LM Studio, Jan, llama.cpp, LocalAI and Ollama's
``/v1`` endpoint) expose this protocol.  It is a practical way to support a
"Vert"-style local server without coupling the module to a specific vendor.
"""

from __future__ import annotations

from collections.abc import Sequence

import requests

from ..config import Settings
from .base import LLMError


class OpenAICompatibleClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.model = self.settings.ollama_model
        self.base_url = self.settings.openai_base_url.rstrip("/")
        self.api_key = self.settings.openai_api_key

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def is_available(self) -> bool:
        try:
            response = requests.get(
                self._url("/models"), headers=self._headers(), timeout=5
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        try:
            response = requests.get(
                self._url("/models"), headers=self._headers(), timeout=10
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise LLMError(f"Could not list models at {self.base_url}: {exc}") from exc
        return [str(entry.get("id", "")) for entry in data.get("data", [])]

    def has_model(self) -> bool:
        try:
            models = self.list_models()
        except LLMError:
            return False
        return any(
            name == self.model
            or name.startswith(self.model + ":")
            or self.model in name
            for name in models
        )

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        num_predict: int | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": (
                self.settings.temperature if temperature is None else temperature
            ),
            "max_tokens": (
                self.settings.num_predict if num_predict is None else num_predict
            ),
            "stream": False,
        }
        try:
            response = requests.post(
                self._url("/chat/completions"),
                json=payload,
                headers=self._headers(),
                timeout=self.settings.ollama_timeout,
            )
        except requests.RequestException as exc:
            raise LLMError(f"Could not reach {self.base_url}: {exc}") from exc

        if response.status_code != 200:
            detail = response.text.strip()[:500]
            raise LLMError(
                f"OpenAI-compatible server returned HTTP {response.status_code}: {detail}"
            )

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                "Invalid chat completion response from local server"
            ) from exc
        if not content or not str(content).strip():
            raise LLMError("The local server returned an empty response")
        return str(content)
