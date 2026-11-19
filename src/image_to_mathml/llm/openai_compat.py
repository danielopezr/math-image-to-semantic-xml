"""OpenAI-compatible local server client.

Most local runners (vLLM, LM Studio, Jan, llama.cpp, LocalAI and Ollama's
``/v1`` endpoint) expose this protocol.  It is a practical way to support a
"Vert"-style local server without coupling the module to a specific vendor.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import requests

from ..config import Settings
from .base import (
    LLMError,
    ModelCall,
    effective_timeout,
    model_matches,
    post_with_retries,
    track_model_call,
)


def _apply_usage(call: ModelCall, response: requests.Response) -> None:
    """Token counts of an OpenAI-style answer (no per-stage durations)."""

    if response.status_code != 200:
        return
    try:
        usage = response.json().get("usage") or {}
    except (ValueError, AttributeError):
        return
    call.prompt_tokens = int(usage.get("prompt_tokens") or 0)
    call.output_tokens = int(usage.get("completion_tokens") or 0)


class OpenAICompatibleClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.model = self.settings.ollama_model
        self.base_url = self.settings.openai_base_url.rstrip("/")
        self.api_key = self.settings.openai_api_key
        # Set when the server rejected ``response_format``; later calls skip it.
        self._structured_output_supported = True

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
            model_matches(name, self.model) or name.endswith("/" + self.model)
            for name in models
        )

    def _post(self, payload: dict[str, Any]) -> requests.Response:
        timeout = effective_timeout(self.settings.ollama_timeout)
        return post_with_retries(
            lambda: requests.post(
                self._url("/chat/completions"),
                json=payload,
                headers=self._headers(),
                timeout=timeout,
            ),
            retries=self.settings.llm_retries,
        )

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        num_predict: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
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
        structured = (
            json_schema is not None
            and self.settings.structured_output
            and self._structured_output_supported
        )
        if structured:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "semantic_annotations", "schema": json_schema},
            }
        with track_model_call("llm", self.model) as call:
            try:
                response = self._post(payload)
                if structured and response.status_code in {400, 422}:
                    # Older servers do not know response_format: retry without it.
                    self._structured_output_supported = False
                    payload.pop("response_format", None)
                    response = self._post(payload)
            except requests.RequestException as exc:
                raise LLMError(f"Could not reach {self.base_url}: {exc}") from exc
            _apply_usage(call, response)

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
        if isinstance(content, list):
            # Some servers return content parts: [{"type": "text", "text": ...}].
            content = "".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            )
        if not content or not str(content).strip():
            raise LLMError("The local server returned an empty response")
        return str(content)
