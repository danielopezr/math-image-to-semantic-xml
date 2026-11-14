"""Ollama multimodal client used for direct image-to-MathML generation."""

from __future__ import annotations

import base64
import io
import json
import os
from collections.abc import Sequence

import requests

from .base import LLMError
from .ollama_stream import collect_stream_content

DEFAULT_MODEL = "qwen3-vl:4b-instruct"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 600.0
DEFAULT_NUM_PREDICT = 1024
DEFAULT_NUM_CTX = 4096
DEFAULT_MAX_SIDE = 1600


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _encode_image(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class OllamaVisionClient:
    """Small client for Ollama vision requests through ``/api/chat``."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        num_predict: int | None = None,
        num_ctx: int | None = None,
        max_side: int | None = None,
        stream: bool | None = None,
    ) -> None:
        self.model = (
            model or os.getenv("IMAGE_TO_MATHML_OLLAMA_VISION_MODEL") or DEFAULT_MODEL
        )
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.timeout = (
            timeout
            if timeout is not None
            else _env_float("IMAGE_TO_MATHML_OLLAMA_VISION_TIMEOUT", DEFAULT_TIMEOUT)
        )
        self.num_predict = (
            num_predict
            if num_predict is not None
            else _env_int(
                "IMAGE_TO_MATHML_OLLAMA_VISION_NUM_PREDICT", DEFAULT_NUM_PREDICT
            )
        )
        self.num_ctx = (
            num_ctx
            if num_ctx is not None
            else _env_int("IMAGE_TO_MATHML_OLLAMA_VISION_NUM_CTX", DEFAULT_NUM_CTX)
        )
        self.max_side = (
            max_side
            if max_side is not None
            else _env_int("IMAGE_TO_MATHML_OLLAMA_VISION_MAX_SIDE", DEFAULT_MAX_SIDE)
        )
        self.stream = (
            stream if stream is not None else _env_bool("OLLAMA_STREAM", False)
        )

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _prepare_image(self, image):
        if self.max_side <= 0 or max(image.size) <= self.max_side:
            return image
        from PIL import Image

        resized = image.copy()
        resized.thumbnail((self.max_side, self.max_side), Image.LANCZOS)
        return resized

    def _payload(
        self,
        messages: Sequence[dict[str, object]],
        image,
        temperature: float,
        num_predict: int,
    ) -> dict[str, object]:
        payload_messages: list[dict[str, object]] = [dict(item) for item in messages]
        for message in reversed(payload_messages):
            if message.get("role") == "user":
                message["images"] = [_encode_image(self._prepare_image(image))]
                break
        else:
            raise LLMError("Ollama vision requires one user message with an image")
        return {
            "model": self.model,
            "messages": payload_messages,
            "stream": self.stream,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "num_ctx": self.num_ctx,
            },
        }

    def chat(
        self,
        messages: Sequence[dict[str, object]],
        image,
        *,
        temperature: float = 0.0,
        num_predict: int | None = None,
    ) -> str:
        payload = self._payload(
            messages,
            image,
            temperature,
            self.num_predict if num_predict is None else num_predict,
        )
        try:
            response = requests.post(
                self._url("/api/chat"),
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
                stream=self.stream,
            )
        except requests.RequestException as exc:
            raise LLMError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc

        if response.status_code != 200:
            detail = response.text.strip()[:500]
            raise LLMError(
                f"Ollama vision returned HTTP {response.status_code}: {detail}"
            )

        if self.stream:
            with response:
                content = collect_stream_content(
                    response, source=f"Ollama vision model {self.model!r}"
                )
        else:
            try:
                data = response.json()
                content = (data.get("message") or {}).get("content") or ""
            except (ValueError, AttributeError) as exc:
                raise LLMError(
                    "Ollama vision returned an invalid JSON response"
                ) from exc

        if not str(content).strip():
            raise LLMError(
                f"Ollama vision model {self.model!r} returned an empty response"
            )
        return str(content)

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
        return [str(entry.get("name", "")) for entry in data.get("models", [])]

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
