"""Minimal LLM interface used by the agents."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class LLMError(RuntimeError):
    """Raised when the local model cannot be reached or returns no text."""


class LLMClient(Protocol):
    model: str
    base_url: str

    def is_available(self) -> bool:
        """Return True when the local server can be reached."""

    def has_model(self) -> bool:
        """Return True when ``model`` is installed on the server."""

    def list_models(self) -> list[str]:
        """Return the model names exposed by the local server."""

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        num_predict: int | None = None,
    ) -> str:
        """Return the assistant text for a chat completion."""
