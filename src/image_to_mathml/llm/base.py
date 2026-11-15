"""Minimal LLM interface and HTTP helpers shared by the clients."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol

import requests


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
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Return the assistant text for a chat completion.

        ``json_schema`` asks the server to constrain the answer to that JSON
        Schema when it supports structured output.
        """


# ---------------------------------------------------------------------------
# per-request timeout
# ---------------------------------------------------------------------------
_REQUEST_TIMEOUT: ContextVar[float | None] = ContextVar(
    "image_to_mathml_request_timeout", default=None
)


@contextmanager
def request_timeout(timeout: float | None) -> Iterator[None]:
    """Override the HTTP timeout of every LLM/vision call inside the block.

    A :class:`~contextvars.ContextVar` is used instead of mutating shared
    settings, so concurrent pipeline runs in other threads are not affected.
    """

    if timeout is None:
        yield
        return
    token = _REQUEST_TIMEOUT.set(float(timeout))
    try:
        yield
    finally:
        _REQUEST_TIMEOUT.reset(token)


def effective_timeout(default: float) -> float:
    override = _REQUEST_TIMEOUT.get()
    return default if override is None else override


def current_request_timeout() -> float | None:
    return _REQUEST_TIMEOUT.get()


# ---------------------------------------------------------------------------
# retries
# ---------------------------------------------------------------------------
_RETRYABLE_STATUS = {502, 503, 504}


def post_with_retries(
    send: Callable[[], requests.Response],
    *,
    retries: int,
    backoff: float = 1.0,
) -> requests.Response:
    """Call ``send`` and retry connection errors and 502/503/504 answers.

    Read timeouts are not retried: repeating a request that already waited
    the whole timeout doubles the wait, and the evaluation has its own,
    explicit timeout policy.
    """

    attempt = 0
    while True:
        try:
            response = send()
        except requests.Timeout:
            raise
        except requests.ConnectionError:
            if attempt >= retries:
                raise
        else:
            if response.status_code not in _RETRYABLE_STATUS or attempt >= retries:
                return response
        attempt += 1
        time.sleep(backoff * attempt)


# ---------------------------------------------------------------------------
# model names
# ---------------------------------------------------------------------------
def model_matches(installed: str, wanted: str) -> bool:
    """Return True when the installed model name is the requested model.

    ``qwen2.5`` matches ``qwen2.5:latest`` and ``qwen2.5:3b``; ``qwen3-vl:2b``
    only matches itself (a substring test would accept ``qwen3-vl:2b-q8``,
    which the server then refuses).
    """

    if not installed or not wanted:
        return False
    if installed == wanted:
        return True
    if ":" in wanted:
        return wanted.endswith(":latest") and installed == wanted.rsplit(":", 1)[0]
    return installed.startswith(wanted + ":")
