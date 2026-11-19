"""Minimal LLM interface and HTTP helpers shared by the clients."""

from __future__ import annotations

import ctypes
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
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
# model call log (time per stage)
# ---------------------------------------------------------------------------
@dataclass
class ModelCall:
    """One request to a model server and where its time went.

    ``stage`` is ``"vision"`` for the multimodal model (the OCR in the hybrid
    flow) and ``"llm"`` for the text model. ``seconds`` is the wall time seen
    by the client, including a request cut by a timeout. The other times come
    from Ollama's answer and stay at zero when the server gave none.
    ``partial_output`` keeps what a streamed request had written when it was
    cut, and ``free_memory_mb`` the free RAM when the request ended.
    """

    stage: str
    model: str
    seconds: float = 0.0
    load_seconds: float = 0.0
    prompt_seconds: float = 0.0
    generate_seconds: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    completed: bool = False
    partial_output: str = ""
    free_memory_mb: float | None = None


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def free_memory_mb() -> float | None:
    """Free physical memory in MB, or None when it cannot be read."""

    try:
        if sys.platform == "win32":
            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return status.ullAvailPhys / 2**20
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError, AttributeError):
        return None
    return None


_MODEL_CALLS: ContextVar[list[ModelCall] | None] = ContextVar(
    "image_to_mathml_model_calls", default=None
)


@contextmanager
def record_model_calls() -> Iterator[list[ModelCall]]:
    """Collect every model request made inside the block."""

    calls: list[ModelCall] = []
    token = _MODEL_CALLS.set(calls)
    try:
        yield calls
    finally:
        _MODEL_CALLS.reset(token)


@contextmanager
def track_model_call(stage: str, model: str) -> Iterator[ModelCall]:
    """Time one request; it is logged even when it raises (e.g. a timeout)."""

    call = ModelCall(stage=stage, model=model)
    started = time.perf_counter()
    try:
        yield call
    finally:
        call.seconds = time.perf_counter() - started
        call.free_memory_mb = free_memory_mb()
        calls = _MODEL_CALLS.get()
        if calls is not None:
            calls.append(call)


def _ns_to_seconds(value: object) -> float:
    return value / 1e9 if isinstance(value, int | float) else 0.0


def apply_ollama_stats(call: ModelCall, data: object) -> None:
    """Copy the durations and token counts of an Ollama answer into ``call``."""

    if not isinstance(data, dict):
        return
    call.load_seconds = _ns_to_seconds(data.get("load_duration"))
    call.prompt_seconds = _ns_to_seconds(data.get("prompt_eval_duration"))
    call.generate_seconds = _ns_to_seconds(data.get("eval_duration"))
    call.prompt_tokens = int(data.get("prompt_eval_count") or 0)
    call.output_tokens = int(data.get("eval_count") or 0)
    call.completed = True


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
