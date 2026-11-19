"""Helpers for consuming Ollama's NDJSON streaming responses."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable

import requests

from .base import LLMError, ModelCall


def collect_stream_content(
    response,
    *,
    source: str,
    final: dict | None = None,
    deadline: float | None = None,
    timeout: float | None = None,
    call: ModelCall | None = None,
) -> str:
    """Join the streamed content; ``final`` receives the last (``done``) chunk,
    which carries Ollama's durations and token counts.

    ``deadline`` (a :func:`time.monotonic` value) bounds the whole answer:
    the HTTP read timeout only bounds the wait between two chunks, so a model
    that keeps writing slowly would otherwise never time out.  ``timeout`` is
    only quoted in the error, which the evaluation reads to retry.  When the
    stream fails, what had arrived is kept in ``call.partial_output``.
    """

    parts: list[str] = []
    try:
        lines: Iterable[bytes] = response.iter_lines(decode_unicode=False)
        for raw_line in lines:
            if not raw_line:
                continue
            try:
                data = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise LLMError(
                    f"{source} returned invalid streaming JSON: {exc}"
                ) from exc
            if not isinstance(data, dict):
                raise LLMError(f"{source} returned an invalid streaming chunk")
            error = data.get("error")
            if error:
                raise LLMError(f"{source} error: {error}")
            message = data.get("message") or {}
            content = message.get("content") or data.get("response") or ""
            if content:
                parts.append(str(content))
            if data.get("done"):
                if final is not None:
                    final.update(data)
                break
            if deadline is not None and time.monotonic() > deadline:
                limit = f" (read timeout={timeout:g})" if timeout is not None else ""
                raise LLMError(f"{source} timed out while streaming{limit}")
    except requests.RequestException as exc:
        if call is not None:
            call.partial_output = "".join(parts)
        raise LLMError(f"Streaming from {source} failed: {exc}") from exc
    except LLMError:
        if call is not None:
            call.partial_output = "".join(parts)
        raise
    return "".join(parts)
