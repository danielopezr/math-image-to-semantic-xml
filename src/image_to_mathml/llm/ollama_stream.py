"""Helpers for consuming Ollama's NDJSON streaming responses."""

from __future__ import annotations

import json
from collections.abc import Iterable

import requests

from .base import LLMError


def collect_stream_content(response, *, source: str) -> str:
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
                break
    except requests.RequestException as exc:
        raise LLMError(f"Streaming from {source} failed: {exc}") from exc
    return "".join(parts)
