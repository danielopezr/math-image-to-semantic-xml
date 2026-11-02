"""LaTeX clean-up helpers for OCR output and user input."""

from __future__ import annotations

import re

_DELIMITER_PATTERNS = (
    (re.compile(r"^\s*\$\$\s*(.*?)\s*\$\$\s*$", re.DOTALL), r"\1"),
    (re.compile(r"^\s*\$\s*(.*?)\s*\$\s*$", re.DOTALL), r"\1"),
    (re.compile(r"^\s*\\\[\s*(.*?)\s*\\\]\s*$", re.DOTALL), r"\1"),
    (re.compile(r"^\s*\\\(\s*(.*?)\s*\\\)\s*$", re.DOTALL), r"\1"),
    (
        re.compile(
            r"^\s*\\begin\{(?:equation|displaymath|math)\*?\}\s*(.*?)\s*\\end\{\1(?:\*)?\}\s*$",
            re.DOTALL,
        ),
        r"\1",
    ),
)

_NOISE_PATTERNS = (
    re.compile(r"\\label\{[^{}]*\}"),
    re.compile(r"\\tag\*?\{[^{}]*\}"),
    re.compile(r"\\nonumber\b"),
    re.compile(r"\\notag\b"),
)


def clean_latex(latex: str) -> str:
    """Remove math delimiters and equation-numbering noise added by OCR."""

    text = (latex or "").strip()
    text = text.replace("\u00a0", " ")
    for pattern, replacement in _DELIMITER_PATTERNS:
        new_text = pattern.sub(replacement, text)
        if new_text != text:
            text = new_text
            break
    for pattern in _NOISE_PATTERNS:
        text = pattern.sub("", text)
    return " ".join(text.split())
