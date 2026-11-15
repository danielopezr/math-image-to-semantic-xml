"""LaTeX clean-up helpers for OCR output and user input."""

from __future__ import annotations

import re

# Each pattern strips one outer math delimiter.  The body groups exclude "$"
# so that "$a$ y $b$" (two separate formulas) is left untouched instead of
# being turned into "a$ y $b".
_DELIMITER_PATTERNS = (
    (re.compile(r"^\s*\$\$\s*([^$]*?)\s*\$\$\s*$", re.DOTALL), r"\1"),
    (re.compile(r"^\s*\$(?!\$)\s*([^$]*?)\s*\$\s*$", re.DOTALL), r"\1"),
    (re.compile(r"^\s*\\\[\s*(.*?)\s*\\\]\s*$", re.DOTALL), r"\1"),
    (re.compile(r"^\s*\\\(\s*(.*?)\s*\\\)\s*$", re.DOTALL), r"\1"),
    (
        re.compile(
            r"^\s*\\begin\{(?P<env>equation|displaymath|math)(?P<star>\*?)\}"
            r"\s*(?P<body>.*?)\s*\\end\{(?P=env)(?P=star)\}\s*$",
            re.DOTALL,
        ),
        r"\g<body>",
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
