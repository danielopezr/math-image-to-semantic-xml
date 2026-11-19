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


# Operators LaTeX already has a command for.  OCR often writes them as
# \mathrm{det} or \operatorname{sin}, which latex2mathml splits into letters
# (d·e·t) instead of one operator name.
_NAMED_OPERATORS = (
    "arccos|arcsin|arctan|arg|cosh|coth|cos|cot|csc|deg|det|dim|exp|gcd|hom|"
    "inf|ker|lg|liminf|limsup|lim|ln|log|max|min|Pr|sec|sinh|sin|sup|tanh|tan"
)
_OPERATOR_NAME = re.compile(
    r"\\(?:mathrm|operatorname|textrm|text)\s*\{\s*(" + _NAMED_OPERATORS + r")\s*\}"
)
# A Spanish word set apart by spacing commands ("x > 0 \, y \, s ≠ 0",
# "J(x) \quad y \quad J(-x)") is prose, not a variable: it becomes \text{}
# before the small spaces are dropped.
_SPACING = r"(?:\\[,;:]|\\q?quad\b)"
_SPANISH_WORD = re.compile(
    r"(" + _SPACING + r")\s*(y|o|si|para|donde|con|entonces|cuando)\s*"
    r"(?=" + _SPACING + r")"
)
# Thin, medium, thick and negative spaces: pure typography, but latex2mathml
# turns them into <mspace> elements that split products such as \sqrt{2}\,t.
# A preceding backslash means "\\" (a row break) followed by a comma.
_SMALL_SPACES = re.compile(r"(?<!\\)\\[,;:!]")


def clean_latex(latex: str) -> str:
    """Remove math delimiters and equation-numbering noise added by OCR.

    Also writes operator names as their LaTeX commands (\\mathrm{det} becomes
    \\det) and drops the small spacing commands (\\, \\; \\: \\!).
    """

    text = (latex or "").strip()
    text = text.replace("\u00a0", " ")
    for pattern, replacement in _DELIMITER_PATTERNS:
        new_text = pattern.sub(replacement, text)
        if new_text != text:
            text = new_text
            break
    for pattern in _NOISE_PATTERNS:
        text = pattern.sub("", text)
    text = _OPERATOR_NAME.sub(r"\\\1 ", text)
    text = _SPANISH_WORD.sub(r"\1\\text{ \2 }", text)
    text = _SMALL_SPACES.sub(" ", text)
    return " ".join(text.split())
