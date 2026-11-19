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
# Spanish names of the same functions: \sen, \mathrm{sen}, \operatorname{tg}.
_SPANISH_OPERATORS = {"sen": "sin", "senh": "sinh", "tg": "tan"}
_SPANISH_OPERATOR_NAME = re.compile(
    r"\\(?:(?:mathrm|operatorname|textrm|text)\s*\{\s*(senh|sen|tg)\s*\}"
    r"|(senh|sen|tg)(?![A-Za-z]))"
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

# Alignment environments: latex2mathml does not know aligned, alignedat or
# eqnarray and leaves every "&" as a raw <mi>&</mi> (invalid XML), and it
# splits align*/split rows into one cell per column.  A system of equations is
# written as one cell per row (as in the dataset labels), so the alignment
# points are dropped and the environment becomes a one-column array.
_ALIGN_ENVS = (
    "aligned|alignedat|align|align\\*|alignat|alignat\\*|eqnarray|eqnarray\\*|"
    "split|flalign|flalign\\*|gathered|gather|gather\\*"
)
_ALIGN_BEGIN = re.compile(r"\\begin\{(" + _ALIGN_ENVS + r")\}(?:\{\d+\})?")
_ENV_TOKEN = re.compile(r"\\(begin|end)\{([^{}]*)\}")


def _drop_alignment_points(body: str) -> str:
    """Remove the "&" of ``body`` that are not inside a nested environment."""

    out: list[str] = []
    depth = 0
    index = 0
    while index < len(body):
        match = _ENV_TOKEN.match(body, index)
        if match:
            depth += 1 if match.group(1) == "begin" else -1
            out.append(match.group(0))
            index = match.end()
            continue
        character = body[index]
        if character == "\\" and index + 1 < len(body):
            out.append(body[index : index + 2])
            index += 2
            continue
        if character == "&" and depth == 0:
            out.append(" ")
        else:
            out.append(character)
        index += 1
    return "".join(out)


def _flatten_alignments(text: str) -> str:
    """Rewrite alignment environments, and bare "&" rows, as one-column arrays."""

    while True:
        begin = _ALIGN_BEGIN.search(text)
        if begin is None:
            break
        env = begin.group(1)
        end_token = "\\end{" + env + "}"
        end = text.find(end_token, begin.end())
        if end < 0:
            break
        body = _drop_alignment_points(text[begin.end() : end])
        text = (
            text[: begin.start()]
            + "\\begin{array}{l}"
            + body
            + "\\end{array}"
            + text[end + len(end_token) :]
        )
    # Rows aligned with "&" but without any environment ("a &= b \\ c &= d").
    flattened = _drop_alignment_points(text)
    if flattened != text:
        if "\\\\" in flattened and "\\begin" not in flattened:
            flattened = "\\begin{array}{l}" + flattened + "\\end{array}"
        text = flattened
    return text


_INLINE_DELIMITER = re.compile(r"(?<!\\)\$+")
# Prose between two inline formulas: a connective or an item label ("d)").
_PROSE_BETWEEN_FORMULAS = re.compile(
    r"(?:[a-z]\)|y|o|e|u|si|para|donde|con|entonces|cuando|sea|and|or|if|for|"
    r"where|with|then|when|let)(?:\s*[.,;:])?",
    re.IGNORECASE,
)


def _join_inline_formulas(text: str) -> str:
    """Write ``$a$ si $b$`` as ``a \\text{ si } b``.

    OCR of a line with prose returns several inline formulas.  Left as they
    are, latex2mathml reads every "$" (and the prose between formulas) as
    identifiers, which the rules then take as factors of a product.
    """

    text = text.replace("\\(", "$").replace("\\)", "$")
    parts = _INLINE_DELIMITER.split(text)
    if len(parts) == 1:
        return text
    if len(parts) % 2 == 0:
        # Unpaired "$": drop them all.
        return " ".join(part.strip() for part in parts if part.strip())
    out: list[str] = []
    for index, part in enumerate(parts):
        stripped = part.strip()
        if not stripped:
            continue
        if index % 2 == 0 and _PROSE_BETWEEN_FORMULAS.fullmatch(stripped):
            out.append("\\text{ " + stripped + " }")
        else:
            # A formula, punctuation, or math split by stray "$$" (dz$$ $$=).
            out.append(stripped)
    return " ".join(out)


def clean_latex(latex: str) -> str:
    """Remove math delimiters and equation-numbering noise added by OCR.

    Also writes operator names as their LaTeX commands (\\mathrm{det} becomes
    \\det), drops the small spacing commands (\\, \\; \\: \\!) and writes
    alignment environments (aligned, eqnarray, ...) as one-column arrays.
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
    text = _join_inline_formulas(text)
    text = _OPERATOR_NAME.sub(r"\\\1 ", text)
    text = _SPANISH_OPERATOR_NAME.sub(
        lambda match: "\\" + _SPANISH_OPERATORS[match.group(1) or match.group(2)] + " ",
        text,
    )
    text = _SPANISH_WORD.sub(r"\1\\text{ \2 }", text)
    text = _SMALL_SPACES.sub(" ", text)
    text = _flatten_alignments(text)
    return " ".join(text.split())
