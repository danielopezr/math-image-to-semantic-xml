"""LaTeX to Presentation MathML conversion (flow 2, first step)."""

from __future__ import annotations

from .latex import clean_latex


class ConversionError(RuntimeError):
    """Raised when LaTeX cannot be converted to MathML."""


class LatexToMathML:
    """Thin lazy wrapper around ``latex2mathml``."""

    def __init__(self) -> None:
        self._converter = None

    def _load(self):
        if self._converter is not None:
            return self._converter
        try:
            from latex2mathml.converter import convert
        except ImportError as exc:  # pragma: no cover - dependency is required
            raise ConversionError(
                "latex2mathml is not installed. Install the project dependencies "
                "with `pip install -e .`."
            ) from exc
        self._converter = convert
        return self._converter

    def convert(self, latex: str) -> str:
        text = clean_latex(latex)
        if not text:
            raise ConversionError("The LaTeX expression is empty")
        convert = self._load()
        try:
            markup = convert(text)
        except Exception as exc:
            raise ConversionError(f"Could not convert LaTeX to MathML: {exc}") from exc
        if not markup or "<math" not in markup:
            raise ConversionError("latex2mathml returned no <math> element")
        return markup
