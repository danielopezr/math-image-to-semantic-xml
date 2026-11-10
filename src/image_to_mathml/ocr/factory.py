"""Build the configured OCR backend without importing heavy models eagerly."""

from __future__ import annotations

import importlib.util

from .base import OCREngine, OCRError
from .preprocess import PreprocessConfig

# Canonical names accepted by ``--ocr``/``IMAGE_TO_MATHML_OCR``.
OCR_BACKENDS = (
    "auto",
    "texify",
    "pix2tex",
    "rapid_latex_ocr",
    "pix2text",
    "paddleocr",
    "none",
)

_ALIASES = {
    "off": "none",
    "disabled": "none",
    "latexocr": "pix2tex",
    "latex_ocr": "pix2tex",
    "rapidocr": "rapid_latex_ocr",
    "rapid": "rapid_latex_ocr",
    "rapidlatexocr": "rapid_latex_ocr",
    "p2t": "pix2text",
    "mfr": "pix2text",
    "paddle": "paddleocr",
    "paddle_ocr": "paddleocr",
    "ppformulanet": "paddleocr",
    "pp_formulanet": "paddleocr",
    "pp_formula_net": "paddleocr",
    "formulanet": "paddleocr",
}

# Highest-quality/commonly-installed first.  ``auto`` picks the first available.
_AUTO_ORDER = ("texify", "pix2text", "pix2tex", "paddleocr", "rapid_latex_ocr")

_MODULES = {
    "texify": "texify",
    "pix2tex": "pix2tex",
    "rapid_latex_ocr": "rapid_latex_ocr",
    "pix2text": "pix2text",
    "paddleocr": "paddleocr",
}

_INSTALL_HINTS = {
    "texify": "pip install -e '.[texify]'",
    "pix2tex": "pip install -e '.[pix2tex]'",
    "rapid_latex_ocr": "pip install -e '.[rapid-latex-ocr]'",
    "pix2text": "pip install -e '.[pix2text]'",
    "paddleocr": "pip install -e '.[paddleocr]'",
}


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _canonical_name(backend: str | None) -> str:
    name = (backend or "auto").strip().lower().replace("-", "_")
    return _ALIASES.get(name, name)


def _make(
    name: str,
    device: str | None,
    model: str | None,
    preprocess: PreprocessConfig,
) -> OCREngine:
    if name == "texify":
        from .texify import TexifyOCR

        return TexifyOCR(checkpoint=model, device=device, preprocess=preprocess)
    if name == "pix2tex":
        from .pix2tex import Pix2TexOCR

        return Pix2TexOCR(preprocess=preprocess)
    if name == "rapid_latex_ocr":
        from .rapid_latex_ocr import RapidLaTeXOCR

        return RapidLaTeXOCR(preprocess=preprocess)
    if name == "pix2text":
        from .pix2text import Pix2TextOCR

        return Pix2TextOCR(model_name=model, device=device, preprocess=preprocess)
    if name == "paddleocr":
        from .paddleocr import PaddleOCRFormula

        return PaddleOCRFormula(model_name=model, preprocess=preprocess)
    raise AssertionError(f"unhandled OCR backend: {name}")


def build_ocr(
    backend: str | None,
    *,
    device: str | None = None,
    model: str | None = None,
    preprocess: PreprocessConfig | None = None,
) -> OCREngine | None:
    name = _canonical_name(backend)
    if name == "none":
        return None

    preprocess = preprocess or PreprocessConfig.from_env()

    if name == "auto":
        for candidate in _AUTO_ORDER:
            if _installed(_MODULES[candidate]):
                return _make(candidate, device, model, preprocess)
        hints = ", ".join(f"`{_INSTALL_HINTS[candidate]}`" for candidate in _AUTO_ORDER)
        raise OCRError(
            "No equation OCR backend is installed. Install one of texify, "
            f"pix2tex, rapid_latex_ocr, pix2text or paddleocr ({hints}), "
            "or pass the LaTeX directly with --latex."
        )

    if name not in _MODULES:
        options = ", ".join(OCR_BACKENDS)
        raise OCRError(f"Unknown OCR backend {backend!r}. Use one of: {options}.")
    return _make(name, device, model, preprocess)
