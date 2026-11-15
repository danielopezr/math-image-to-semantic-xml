"""Build the configured OCR backend without importing heavy models eagerly."""

from __future__ import annotations

import importlib.util
from typing import Any

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
    "ollama_vision",
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
    "ollama": "ollama_vision",
    "vision": "ollama_vision",
    "qwen3vl": "ollama_vision",
    "qwen3_vl": "ollama_vision",
}

OCR_BACKEND_CHOICES = tuple(dict.fromkeys((*OCR_BACKENDS, *_ALIASES)))

# Highest-quality/commonly-installed first.  ``auto`` picks the first available.
_AUTO_ORDER = (
    "texify",
    "pix2text",
    "pix2tex",
    "paddleocr",
    "rapid_latex_ocr",
    "ollama_vision",
)

_MODULES = {
    "texify": "texify",
    "pix2tex": "pix2tex",
    "rapid_latex_ocr": "rapid_latex_ocr",
    "pix2text": "pix2text",
    "paddleocr": "paddleocr",
    "ollama_vision": "requests",
}

_INSTALL_HINTS = {
    "texify": "pip install -e '.[texify]'",
    "pix2tex": "pip install -e '.[pix2tex]'",
    "rapid_latex_ocr": "pip install -e '.[rapid-latex-ocr]'",
    "pix2text": "pip install -e '.[pix2text]'",
    "paddleocr": "pip install -e '.[paddleocr]'",
    "ollama_vision": "ollama pull qwen3-vl:4b-instruct",
}


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def canonical_backend_name(backend: str | None) -> str:
    name = (backend or "auto").strip().lower().replace("-", "_")
    return _ALIASES.get(name, name)


_canonical_name = canonical_backend_name


def _make(
    name: str,
    device: str | None,
    model: str | None,
    preprocess: PreprocessConfig,
    stream: bool | None = None,
    timeout: float | None = None,
    ollama_options: dict[str, Any] | None = None,
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
    if name == "ollama_vision":
        from .ollama_vision import OllamaVisionOCR

        options = dict(ollama_options or {})
        if model is not None:
            options["model"] = model
        if timeout is not None:
            options["timeout"] = timeout
        if stream is not None:
            options["stream"] = stream
        return OllamaVisionOCR(preprocess=preprocess, **options)
    raise AssertionError(f"unhandled OCR backend: {name}")


def build_ocr(
    backend: str | None,
    *,
    device: str | None = None,
    model: str | None = None,
    preprocess: PreprocessConfig | None = None,
    stream: bool | None = None,
    timeout: float | None = None,
    ollama_options: dict[str, Any] | None = None,
) -> OCREngine | None:
    """Build an OCR engine.

    ``ollama_options`` configures the ``ollama_vision`` backend (``model``,
    ``base_url``, ``timeout``, ``num_ctx``, ``num_predict``, ``max_side``,
    ``think``, ``retries``); explicit ``model``/``timeout``/``stream``
    arguments take precedence over it.
    """

    name = canonical_backend_name(backend)
    if name == "none":
        return None

    preprocess = preprocess or PreprocessConfig.from_env()

    if name == "auto":
        for candidate in _AUTO_ORDER:
            if _installed(_MODULES[candidate]):
                return _make(
                    candidate,
                    device,
                    model,
                    preprocess,
                    stream,
                    timeout,
                    ollama_options,
                )
        hints = ", ".join(f"`{_INSTALL_HINTS[candidate]}`" for candidate in _AUTO_ORDER)
        raise OCRError(
            "No equation OCR backend is available. Install one of texify, "
            f"pix2tex, rapid_latex_ocr, pix2text or paddleocr, or pull an "
            f"Ollama vision model ({hints}), or pass the LaTeX directly "
            "with --latex."
        )

    if name not in _MODULES:
        options = ", ".join(OCR_BACKENDS)
        raise OCRError(f"Unknown OCR backend {backend!r}. Use one of: {options}.")
    return _make(name, device, model, preprocess, stream, timeout, ollama_options)
