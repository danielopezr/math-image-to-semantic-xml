"""Build the configured OCR backend without importing heavy models eagerly."""

from __future__ import annotations

import importlib.util

from .base import OCREngine, OCRError


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def build_ocr(backend: str | None) -> OCREngine | None:
    name = (backend or "auto").strip().lower()
    if name in {"none", "off"}:
        return None

    if name == "pix2tex":
        from .pix2tex import Pix2TexOCR

        return Pix2TexOCR()
    if name == "texify":
        from .texify import TexifyOCR

        return TexifyOCR()

    if name == "auto":
        if _installed("texify"):
            from .texify import TexifyOCR

            return TexifyOCR()
        if _installed("pix2tex"):
            from .pix2tex import Pix2TexOCR

            return Pix2TexOCR()
        raise OCRError(
            "No equation OCR backend is installed. Install texify or pix2tex, "
            "e.g. `pip install -e '.[texify]'` or `pip install -e '.[pix2tex]'`, "
            "or pass the LaTeX directly with --latex."
        )

    raise OCRError(
        f"Unknown OCR backend {backend!r}. Use auto, texify, pix2tex or none."
    )
