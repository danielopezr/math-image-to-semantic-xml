"""pix2tex adapter (lazy import: the model is loaded on first use)."""

from __future__ import annotations

from .base import OCRError


class Pix2TexOCR:
    name = "pix2tex"

    def __init__(self) -> None:
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from pix2tex.cli import LatexOCR
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise OCRError(
                "pix2tex is not installed. Install it with "
                "`pip install -e '.[pix2tex]'`."
            ) from exc
        try:
            self._model = LatexOCR()
        except Exception as exc:  # pragma: no cover - model download/runtime errors
            raise OCRError(f"Could not load the pix2tex model: {exc}") from exc
        return self._model

    def image_to_latex(self, image_path: str) -> str:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise OCRError("Pillow is required for image input") from exc
        model = self._load()
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise OCRError(f"Could not open image {image_path!r}: {exc}") from exc
        try:
            latex = model(image)
        except Exception as exc:  # pragma: no cover - runtime/model errors
            raise OCRError(f"pix2tex failed on {image_path!r}: {exc}") from exc
        if not latex or not latex.strip():
            raise OCRError("pix2tex returned an empty LaTeX string")
        return latex.strip()
