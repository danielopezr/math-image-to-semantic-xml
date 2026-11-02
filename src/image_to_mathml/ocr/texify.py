"""Texify adapter (lazy import: the model is loaded on first use)."""

from __future__ import annotations

from .base import OCRError


class TexifyOCR:
    name = "texify"

    def __init__(self) -> None:
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        try:
            from texify.model.model import load_model
            from texify.model.processor import load_processor
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise OCRError(
                "texify is not installed. Install it with `pip install -e '.[texify]'`."
            ) from exc
        try:
            self._model = load_model()
            self._processor = load_processor()
        except Exception as exc:  # pragma: no cover - model download/runtime errors
            raise OCRError(f"Could not load the texify model: {exc}") from exc
        return self._model, self._processor

    def image_to_latex(self, image_path: str) -> str:
        try:
            from PIL import Image
            from texify.inference import batch_inference
        except ImportError as exc:  # pragma: no cover
            raise OCRError("Pillow and texify are required for this backend") from exc
        model, processor = self._load()
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise OCRError(f"Could not open image {image_path!r}: {exc}") from exc
        try:
            results = batch_inference([image], model, processor)
        except Exception as exc:  # pragma: no cover - runtime/model errors
            raise OCRError(f"texify failed on {image_path!r}: {exc}") from exc
        latex = results[0] if results else ""
        if not latex or not latex.strip():
            raise OCRError("texify returned an empty LaTeX string")
        return latex.strip()
