"""Pix2Text math-formula-recognition (MFR) adapter, loaded lazily."""

from __future__ import annotations

import os

from .base import OCRError, coerce_latex
from .preprocess import PreprocessConfig, preprocess_image


class Pix2TextOCR:
    name = "pix2text"

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        preprocess: PreprocessConfig | None = None,
    ):
        self._model = None
        self.model_name = (
            model_name or os.getenv("IMAGE_TO_MATHML_PIX2TEXT_MODEL") or "mfr-1.5"
        )
        self.model_backend = os.getenv("IMAGE_TO_MATHML_PIX2TEXT_BACKEND") or "onnx"
        self.device = device or os.getenv("IMAGE_TO_MATHML_OCR_DEVICE") or None
        self.preprocess = preprocess or PreprocessConfig.from_env()

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from pix2text import LatexOCR
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise OCRError(
                "pix2text is not installed. Install it with "
                "`pip install -e '.[pix2text]'`."
            ) from exc
        kwargs = {
            "model_name": self.model_name,
            "model_backend": self.model_backend,
        }
        if self.device:
            kwargs["device"] = self.device
        try:
            self._model = LatexOCR(**kwargs)
        except Exception as exc:  # pragma: no cover - model download/runtime errors
            raise OCRError(f"Could not load the pix2text model: {exc}") from exc
        return self._model

    def image_to_latex(self, image_path: str) -> str:
        model = self._load()
        image = preprocess_image(image_path, self.preprocess)
        try:
            result = model.recognize(image)
        except Exception as exc:  # pragma: no cover - runtime/model errors
            raise OCRError(f"pix2text failed on {image_path!r}: {exc}") from exc
        return coerce_latex(result, self.name)
