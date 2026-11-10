"""PaddleOCR PP-FormulaNet adapter, loaded lazily."""

from __future__ import annotations

import os

from .base import OCRError, coerce_latex
from .preprocess import PreprocessConfig, preprocess_image


class PaddleOCRFormula:
    name = "paddleocr"

    def __init__(
        self, model_name: str | None = None, preprocess: PreprocessConfig | None = None
    ):
        self._model = None
        self.model_name = (
            model_name
            or os.getenv("IMAGE_TO_MATHML_PADDLEOCR_MODEL")
            or "PP-FormulaNet_plus-M"
        )
        self.preprocess = preprocess or PreprocessConfig.from_env()

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from paddleocr import FormulaRecognition
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise OCRError(
                "paddleocr is not installed. Install it with "
                "`pip install -e '.[paddleocr]'`."
            ) from exc
        try:
            self._model = FormulaRecognition(model_name=self.model_name)
        except Exception as exc:  # pragma: no cover - model download/runtime errors
            raise OCRError(f"Could not load the PaddleOCR model: {exc}") from exc
        return self._model

    def image_to_latex(self, image_path: str) -> str:
        model = self._load()
        image = preprocess_image(image_path, self.preprocess)
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - numpy is a paddle dependency
            raise OCRError("numpy is required for paddleocr") from exc
        array = np.asarray(image)[:, :, ::-1].copy()
        try:
            results = model.predict(array)
        except Exception as exc:  # pragma: no cover - runtime/model errors
            raise OCRError(f"paddleocr failed on {image_path!r}: {exc}") from exc
        if not results:
            raise OCRError("paddleocr returned an empty result")
        return coerce_latex(results, self.name)
