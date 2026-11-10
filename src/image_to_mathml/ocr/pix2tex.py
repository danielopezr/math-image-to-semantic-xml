"""pix2tex adapter (lazy import: the model is loaded on first use)."""

from __future__ import annotations

from .base import OCRError, coerce_latex
from .preprocess import PreprocessConfig, preprocess_image


class Pix2TexOCR:
    name = "pix2tex"

    def __init__(self, preprocess: PreprocessConfig | None = None) -> None:
        self._model = None
        self.preprocess = preprocess or PreprocessConfig.from_env()

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
        model = self._load()
        image = preprocess_image(image_path, self.preprocess)
        try:
            latex = model(image)
        except Exception as exc:  # pragma: no cover - runtime/model errors
            raise OCRError(f"pix2tex failed on {image_path!r}: {exc}") from exc
        return coerce_latex(latex, self.name)
