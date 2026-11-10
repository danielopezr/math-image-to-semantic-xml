"""Texify adapter (lazy import: the model is loaded on first use)."""

from __future__ import annotations

import os

from .base import OCRError, coerce_latex
from .preprocess import PreprocessConfig, preprocess_image


def _patch_transformers_config_compat() -> None:
    """Make texify 0.2.x work with recent ``transformers`` releases.

    ``transformers>=4.45`` builds the config ``repr`` through ``to_diff_dict``.
    Texify's composition config keeps ``encoder``/``decoder`` as plain dicts at
    that point, so the recursive diff raises ``AttributeError``.  Returning the
    full config dictionary is enough for logging and serialisation and avoids
    the crash.
    """

    try:
        from texify.model.config import TexifyConfig
    except ImportError:  # pragma: no cover - depends on the optional extra
        return

    if getattr(TexifyConfig, "_img2mathml_compat", False):
        return

    def _to_diff_dict(self):
        return self.to_dict()

    _to_diff_dict._img2mathml_compat = True
    TexifyConfig.to_diff_dict = _to_diff_dict
    TexifyConfig._img2mathml_compat = True


class TexifyOCR:
    name = "texify"

    def __init__(
        self,
        checkpoint: str | None = None,
        device: str | None = None,
        preprocess: PreprocessConfig | None = None,
    ):
        self._model = None
        self._processor = None
        self.checkpoint = (
            checkpoint or os.getenv("IMAGE_TO_MATHML_TEXIFY_MODEL") or None
        )
        self.device = device or os.getenv("IMAGE_TO_MATHML_OCR_DEVICE") or None
        self.preprocess = preprocess or PreprocessConfig.from_env()

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

        _patch_transformers_config_compat()

        model_kwargs = {}
        if self.checkpoint:
            model_kwargs["checkpoint"] = self.checkpoint
        if self.device:
            model_kwargs["device"] = self.device

        try:
            import contextlib
            import io

            with contextlib.redirect_stdout(io.StringIO()):
                self._model = load_model(**model_kwargs)
                self._processor = load_processor()
        except Exception as exc:  # pragma: no cover - model download/runtime errors
            raise OCRError(f"Could not load the texify model: {exc}") from exc
        return self._model, self._processor

    def image_to_latex(self, image_path: str) -> str:
        try:
            from texify.inference import batch_inference
        except ImportError as exc:  # pragma: no cover
            raise OCRError("texify is required for this backend") from exc
        model, processor = self._load()
        image = preprocess_image(image_path, self.preprocess)
        try:
            results = batch_inference([image], model, processor)
        except Exception as exc:  # pragma: no cover - runtime/model errors
            raise OCRError(f"texify failed on {image_path!r}: {exc}") from exc
        return coerce_latex(results, self.name)
