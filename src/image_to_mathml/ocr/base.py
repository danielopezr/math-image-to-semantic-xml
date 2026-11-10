"""OCR interface for equation images."""

from __future__ import annotations

from typing import Any, Protocol


class OCRError(RuntimeError):
    """Raised when the image-to-LaTeX backend is unavailable or fails."""


class OCREngine(Protocol):
    name: str

    def image_to_latex(self, image_path: str) -> str:
        """Return a LaTeX representation of the equation in ``image_path``."""


def open_rgb(image_path: str):
    """Open ``image_path`` as an RGB Pillow image or raise :class:`OCRError`."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is a base dependency
        raise OCRError("Pillow is required for image input") from exc
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise OCRError(f"Could not open image {image_path!r}: {exc}") from exc


def coerce_latex(value: Any, backend: str) -> str:
    """Normalise the many result shapes returned by OCR packages.

    Backends return a string, a ``(latex, elapsed)`` tuple, a list of batched
    results or a dictionary with fields such as ``text`` or ``rec_formula``.
    """

    while isinstance(value, (list, tuple)):
        if not value:
            raise OCRError(f"{backend} returned an empty result")
        value = value[0]

    if isinstance(value, dict):
        for key in ("text", "latex", "rec_formula", "rec_text", "formula"):
            if key in value:
                return coerce_latex(value[key], backend)
        raise OCRError(
            f"{backend} returned an unexpected result object without LaTeX text"
        )

    if not isinstance(value, str) or not value.strip():
        raise OCRError(f"{backend} returned an empty LaTeX string")
    return value.strip()
