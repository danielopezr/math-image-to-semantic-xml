"""OCR interface for equation images."""

from __future__ import annotations

from typing import Any, Protocol


class OCRError(RuntimeError):
    """Raised when the image-to-LaTeX backend is unavailable or fails."""


class OCREngine(Protocol):
    name: str

    def image_to_latex(self, image_path: str) -> str:
        """Return a LaTeX representation of the equation in ``image_path``."""


def flatten_transparency(image):
    """Composite a transparent image on a background that contrasts with it.

    ``Image.convert("RGB")`` turns transparent pixels black, so black text on
    a transparent background (the usual export of equation editors) becomes a
    completely black image.  The ink colour is estimated from the opaque
    pixels: dark ink goes on white, light ink on black (polarity normalisation
    later inverts the light case).
    """

    from PIL import Image

    has_alpha = image.mode in {"RGBA", "LA", "PA"} or (
        image.mode == "P" and "transparency" in image.info
    )
    if not has_alpha:
        return image.convert("RGB")

    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    if alpha.getextrema() == (255, 255):
        return rgba.convert("RGB")

    opaque = alpha.point(lambda value: 255 if value > 127 else 0)
    gray = rgba.convert("L")
    histogram = gray.histogram(mask=opaque)
    count = sum(histogram)
    mean = (
        sum(level * hits for level, hits in enumerate(histogram)) / count
        if count
        else 0
    )
    background_color = (255, 255, 255, 255) if mean < 128 else (0, 0, 0, 255)

    background = Image.new("RGBA", rgba.size, background_color)
    return Image.alpha_composite(background, rgba).convert("RGB")


def open_rgb(image_path: str):
    """Open ``image_path`` as an RGB Pillow image or raise :class:`OCRError`."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is a base dependency
        raise OCRError("Pillow is required for image input") from exc
    try:
        with Image.open(image_path) as image:
            image.load()
            return flatten_transparency(image)
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
