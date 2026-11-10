"""Conservative image preprocessing for equation OCR."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

from .base import OCRError, open_rgb

PREPROCESSED_IMAGE_DIR = "results/preprocessed"


def _env_optional_bool(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _env_optional_float(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


_PROFILES = {
    "off": {"enabled": False},
    "safe": {"enabled": True, "normalize_polarity": True},
    "standard": {
        "enabled": True,
        "normalize_polarity": True,
        "crop_border": True,
        "deskew": True,
        "contrast": True,
        "sharpen": True,
        "upscale_small": True,
    },
    "aggressive": {
        "enabled": True,
        "normalize_polarity": True,
        "crop_border": True,
        "deskew": True,
        "contrast": True,
        "clahe": True,
        "sharpen": True,
        "denoise": True,
        "upscale_small": True,
    },
}

_ENV_OPTIONS = (
    ("normalize_polarity", "IMAGE_TO_MATHML_PREPROCESS_POLARITY", "bool"),
    ("crop_border", "IMAGE_TO_MATHML_PREPROCESS_CROP", "bool"),
    ("border_padding", "IMAGE_TO_MATHML_PREPROCESS_PADDING", "int"),
    ("deskew", "IMAGE_TO_MATHML_PREPROCESS_DESKEW", "bool"),
    ("max_deskew_angle", "IMAGE_TO_MATHML_PREPROCESS_MAX_ANGLE", "float"),
    ("contrast", "IMAGE_TO_MATHML_PREPROCESS_CONTRAST", "bool"),
    ("clahe", "IMAGE_TO_MATHML_PREPROCESS_CLAHE", "bool"),
    ("sharpen", "IMAGE_TO_MATHML_PREPROCESS_SHARPEN", "bool"),
    ("denoise", "IMAGE_TO_MATHML_PREPROCESS_DENOISE", "bool"),
    ("upscale_small", "IMAGE_TO_MATHML_PREPROCESS_UPSCALE", "bool"),
    ("min_short_side", "IMAGE_TO_MATHML_PREPROCESS_MIN_SIDE", "int"),
    ("binarize", "IMAGE_TO_MATHML_PREPROCESS_BINARIZE", "bool"),
)


@dataclass(frozen=True)
class PreprocessConfig:
    """Options for the image-to-image enhancement pipeline.

    The dataclass defaults and the ``safe`` profile are deliberately minimal:
    only background polarity is normalised.  Cropping, deskewing, contrast,
    sharpening and upscaling are available through the ``standard`` and
    ``aggressive`` profiles, but they can hurt already-clean images.
    """

    enabled: bool = True
    normalize_polarity: bool = True
    crop_border: bool = False
    border_padding: int = 8
    deskew: bool = False
    max_deskew_angle: float = 10.0
    contrast: bool = False
    clahe: bool = False
    sharpen: bool = False
    denoise: bool = False
    upscale_small: bool = False
    min_short_side: int = 48
    binarize: bool = False

    @classmethod
    def from_env(
        cls, enabled: bool | None = None, profile: str | None = None
    ) -> PreprocessConfig:
        profile_name = (
            (profile or os.getenv("IMAGE_TO_MATHML_PREPROCESS_PROFILE") or "safe")
            .strip()
            .lower()
        )
        values = {field.name: field.default for field in fields(cls)}
        values.update(_PROFILES.get(profile_name, _PROFILES["safe"]))

        parsers = {
            "bool": _env_optional_bool,
            "int": _env_optional_int,
            "float": _env_optional_float,
        }
        for field_name, env_name, kind in _ENV_OPTIONS:
            value = parsers[kind](env_name)
            if value is not None:
                values[field_name] = value
        if enabled is None:
            enabled = _env_optional_bool("IMAGE_TO_MATHML_PREPROCESS")
        if enabled is not None:
            values["enabled"] = enabled
        return cls(**values)


def preprocess_image(image_path: str, config: PreprocessConfig | None = None):
    """Load and enhance an equation image before sending it to an OCR engine."""

    cfg = config or PreprocessConfig.from_env()
    image = open_rgb(image_path)
    if not cfg.enabled:
        return image

    try:
        from PIL import ImageFilter, ImageOps

        image = ImageOps.exif_transpose(image)
        if cfg.normalize_polarity:
            image = _normalize_polarity(image)
        if cfg.crop_border:
            image = _crop_border(image, cfg.border_padding)
        if cfg.deskew:
            image = _deskew(image, cfg.max_deskew_angle)
        if cfg.denoise:
            image = image.filter(ImageFilter.MedianFilter(3))
        if cfg.upscale_small:
            image = _upscale_small(image, cfg.min_short_side)
        if cfg.contrast:
            image = _enhance_contrast(image, cfg.clahe)
        if cfg.sharpen:
            image = image.filter(
                ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3)
            )
        if cfg.binarize:
            image = _binarize(image)
    except Exception as exc:
        raise OCRError(f"Could not preprocess image {image_path!r}: {exc}") from exc

    debug_dir = os.getenv(
        "IMAGE_TO_MATHML_PREPROCESS_DEBUG_DIR", PREPROCESSED_IMAGE_DIR
    )
    if debug_dir:
        from pathlib import Path

        try:
            output = Path(debug_dir)
            output.mkdir(parents=True, exist_ok=True)
            image.save(output / f"{Path(image_path).stem}_preprocessed.png")
        except OSError:
            pass
    return image


def _background_level(gray) -> int:
    """Return the most frequent intensity, which is usually the page background."""

    histogram = gray.histogram()
    return max(range(256), key=histogram.__getitem__)


def _normalize_polarity(image):
    from PIL import ImageOps

    background = _background_level(image.convert("L"))
    if background < 127:
        return ImageOps.invert(image)
    return image


def _crop_border(image, padding: int):
    gray = image.convert("L")
    background = _background_level(gray)
    content = gray.point(lambda value: 255 if abs(value - background) > 30 else 0)
    bbox = content.getbbox()
    if bbox is None:
        return image
    left, top, right, bottom = bbox
    pad = max(0, padding)
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(image.width, right + pad)
    bottom = min(image.height, bottom + pad)
    if (left, top, right, bottom) == (0, 0, image.width, image.height):
        return image
    return image.crop((left, top, right, bottom))


def _deskew(image, max_angle: float):
    try:
        import numpy as np
        from PIL import Image
    except ImportError:  # pragma: no cover - numpy is an OCR dependency
        return image

    gray = image.convert("L")
    if max(gray.size) > 900:
        ratio = 900 / max(gray.size)
        gray = gray.resize(
            (max(1, int(gray.width * ratio)), max(1, int(gray.height * ratio))),
            Image.BILINEAR,
        )
    array = np.asarray(gray)
    threshold = _otsu_threshold(array)
    mask = array <= threshold
    ratio = float(mask.mean())
    if ratio < 0.005 or ratio > 0.6:
        return image

    angles = np.arange(-max_angle, max_angle + 0.01, 0.5)
    scores: dict[float, float] = {}
    for angle in angles:
        rotated = Image.fromarray((mask * 255).astype(np.uint8)).rotate(
            float(angle), resample=Image.BILINEAR, fillcolor=0
        )
        projection = np.asarray(rotated, dtype=np.float32).sum(axis=1)
        scores[float(angle)] = float(np.sum(np.diff(projection) ** 2))

    best_angle = max(scores, key=scores.get)
    baseline = scores.get(0.0, 0.0)
    if abs(best_angle) < 0.5:
        return image
    if baseline and scores[best_angle] < baseline * 1.02:
        return image

    return image.rotate(
        best_angle,
        resample=Image.BICUBIC,
        expand=True,
        fillcolor=(255, 255, 255),
    )


def _otsu_threshold(array) -> int:
    import numpy as np

    histogram = np.bincount(array.ravel(), minlength=256).astype(np.float64)
    total = array.size
    sum_all = float(np.dot(np.arange(256), histogram))
    sum_background = 0.0
    weight_background = 0.0
    best_variance = -1.0
    best_threshold = 127
    for threshold in range(256):
        weight_background += histogram[threshold]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += threshold * histogram[threshold]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_all - sum_background) / weight_foreground
        variance = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return best_threshold


def _enhance_contrast(image, clahe: bool):
    from PIL import ImageOps

    enhanced = ImageOps.autocontrast(image, cutoff=1)
    if not clahe:
        return enhanced
    try:
        import cv2
        import numpy as np
    except ImportError:  # pragma: no cover - OpenCV is optional
        return enhanced

    try:
        lab = cv2.cvtColor(np.asarray(enhanced), cv2.COLOR_RGB2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
        merged = cv2.cvtColor(
            cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2RGB
        )
        from PIL import Image

        return Image.fromarray(merged)
    except (cv2.error, TypeError, ValueError):  # pragma: no cover - fallback
        return enhanced


def _upscale_small(image, min_short_side: int):
    from PIL import Image

    short_side = min(image.size)
    if short_side <= 0 or short_side >= min_short_side:
        return image
    factor = min(8.0, min_short_side / short_side)
    size = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
    return image.resize(size, Image.LANCZOS)


def _binarize(image):
    from PIL import ImageOps

    gray = ImageOps.autocontrast(image.convert("L"))
    return gray.point(lambda value: 255 if value > 180 else 0).convert("RGB")
