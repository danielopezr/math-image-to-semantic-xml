import numpy as np
import pytest
from PIL import Image, ImageDraw

from image_to_mathml.ocr import preprocess as preprocess_module
from image_to_mathml.ocr.preprocess import (
    PreprocessConfig,
    _deskew,
    preprocess_image,
)


def _make_image(tmp_path, size, background="white", bar=None):
    image = Image.new("RGB", size, background)
    if bar is not None:
        ImageDraw.Draw(image).rectangle(bar, fill="black")
    path = tmp_path / "equation.png"
    image.save(path)
    return str(path)


def _principal_angle(image):
    gray = np.asarray(image.convert("L"))
    mask = gray < 128
    ys, xs = np.nonzero(mask)
    xs = xs.astype(np.float64) - xs.mean()
    ys = ys.astype(np.float64) - ys.mean()
    return 0.5 * np.degrees(
        np.arctan2(2 * np.sum(xs * ys), np.sum(xs * xs) - np.sum(ys * ys))
    )


def test_disabled_preprocessing_returns_original_size(tmp_path):
    path = _make_image(tmp_path, (120, 40))
    image = preprocess_image(path, PreprocessConfig(enabled=False))
    assert image.size == (120, 40)
    assert image.mode == "RGB"


def test_default_pipeline_runs_end_to_end(tmp_path):
    path = _make_image(tmp_path, (120, 40), bar=(30, 15, 90, 25))
    image = preprocess_image(path, PreprocessConfig.from_env(profile="standard"))
    assert image.mode == "RGB"
    assert image.size[0] > 0
    assert image.size[1] > 0


def test_safe_profile_is_minimal():
    config = PreprocessConfig.from_env(profile="safe")
    assert config.enabled
    assert config.normalize_polarity
    assert not config.crop_border
    assert not config.deskew
    assert not config.contrast
    assert not config.sharpen
    assert not config.upscale_small


def test_standard_profile_enables_common_steps():
    config = PreprocessConfig.from_env(profile="standard")
    assert config.crop_border
    assert config.deskew
    assert config.contrast
    assert config.sharpen
    assert config.upscale_small
    assert not config.clahe
    assert not config.binarize


def test_polarity_ignores_dark_page_border(tmp_path):
    image = Image.new("RGB", (120, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 119, 79), outline="black", width=4)
    draw.rectangle((30, 35, 90, 45), fill="black")
    path = tmp_path / "equation.png"
    image.save(path)

    result = preprocess_image(
        str(path),
        PreprocessConfig(
            crop_border=False,
            deskew=False,
            contrast=False,
            sharpen=False,
            upscale_small=False,
        ),
    )
    assert np.asarray(result.convert("L")).mean() > 127


def test_crop_border_removes_empty_margins(tmp_path):
    path = _make_image(tmp_path, (200, 100), bar=(50, 30, 149, 69))
    config = PreprocessConfig(
        normalize_polarity=False,
        crop_border=True,
        border_padding=0,
        deskew=False,
        contrast=False,
        sharpen=False,
        upscale_small=False,
    )
    image = preprocess_image(path, config)
    assert image.size == (100, 40)


def test_normalize_polarity_inverts_dark_background(tmp_path):
    path = _make_image(
        tmp_path, (100, 50), background=(25, 25, 25), bar=(20, 15, 80, 35)
    )
    config = PreprocessConfig(
        crop_border=False,
        deskew=False,
        contrast=False,
        sharpen=False,
        upscale_small=False,
    )
    image = preprocess_image(path, config)
    assert np.asarray(image.convert("L")).mean() > 127


def test_upscale_small_images(tmp_path):
    path = _make_image(tmp_path, (20, 10), bar=(5, 3, 15, 7))
    config = PreprocessConfig(
        normalize_polarity=False,
        crop_border=False,
        deskew=False,
        contrast=False,
        sharpen=False,
        upscale_small=True,
        min_short_side=48,
    )
    image = preprocess_image(path, config)
    assert min(image.size) >= 48


def test_deskew_reduces_rotation():
    image = Image.new("RGB", (300, 120), "white")
    ImageDraw.Draw(image).rectangle((60, 57, 250, 63), fill="black")
    slanted = image.rotate(8, resample=Image.BICUBIC, expand=True, fillcolor="white")

    result = _deskew(slanted, 10)

    assert abs(_principal_angle(result)) < abs(_principal_angle(slanted))
    assert abs(_principal_angle(result)) < 2


def test_preprocessed_image_is_saved_by_default(tmp_path, monkeypatch):
    output_dir = tmp_path / "preprocessed"
    monkeypatch.delenv("IMAGE_TO_MATHML_PREPROCESS_DEBUG_DIR", raising=False)
    monkeypatch.setattr(preprocess_module, "PREPROCESSED_IMAGE_DIR", str(output_dir))
    path = _make_image(tmp_path, (80, 30), bar=(20, 10, 60, 20))

    preprocess_image(path, PreprocessConfig.from_env(profile="safe"))

    assert (output_dir / "equation_preprocessed.png").is_file()


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("IMAGE_TO_MATHML_PREPROCESS", "false")
    monkeypatch.setenv("IMAGE_TO_MATHML_PREPROCESS_DESKEW", "false")
    monkeypatch.setenv("IMAGE_TO_MATHML_PREPROCESS_MIN_SIDE", "64")
    config = PreprocessConfig.from_env()
    assert not config.enabled
    assert not config.deskew
    assert config.min_short_side == 64


@pytest.mark.parametrize("bad_value", ["not-a-number", ""])
def test_config_from_env_ignores_invalid_numbers(monkeypatch, bad_value):
    monkeypatch.setenv("IMAGE_TO_MATHML_PREPROCESS_MIN_SIDE", bad_value)
    config = PreprocessConfig.from_env()
    assert config.min_short_side == PreprocessConfig.min_short_side
