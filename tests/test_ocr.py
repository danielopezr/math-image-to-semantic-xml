import sys
import types

import pytest

from image_to_mathml.ocr import factory
from image_to_mathml.ocr.base import OCRError, coerce_latex
from image_to_mathml.ocr.paddleocr import PaddleOCRFormula
from image_to_mathml.ocr.pix2text import Pix2TextOCR
from image_to_mathml.ocr.preprocess import PreprocessConfig
from image_to_mathml.ocr.rapid_latex_ocr import RapidLaTeXOCR
from image_to_mathml.ocr.texify import _patch_transformers_config_compat


@pytest.fixture
def equation_image(tmp_path):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (160, 60), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "x", fill="black")
    path = tmp_path / "equation.png"
    image.save(path)
    return str(path)


def test_build_ocr_none_disables_ocr():
    assert factory.build_ocr("none") is None
    assert factory.build_ocr("off") is None


def test_build_ocr_unknown_backend_lists_options():
    with pytest.raises(OCRError, match="Unknown OCR backend"):
        factory.build_ocr("does-not-exist")


def test_canonical_aliases():
    assert factory._canonical_name("latexocr") == "pix2tex"
    assert factory._canonical_name("rapidocr") == "rapid_latex_ocr"
    assert factory._canonical_name("p2t") == "pix2text"
    assert factory._canonical_name("PP-FormulaNet") == "paddleocr"


def test_auto_uses_first_installed_backend(monkeypatch):
    monkeypatch.setattr(
        factory, "_installed", lambda module: module == "rapid_latex_ocr"
    )
    engine = factory.build_ocr("auto")
    assert engine is not None
    assert engine.name == "rapid_latex_ocr"


def test_auto_prefers_texify(monkeypatch):
    monkeypatch.setattr(factory, "_installed", lambda module: True)
    engine = factory.build_ocr("auto")
    assert engine is not None
    assert engine.name == "texify"


def test_auto_reports_install_hints(monkeypatch):
    monkeypatch.setattr(factory, "_installed", lambda module: False)
    with pytest.raises(OCRError, match="No equation OCR backend"):
        factory.build_ocr("auto")


@pytest.mark.parametrize(
    "value",
    [r"\frac{a}{b}", (r"\frac{a}{b}", 0.01), [{"text": r"\frac{a}{b}"}]],
)
def test_coerce_latex_result_shapes(value):
    assert coerce_latex(value, "test") == r"\frac{a}{b}"


def test_coerce_latex_rejects_empty_result():
    with pytest.raises(OCRError, match="unexpected result object"):
        coerce_latex({"score": 0.1}, "test")


def test_rapid_latex_ocr_adapter_unwraps_tuple(equation_image):
    engine = RapidLaTeXOCR(preprocess=PreprocessConfig(enabled=False))
    engine._model = lambda image: (r"\frac{a}{b}", 0.01)
    assert engine.image_to_latex(equation_image) == r"\frac{a}{b}"


def test_pix2text_adapter_reads_text_field(equation_image):
    engine = Pix2TextOCR(preprocess=PreprocessConfig(enabled=False))

    class FakeModel:
        def recognize(self, image):
            return {"text": "x^{2}", "score": 0.99}

    engine._model = FakeModel()
    assert engine.image_to_latex(equation_image) == "x^{2}"


def test_paddleocr_adapter_reads_rec_formula(equation_image):
    engine = PaddleOCRFormula(preprocess=PreprocessConfig(enabled=False))

    class FakeModel:
        def predict(self, image):
            return [{"rec_formula": [r"\frac{a}{b}"]}]

    engine._model = FakeModel()
    assert engine.image_to_latex(equation_image) == r"\frac{a}{b}"


def test_texify_transformers_compatibility_patch(monkeypatch):
    class FakeConfig:
        def to_dict(self):
            return {"model_type": "vision-encoder-decoder"}

    config_module = types.ModuleType("texify.model.config")
    config_module.TexifyConfig = FakeConfig
    model_module = types.ModuleType("texify.model")
    package_module = types.ModuleType("texify")

    monkeypatch.setitem(sys.modules, "texify", package_module)
    monkeypatch.setitem(sys.modules, "texify.model", model_module)
    monkeypatch.setitem(sys.modules, "texify.model.config", config_module)

    _patch_transformers_config_compat()

    assert FakeConfig().to_diff_dict() == {"model_type": "vision-encoder-decoder"}
