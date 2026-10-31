"""Pruebas de la etapa OCR usando un motor falso (no requieren pix2tex)."""
from pathlib import Path

import pytest
from PIL import Image

from app.models import StageStatus
from app.ocr import apply_manual_review, recognize_formula, retry_recognition


@pytest.fixture()
def png(tmp_path: Path) -> Path:
    p = tmp_path / "f.png"
    Image.new("RGB", (40, 20), "white").save(p)
    return p


def test_ok_conserva_latex_original(png):
    r = recognize_formula(png, engine=lambda img: r"\frac{a}{b}")
    assert r.status is StageStatus.OK
    assert r.latex_raw == r"\frac{a}{b}"
    assert r.latex == r"\frac{a}{b}"
    assert r.errors == []


def test_imagen_inexistente():
    r = recognize_formula("no_existe.png", engine=lambda img: "x")
    assert r.status is StageStatus.FAILED
    assert r.errors[0].code == "IMAGE_NOT_FOUND"


def test_formato_no_soportado(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hola")
    r = recognize_formula(p, engine=lambda img: "x")
    assert r.errors[0].code == "UNSUPPORTED_FORMAT"


def test_imagen_ilegible(tmp_path):
    p = tmp_path / "rota.png"
    p.write_bytes(b"no es una imagen")
    r = recognize_formula(p, engine=lambda img: "x")
    assert r.errors[0].code == "IMAGE_UNREADABLE"


def test_salida_vacia(png):
    r = recognize_formula(png, engine=lambda img: "   ")
    assert r.status is StageStatus.FAILED
    assert r.errors[0].code == "EMPTY_OUTPUT"


def test_error_del_motor(png):
    def boom(img):
        raise RuntimeError("fallo")

    r = recognize_formula(png, engine=boom)
    assert r.errors[0].code == "OCR_RUNTIME_ERROR"


def test_revision_manual_no_toca_original(png):
    r = recognize_formula(png, engine=lambda img: "x+1")
    rev = apply_manual_review(r, "x+2")
    assert rev.latex_raw == "x+1"
    assert rev.latex == "x+2"
    assert rev.reviewed_by_human


def test_revision_manual_rescata_fallo(png):
    r = recognize_formula(png, engine=lambda img: "")
    rev = apply_manual_review(r, "y=1")
    assert rev.status is StageStatus.OK
    assert rev.latex_raw is None
    assert rev.latex == "y=1"


def test_reintento_incrementa_intento(png):
    r1 = recognize_formula(png, engine=lambda img: "a")
    r2 = retry_recognition(r1, engine=lambda img: "b")
    assert (r1.attempt, r2.attempt) == (1, 2)
    assert r2.latex_raw == "b"
