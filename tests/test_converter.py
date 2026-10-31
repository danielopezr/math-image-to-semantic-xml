"""Pruebas de la etapa de conversion.

Las pruebas con ``fake`` no necesitan latex2mathml. Las del final usan la libreria real y se omiten si no esta instalada.
Sus aserciones sobre la estructura exacta del MathML no se han verificado
todavia: si alguna falla, es informacion util sobre la libreria, no un
error del pipeline.
"""
import pytest

from app.converter import MATHML_NS, latex_to_mathml, validate_mathml
from app.models import ConversionErrorKind, StageStatus

NS = f'xmlns="{MATHML_NS}"'


class FakeSyntaxError(Exception):
    pass


FakeSyntaxError.__module__ = "latex2mathml.exceptions"


def codes(issues):
    return [i.code for i in issues]


# ----------------------------------------------------------- con conversor falso
def test_ok_con_mathml_valido():
    r = latex_to_mathml("x", converter=lambda s: f"<math {NS}><mi>x</mi></math>")
    assert r.status is StageStatus.OK
    assert r.error_kind is ConversionErrorKind.NONE
    assert "PRESENTATION_ONLY" in codes(r.warnings)


def test_entrada_vacia_es_error_de_sintaxis():
    r = latex_to_mathml("  ", converter=lambda s: "")
    assert r.status is StageStatus.FAILED
    assert r.error_kind is ConversionErrorKind.LATEX_SYNTAX


def test_excepcion_de_la_libreria_es_sintaxis():
    def boom(s):
        raise FakeSyntaxError("falta denominador")

    r = latex_to_mathml(r"\frac{1}", converter=boom)
    assert r.error_kind is ConversionErrorKind.LATEX_SYNTAX
    assert r.errors[0].code == "FakeSyntaxError"


def test_otra_excepcion_es_error_de_conversion():
    def boom(s):
        raise KeyError("x")

    r = latex_to_mathml("x", converter=boom)
    assert r.error_kind is ConversionErrorKind.CONVERSION


def test_mathml_mal_formado():
    r = latex_to_mathml("x", converter=lambda s: f"<math {NS}><mi>x</math>")
    assert r.status is StageStatus.FAILED
    assert r.error_kind is ConversionErrorKind.CONVERSION
    assert "MATHML_NOT_WELL_FORMED" in codes(r.errors)


def test_sin_espacio_de_nombres():
    errors, _ = validate_mathml("<math><mi>x</mi></math>")
    assert "BAD_ROOT" in codes(errors)


def test_mathml_de_contenido_rechazado():
    errors, _ = validate_mathml(f"<math {NS}><apply><ci>f</ci></apply></math>")
    assert "FOREIGN_ELEMENT" in codes(errors) or "CONTENT_MATHML" in codes(errors)


def test_contenido_con_espacio_de_nombres_rechazado():
    ml = f"<math {NS}><mrow><apply><ci>f</ci></apply></mrow></math>"
    # sin prefijo, <apply> hereda el espacio de nombres MathML por defecto
    errors, _ = validate_mathml(ml)
    assert "CONTENT_MATHML" in codes(errors)


def test_merror_rechazado():
    errors, _ = validate_mathml(f"<math {NS}><merror><mtext>x</mtext></merror></math>")
    assert "MERROR_PRESENT" in codes(errors)


def test_elemento_desconocido_es_advertencia():
    errors, warnings = validate_mathml(f"<math {NS}><mfoo/></math>")
    assert errors == []
    assert "UNKNOWN_ELEMENT" in codes(warnings)


def test_salida_vacia_del_conversor():
    r = latex_to_mathml("x", converter=lambda s: "")
    assert "EMPTY_OUTPUT" in codes(r.errors)


# --------------------------------------------------------------- con la libreria
@pytest.fixture()
def lib():
    return pytest.importorskip("latex2mathml")


@pytest.mark.parametrize(
    "latex, tag",
    [
        (r"\frac{a}{b}", "mfrac"),
        (r"\sqrt{x}", "msqrt"),
        (r"x^2", "msup"),
    ],
)
def test_real_estructura_basica(lib, latex, tag):
    r = latex_to_mathml(latex)
    assert r.status is StageStatus.OK, r.errors
    assert f"<{tag}" in r.mathml


def test_real_ecuacion_simple(lib):
    r = latex_to_mathml("x = 0")
    assert r.status is StageStatus.OK, r.errors
    assert r.mathml_kind == "presentation"


def test_real_sumatoria_e_integral_convierten(lib):
    assert latex_to_mathml(r"\sum_{i=1}^{n} i").status is StageStatus.OK
    assert latex_to_mathml(r"\int_0^1 x\,dx").status is StageStatus.OK


def test_real_latex_invalido_no_da_ok(lib):
    r = latex_to_mathml(r"\frac{1}")
    assert r.status is StageStatus.FAILED
