"""Pruebas de la etapa de braille con MathCAT FALSO (no se simulan resultados reales de MathCAT)."""
import os
import tempfile
from pathlib import Path

from app.braille import apply_braille_review, export_braille_text, mathml_to_braille, non_braille_chars
from app.converter import validate_mathml
from app.mathcat import MathCAT, MathCATError
from app.models import MathCATConfiguration, StageStatus
from tests.test_mathcat import FakeEngine

NS = "http://www.w3.org/1998/Math/MathML"


def M(inner): return f'<math xmlns="{NS}">{inner}</math>'


# Expresiones de prueba pedidas (MathML de presentacion): se comprueba que cada una es valida
# y que llega SIN MODIFICAR al motor. No se comprueba ninguna salida braille concreta.
EXPRESSIONS = {
    "suma": M("<mi>x</mi><mo>+</mo><mn>2</mn>"),
    "ecuacion cuadratica": M("<msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><mn>3</mn><mi>x</mi><mo>=</mo><mn>10</mn>"),
    "(a+b)/c": M("<mfrac><mrow><mo>(</mo><mi>a</mi><mo>+</mo><mi>b</mi><mo>)</mo></mrow><mi>c</mi></mfrac>"),
    "f(x)=sin(x)": M("<mi>f</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>=</mo><mi>sin</mi><mo>(</mo><mi>x</mi><mo>)</mo>"),
    "log(x)": M("<mi>log</mi><mo>(</mo><mi>x</mi><mo>)</mo>"),
    "e^x": M("<msup><mi>e</mi><mi>x</mi></msup>"),
    "derivada": M("<mfrac><mrow><mi>d</mi><mi>f</mi></mrow><mrow><mi>d</mi><mi>x</mi></mrow></mfrac>"),
    "integral": M("<msubsup><mo>∫</mo><mn>0</mn><mn>1</mn></msubsup><mi>x</mi><mi>d</mi><mi>x</mi>"),
    "limite": M("<munder><mi>lim</mi><mrow><mi>x</mi><mo>→</mo><mn>0</mn></mrow></munder><mi>f</mi><mo>(</mo><mi>x</mi><mo>)</mo>"),
    "sumatoria": M("<munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><mi>i</mi>"),
    "matriz": M("<mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>2</mn></mtd></mtr><mtr><mtd><mn>3</mn></mtd><mtd><mn>4</mn></mtd></mtr></mtable>"),
    "raiz": M("<msqrt><mi>x</mi></msqrt>"),
    "fraccion anidada": M("<mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><mfrac><mn>1</mn><mi>x</mi></mfrac></mrow></mfrac>"),
    "desigualdad": M("<mi>a</mi><mo>≤</mo><mi>b</mi><mo>&lt;</mo><mi>c</mi>"),
}


def mc(engine=None, **cfg):
    return MathCAT(MathCATConfiguration(**cfg), engine or FakeEngine())


def test_las_expresiones_de_prueba_son_mathml_valido():
    for name, ml in EXPRESSIONS.items():
        errors, _ = validate_mathml(ml)
        assert errors == [], (name, [e.code for e in errors])


def test_todas_las_expresiones_llegan_sin_modificar_al_motor():
    for name, ml in EXPRESSIONS.items():
        engine = FakeEngine()
        out = mathml_to_braille(ml, mathcat=mc(engine))
        assert out.status is StageStatus.OK, (name, out.errors)
        assert ("set_mathml", ml) in engine.calls, name


def test_resultado_ok_registra_codigo_idioma_y_avisa_que_no_esta_verificado():
    out = mathml_to_braille(EXPRESSIONS["suma"], mathcat=mc(braille_code="cmu"))
    assert out.status is StageStatus.OK and out.text_unicode == "⠼⠁⠃"
    assert (out.braille_code, out.mathcat_code, out.language, out.mathcat_version) == ("cmu", "CMU", "es", "fake-0")
    assert "UNVERIFIED_TRANSCRIPTION" in [w.code for w in out.warnings]
    assert out.review_required is False


def test_codigo_configurable_nemeth_ueb_cmu():
    for code, expected in [("nemeth", "Nemeth"), ("ueb_technical", "UEB"), ("cmu", "CMU")]:
        engine = FakeEngine()
        out = mathml_to_braille(EXPRESSIONS["suma"], mathcat=mc(engine), braille_code=code)
        assert out.mathcat_code == expected and engine.prefs["BrailleCode"] == expected


def test_codigo_no_soportado():
    out = mathml_to_braille(EXPRESSIONS["suma"], mathcat=mc(), braille_code="brf")
    assert out.status is StageStatus.FAILED and out.errors[0].code == "UNSUPPORTED_BRAILLE_CODE"


def test_mathml_invalido_no_se_envia_a_mathcat():
    engine = FakeEngine()
    out = mathml_to_braille("<math><mi>x</math>", mathcat=mc(engine))
    assert out.status is StageStatus.FAILED and out.errors[0].code == "INVALID_MATHML"
    assert engine.calls == []


def test_salida_vacia_es_expresion_no_compatible():
    out = mathml_to_braille(EXPRESSIONS["suma"], mathcat=mc(FakeEngine(braille="  ")))
    assert out.status is StageStatus.FAILED and out.errors[0].code == "EMPTY_BRAILLE"


def test_caracteres_no_braille_exigen_revision():
    out = mathml_to_braille(EXPRESSIONS["suma"], mathcat=mc(FakeEngine(braille="⠁?⠃")))
    assert out.status is StageStatus.OK and out.review_required is True
    assert "NON_BRAILLE_CHARS" in [w.code for w in out.warnings]
    assert non_braille_chars("⠁ ⠃\n") == []


def test_la_revision_semantica_pendiente_se_propaga():
    out = mathml_to_braille(EXPRESSIONS["suma"], mathcat=mc(), semantic_review_required=True)
    assert out.review_required is True and "SEMANTIC_REVIEW_PENDING" in [w.code for w in out.warnings]


def test_fallo_de_mathcat_se_registra():
    out = mathml_to_braille(EXPRESSIONS["suma"], mathcat=mc(FakeEngine(fail_on="set_mathml")))
    assert out.status is StageStatus.FAILED and out.errors[0].code == "MATHCAT_CALL_FAILED"


def test_sin_carpeta_rules_el_fallo_es_claro():
    out = mathml_to_braille(EXPRESSIONS["suma"], mathcat=MathCAT(MathCATConfiguration()))
    assert out.status is StageStatus.FAILED and out.errors[0].code == "RULES_DIR_MISSING"


def test_revision_humana_conserva_el_original():
    out = mathml_to_braille(EXPRESSIONS["suma"], mathcat=mc())
    rev = apply_braille_review(out, "⠁⠃⠉")
    assert rev.text_unicode == "⠼⠁⠃" and rev.text == "⠁⠃⠉" and rev.reviewed_by_human


def test_exportacion_es_texto_unicode_y_rechaza_brf():
    out = mathml_to_braille(EXPRESSIONS["suma"], mathcat=mc())
    with tempfile.TemporaryDirectory() as tmp:
        target = export_braille_text(out, Path(tmp) / "salida.txt")
        assert target.read_text(encoding="utf-8").strip() == "⠼⠁⠃"
        for bad in ("salida.brf", "salida.BRL"):
            try:
                export_braille_text(out, Path(tmp) / bad)
            except ValueError as exc:
                assert "BRF" in str(exc)
            else:
                raise AssertionError(bad)
            assert not (Path(tmp) / bad).exists()


# ------------------------------------------------ integracion REAL (si esta disponible)
def test_integracion_real_las_expresiones_producen_celdas_braille():
    import pytest
    if not (os.environ.get("MATHCAT_RULES_DIR") or os.environ.get("MathCATRulesDir")):
        pytest.skip("Define MATHCAT_RULES_DIR para ejecutar las pruebas reales de MathCAT.")
    from app.mathcat import load_configuration
    real = MathCAT(load_configuration())
    try:
        real.initialize()
    except MathCATError as exc:
        pytest.skip(f"MathCAT no disponible: {exc.message}")
    report = {}
    for name, ml in EXPRESSIONS.items():
        out = mathml_to_braille(ml, mathcat=real)
        report[name] = (out.status.value, [e.code for e in out.errors])
    failures = {k: v for k, v in report.items() if v[0] != "OK"}
    assert not failures, f"MathCAT no pudo transcribir: {failures}"
