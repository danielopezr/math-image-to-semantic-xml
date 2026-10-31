"""Pruebas de integracion del pipeline con componentes falsos (sin pix2tex, red ni LLM)."""
import json
import tempfile
from pathlib import Path

from PIL import Image

from app.evaluation import levenshtein, ocr_accuracy, summarize_stages
from app.models import (
    SemanticNode, SemanticOutput, SemanticRelation, SemanticResult, StageIssue, StageStatus,
)
from app.pipeline import RETRY, run_pipeline
from app.xml_validator import MATHML_NS

MATHML_EQ = f'<math xmlns="{MATHML_NS}"><mi>x</mi><mo>=</mo><mn>0</mn></math>'


def fake_converter(latex):
    return MATHML_EQ


def make_image(folder: Path) -> Path:
    p = folder / "formula.png"
    Image.new("RGB", (40, 20), "white").save(p)
    return p


class FakeProvider:
    name, model = "fake", "fake-1"


def good_analyzer(latex, mathml, provider):
    out = SemanticOutput(
        expression_type="equation",
        root=SemanticNode(id="n0", kind="equation", symbol="=", children=[
            SemanticNode(id="n1", kind="variable", symbol="x"),
            SemanticNode(id="n2", kind="number", symbol="0"),
        ]),
        relations=[SemanticRelation(type="equality", left="n1", right="n2")],
        requires_review=False,
    )
    return SemanticResult(status=StageStatus.OK, output=out, provider="fake", model="fake-1", attempts=1)


def failing_analyzer(latex, mathml, provider):
    return SemanticResult(
        status=StageStatus.FAILED, provider="fake", model="fake-1", attempts=2,
        errors=[StageIssue(stage="semantic_agent", code="SCHEMA_VALIDATION_FAILED", message="no cumple")],
    )


def run(tmp, **kw):
    args = dict(output_dir=tmp, converter=fake_converter)
    args.update(kw)
    return run_pipeline(**args)


def test_flujo_completo_valido_y_guarda_intermedios():
    with tempfile.TemporaryDirectory() as tmp:
        img = make_image(Path(tmp))
        r = run(tmp, image_path=img, ocr_engine=lambda im: "x = 0",
                provider=FakeProvider(), semantic_analyzer=good_analyzer)
        assert r.validated, r.skipped
        assert r.xml.schema_valid and r.rules_status is StageStatus.OK and r.rules
        out = Path(r.output_dir)
        for name in ("ocr.json", "conversion.json", "semantic.json", "rules.json",
                     "result.xml", "report.json", "pipeline.log"):
            assert (out / name).is_file(), name
        assert json.loads((out / "ocr.json").read_text(encoding="utf-8"))["latex_raw"] == "x = 0"


def test_sin_llm_funciona_pero_no_aparenta_validado():
    with tempfile.TemporaryDirectory() as tmp:
        r = run(tmp, image_path=make_image(Path(tmp)), ocr_engine=lambda im: "x = 0")
        assert r.conversion.status is StageStatus.OK
        assert r.semantic is None and "semantic_agent" in r.skipped
        assert r.xml.status is StageStatus.OK       # estructuralmente valido
        assert r.validated is False                 # pero no validado
        assert (Path(r.output_dir) / "result.xml").is_file()


def test_ocr_fallido_omite_etapas_dependientes():
    with tempfile.TemporaryDirectory() as tmp:
        r = run(tmp, image_path=Path(tmp) / "no_existe.png", ocr_engine=lambda im: "x")
        assert r.ocr.status is StageStatus.FAILED
        assert r.conversion is None and "converter" in r.skipped
        assert r.rules == [] and "rules" in r.skipped
        assert r.validated is False and r.xml.status is StageStatus.OK


def test_entrada_manual_sin_imagen():
    with tempfile.TemporaryDirectory() as tmp:
        r = run(tmp, latex="x = 0", provider=FakeProvider(), semantic_analyzer=good_analyzer)
        assert r.ocr.engine == "manual" and r.ocr.reviewed_by_human
        assert r.ocr.latex_raw is None and r.ocr.latex == "x = 0"
        assert r.conversion.status is StageStatus.OK
        assert r.xml.status is StageStatus.OK, r.xml.errors  # attempt=0 debe ser valido en el XSD
        assert r.validated


def test_conversion_fallida_omite_llm_y_reglas():
    def boom(latex):
        raise KeyError("x")

    with tempfile.TemporaryDirectory() as tmp:
        r = run(tmp, latex="\\frac{1}", converter=boom, provider=FakeProvider(), semantic_analyzer=good_analyzer)
        assert r.conversion.status is StageStatus.FAILED
        assert r.semantic is None and "semantic_agent" in r.skipped and "rules" in r.skipped
        assert r.validated is False
        assert r.xml.status is StageStatus.OK, r.xml.errors


def test_respuesta_llm_invalida_no_valida():
    with tempfile.TemporaryDirectory() as tmp:
        r = run(tmp, latex="x = 0", provider=FakeProvider(), semantic_analyzer=failing_analyzer)
        assert r.semantic.status is StageStatus.FAILED
        assert r.validated is False
        assert b"SCHEMA_VALIDATION_FAILED" in r.xml.xml


def test_revision_manual_conserva_el_original():
    with tempfile.TemporaryDirectory() as tmp:
        r = run(tmp, image_path=make_image(Path(tmp)), ocr_engine=lambda im: "x = O",
                reviewer=lambda ocr: "x = 0")
        assert r.ocr.latex_raw == "x = O" and r.ocr.latex == "x = 0"
        assert r.ocr.reviewed_by_human


def test_repetir_ocr_y_luego_aceptar():
    outputs = iter(["a", "b"])
    calls = []

    def reviewer(ocr):
        calls.append(ocr.attempt)
        return RETRY if ocr.attempt == 1 else None

    with tempfile.TemporaryDirectory() as tmp:
        r = run(tmp, image_path=make_image(Path(tmp)), ocr_engine=lambda im: next(outputs), reviewer=reviewer)
        assert calls == [1, 2]
        assert r.ocr.attempt == 2 and r.ocr.latex_raw == "b"


def test_limite_de_reintentos():
    def reviewer(ocr):
        return RETRY

    with tempfile.TemporaryDirectory() as tmp:
        r = run(tmp, image_path=make_image(Path(tmp)), ocr_engine=lambda im: "z",
                reviewer=reviewer, max_ocr_attempts=3)
        assert r.ocr.attempt == 3


def test_requiere_imagen_o_latex():
    try:
        run_pipeline(output_dir="x")
    except ValueError:
        return
    raise AssertionError("debio lanzar ValueError")


# ------------------------------------------------------------------- evaluacion
def test_levenshtein_y_exactitud_ocr():
    assert levenshtein("kitten", "sitting") == 3
    assert ocr_accuracy("x = 0", "x=0")["exact_match"] is True
    acc = ocr_accuracy("x=O", "x=0")
    assert acc["exact_match"] is False and acc["edit_distance"] == 1
    assert ocr_accuracy(None, "x")["exact_match"] is False


def test_resumen_por_etapa_sin_puntuacion_agregada():
    with tempfile.TemporaryDirectory() as tmp:
        r = run(tmp, image_path=make_image(Path(tmp)), ocr_engine=lambda im: "x = O",
                provider=FakeProvider(), semantic_analyzer=good_analyzer)
        s = summarize_stages(r, expected_latex="x = 0")
        assert set(s) == {"ocr", "conversion", "semantic_structure", "semantic_quality", "rules", "xml"}
        assert s["ocr"]["accuracy"]["exact_match"] is False
        assert s["semantic_quality"]["measured"] is False
        assert not any("score" in k or "total" in k for k in s)
