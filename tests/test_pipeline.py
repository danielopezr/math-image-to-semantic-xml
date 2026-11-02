"""Pruebas de integracion del pipeline con componentes falsos (sin pix2tex, red ni LLM)."""
import json
import tempfile
from pathlib import Path

from PIL import Image

from app.evaluation import levenshtein, ocr_accuracy, summarize_stages
import wave

from app.mathcat import MathCAT
from app.models import (
    MathCATConfiguration, SemanticOutput, SemanticResult, StageIssue, StageStatus,
)
from tests.test_mathcat import FakeEngine
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


def _sem(nodes, review=False):
    return SemanticOutput.model_validate({"vocabulary_version": "1.0.0", "root_id": "n0", "nodes": nodes,
                                          "review_required": review})


def good_analyzer(latex, mathml, provider):
    out = _sem([
        {"id": "n0", "node_type": "structure", "intent": "equation", "operands": [{"role": "statement", "node_id": "n1"}]},
        {"id": "n1", "node_type": "relation", "intent": "equality", "symbol": "=",
         "operands": [{"role": "left", "node_id": "n2"}, {"role": "right", "node_id": "n3"}]},
        {"id": "n2", "node_type": "element", "intent": "variable", "symbol": "x"},
        {"id": "n3", "node_type": "element", "intent": "number", "symbol": "0"}])
    return SemanticResult(status=StageStatus.OK, output=out, provider="fake", model="fake-1", attempts=1)


MATHML_POW = f'<math xmlns="{MATHML_NS}"><msup><mi>x</mi><mn>2</mn></msup></math>'


def power_analyzer(latex, mathml, provider):
    out = _sem([
        {"id": "n0", "node_type": "operation", "intent": "power",
         "operands": [{"role": "base", "node_id": "n1"}, {"role": "exponent", "node_id": "n2"}]},
        {"id": "n1", "node_type": "element", "intent": "variable", "symbol": "x"},
        {"id": "n2", "node_type": "element", "intent": "number", "symbol": "2"}])
    return SemanticResult(status=StageStatus.OK, output=out, provider="fake", model="fake-1", attempts=1)


def review_analyzer(latex, mathml, provider):
    res = good_analyzer(latex, mathml, provider)
    res.output.review_required = True
    return res


def fake_mathcat(**kw):
    return MathCAT(MathCATConfiguration(), FakeEngine(**kw))


def write_wav(path):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000); w.writeframes(b"\x00\x00" * 80)


class FakeTTS:
    name, output_format = "fake-tts", "wav"

    def __init__(self, fail=False): self.fail = fail

    def synthesize(self, text, *, language, output_path, voice_id=None, rate=None):
        if self.fail:
            from app.audio import TTSError
            raise TTSError("NO_VOICE_FOR_LANGUAGE", "sin voz")
        write_wav(output_path)
        return "v1"


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
        assert set(s) == {"ocr", "latex", "mathml", "semantic_response", "intent_structure", "semantic_coherence",
                          "catalog", "xml", "braille", "speech", "audio"}
        assert s["ocr"]["accuracy"]["exact_match"] is False
        assert s["latex"]["state"] == "accepted_by_converter"
        assert s["intent_structure"]["errors"] == [] and s["intent_structure"]["metrics"]["nodes"] == 4
        assert s["semantic_coherence"]["rules"]["SEM-003"] == "PASS"
        assert s["braille"]["generated"] is False and s["audio"]["file_generated"] is False
        assert not any("score" in k or "total" in k for k in s)


# ======================================================= accesibilidad: braille, lectura, audio
def access(tmp, **kw):
    args = dict(latex="x = 0", provider=FakeProvider(), semantic_analyzer=good_analyzer,
                mathcat=fake_mathcat(), braille=True, speech=True)
    args.update(kw)
    return run(tmp, **args)


def stage(report, name):
    return next(s for s in report.stages if s.stage == name)


def test_braille_y_lectura_completan_el_resultado_validado_sin_audio():
    with tempfile.TemporaryDirectory() as tmp:
        r = access(tmp)
        assert r.braille.status is StageStatus.OK and r.speech.status is StageStatus.OK
        assert r.audio is None and r.validated, (r.skipped, r.xml.errors)
        assert stage(r, "braille").required and stage(r, "speech").required and not stage(r, "audio").required
        assert (Path(r.output_dir) / "braille.json").is_file() and (Path(r.output_dir) / "speech.json").is_file()


def test_audio_sin_proveedor_tts_no_invalida_el_braille_generado():
    with tempfile.TemporaryDirectory() as tmp:
        r = access(tmp, audio=True)
        assert r.audio.status is StageStatus.SKIPPED and r.audio.file_name is None
        assert "audio" in r.skipped and r.braille.status is StageStatus.OK
        assert r.validated is True                      # el TTS opcional no impide validar
        assert not list(Path(r.output_dir).glob("*.wav"))   # y no se afirma ningun archivo de audio


def test_audio_generado_con_proveedor_configurado():
    with tempfile.TemporaryDirectory() as tmp:
        r = access(tmp, audio=True, tts_provider=FakeTTS())
        assert r.audio.status is StageStatus.OK and Path(r.audio.file_path).is_file()
        assert b"lectura.wav" in r.xml.xml and r.validated


def test_fallo_del_proveedor_tts_impide_validar_pero_conserva_el_resto():
    with tempfile.TemporaryDirectory() as tmp:
        r = access(tmp, audio=True, tts_provider=FakeTTS(fail=True))
        assert r.audio.status is StageStatus.FAILED and r.audio.errors[0].code == "NO_VOICE_FOR_LANGUAGE"
        assert r.braille.status is StageStatus.OK and r.speech.status is StageStatus.OK   # se conservan
        assert r.validated is False and r.xml.status is StageStatus.OK


def test_audio_solicitado_sin_lectura_se_omite():
    with tempfile.TemporaryDirectory() as tmp:
        r = access(tmp, audio=True, tts_provider=FakeTTS(), mathcat=fake_mathcat(fail_on="set_mathml"))
        assert r.speech.status is StageStatus.FAILED and r.audio is None
        assert "audio" in r.skipped


def test_braille_obligatorio_fallido_no_declara_validado():
    with tempfile.TemporaryDirectory() as tmp:
        r = access(tmp, mathcat=MathCAT(MathCATConfiguration()), speech=False)    # sin carpeta Rules
        assert r.braille.status is StageStatus.FAILED and r.braille.errors[0].code == "RULES_DIR_MISSING"
        assert r.conversion.status is StageStatus.OK and r.semantic.status is StageStatus.OK   # previos intactos
        assert r.validated is False and stage(r, "braille").required


def test_conversion_fallida_omite_las_etapas_de_accesibilidad():
    def boom(latex):
        raise KeyError("x")

    with tempfile.TemporaryDirectory() as tmp:
        r = access(tmp, converter=boom)
        assert r.braille is None and r.speech is None and r.audio is None
        assert {"braille", "speech", "semantic_agent", "rules"} <= set(r.skipped)
        assert r.validated is False and stage(r, "braille").required


def test_un_fallo_semantico_no_impide_el_braille_pero_si_validar():
    with tempfile.TemporaryDirectory() as tmp:
        r = access(tmp, semantic_analyzer=failing_analyzer)
        assert r.semantic.status is StageStatus.FAILED and r.braille.status is StageStatus.OK
        assert r.validated is False


def test_la_revision_semantica_pendiente_se_propaga_al_braille_y_la_lectura():
    with tempfile.TemporaryDirectory() as tmp:
        r = access(tmp, semantic_analyzer=review_analyzer)
        assert r.braille.review_required and r.speech.review_required
        assert stage(r, "semantic_agent").review_required and r.validated   # revisar no es fallar


def test_revision_humana_del_braille():
    with tempfile.TemporaryDirectory() as tmp:
        r = access(tmp, braille_reviewer=lambda out: "⠁⠃")
        assert r.braille.text_unicode == "⠼⠁⠃" and r.braille.text == "⠁⠃" and r.braille.reviewed_by_human


def test_sin_braille_ni_lectura_solicitados_no_se_toca_mathcat():
    engine = FakeEngine()
    with tempfile.TemporaryDirectory() as tmp:
        r = run(tmp, latex="x = 0", provider=FakeProvider(), semantic_analyzer=good_analyzer,
                mathcat=MathCAT(MathCATConfiguration(), engine))
        assert engine.calls == [] and r.braille is None and r.speech is None and r.validated


def test_enriquecimiento_usa_mathml_enriquecido_solo_en_la_lectura():
    engine = FakeEngine()
    with tempfile.TemporaryDirectory() as tmp:
        r = run(tmp, latex="x^2", converter=lambda s: MATHML_POW, provider=FakeProvider(),
                semantic_analyzer=power_analyzer, mathcat=MathCAT(MathCATConfiguration(), engine),
                braille=True, speech=True, enrich_intents=True)
        assert r.enrichment.status is StageStatus.OK and r.enrichment.mathcat_checked
        sent = [c[1] for c in engine.calls if c[0] == "set_mathml"]
        assert "intent=" not in [s for s in sent if s == MATHML_POW][0]            # braille: MathML original
        assert any('intent="power($b,$e)"' in s for s in sent)                      # lectura: enriquecido
        assert "ENRICHED_MATHML_USED" in [w.code for w in r.speech.warnings]
        assert r.conversion.mathml == MATHML_POW                                    # el original no se altera
        assert r.validated


def test_enriquecimiento_desactivado_por_defecto():
    with tempfile.TemporaryDirectory() as tmp:
        r = access(tmp)
        assert r.enrichment is None and "enrichment" in r.skipped


def test_resumen_de_etapas_sin_puntuacion_global():
    with tempfile.TemporaryDirectory() as tmp:
        r = access(tmp, audio=True, tts_provider=FakeTTS())
        names = [s.stage for s in r.stages]
        assert names == ["ocr", "converter", "semantic_agent", "rules", "enrichment", "braille", "speech", "audio", "xml"]
        assert all(isinstance(s.status, StageStatus) for s in r.stages)
        assert not hasattr(r, "score") and not hasattr(r, "total_score")


def test_entrada_mathml_directa_omite_ocr_y_conversion_y_nunca_valida():
    with tempfile.TemporaryDirectory() as tmp:
        r = run_pipeline(output_dir=tmp, mathml=MATHML_EQ, mathcat=fake_mathcat(), braille=True, speech=True)
        assert r.ocr is None and r.conversion.converter == "provided" and r.conversion.status is StageStatus.OK
        assert r.braille.status is StageStatus.OK and r.speech.status is StageStatus.OK
        assert r.xml.status is StageStatus.OK, r.xml.errors
        assert r.validated is False and {"ocr", "converter"} <= set(r.skipped)


def test_mathml_directo_invalido_se_rechaza_antes_de_mathcat():
    engine = FakeEngine()
    with tempfile.TemporaryDirectory() as tmp:
        r = run_pipeline(output_dir=tmp, mathml="<math><mi>x</math>", mathcat=MathCAT(MathCATConfiguration(), engine),
                         braille=True)
        assert r.conversion.status is StageStatus.FAILED and r.braille is None and engine.calls == []
        assert "braille" in r.skipped and r.validated is False


def test_evaluacion_de_braille_lectura_y_audio_por_separado():
    with tempfile.TemporaryDirectory() as tmp:
        r = access(tmp, audio=True, tts_provider=FakeTTS())
        s = summarize_stages(r)
        assert s["braille"]["generated"] and s["braille"]["non_braille_chars"] == []
        assert s["speech"]["generated"] and s["audio"]["file_generated"] and s["audio"]["format"] == "wav"
        r2 = access(tmp, audio=True)          # sin TTS: texto si, audio no
        s2 = summarize_stages(r2)
        assert s2["speech"]["generated"] and s2["audio"]["file_generated"] is False and s2["audio"]["status"] == "SKIPPED"
