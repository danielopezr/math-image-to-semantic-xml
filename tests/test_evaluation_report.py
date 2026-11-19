"""Tests for the evaluation report: fallbacks, timeouts, weighting and metrics."""

import argparse
import csv
import io
import json

import pytest

from image_to_mathml.cli_commands import _image_dataset_defaults
from image_to_mathml.config import Settings
from image_to_mathml.evaluation import (
    SampleEvaluation,
    evaluate_samples,
    load_evaluation_config,
    load_evaluation_samples,
    report_csv,
    report_markdown,
)
from image_to_mathml.models import Flow, PipelineResult, ValidationReport
from image_to_mathml.pipeline import EquationPipeline

_REFERENCE = (
    '<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>'
    "<msup><mi>x</mi><mn>2</mn></msup>"
    "</mrow></math>"
)


def _write_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def _result(mathml, warnings=None, llm_used=False):
    return PipelineResult(
        latex="",
        mathml=mathml,
        flow=Flow.HYBRID,
        validation=ValidationReport(),
        warnings=warnings or [],
        llm_used=llm_used,
    )


class FakeRunPipeline:
    settings = Settings()

    def __init__(self, responses):
        self.responses = list(responses)
        self.timeouts = []

    def run(self, request):
        self.timeouts.append(request.timeout)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _NoLLM:
    model = "none"
    base_url = "http://none"

    def is_available(self):
        return False

    def has_model(self):
        return False

    def list_models(self):
        return []

    def chat(self, messages, **kwargs):
        raise AssertionError("not called")


def _samples(tmp_path, records):
    dataset = tmp_path / "samples.jsonl"
    _write_jsonl(dataset, records)
    return load_evaluation_samples(dataset)


def _power(tmp_path, **extra):
    record = {"id": "power", "latex": "x^{2}", "reference_mathml": _REFERENCE}
    record.update(extra)
    return _samples(tmp_path, [record])


def test_fallback_output_is_flagged_and_excluded_from_llm_scores(tmp_path):
    pipeline = EquationPipeline(Settings(), llm=_NoLLM())

    report = evaluate_samples(_power(tmp_path), pipeline, default_flow=Flow.HYBRID)

    sample = report.samples[0]
    assert sample.ok
    assert sample.fallback
    assert sample.llm_used is False
    assert report.llm_samples == []
    assert report.llm_structure_bleu is None
    markdown = report_markdown(report)
    assert "0 con LLM, 1 con respaldo determinista" in markdown
    assert "| respaldo |" in markdown


def test_llm_samples_get_their_own_corpus_score(tmp_path):
    pipeline = FakeRunPipeline([_result(_REFERENCE, llm_used=True)])

    report = evaluate_samples(_power(tmp_path), pipeline)

    assert len(report.llm_samples) == 1
    assert report.llm_structure_bleu.score == 1.0


def test_exhausted_timeouts_are_visible_in_the_report(tmp_path):
    warning = ["The local LLM failed: Read timed out"]
    pipeline = FakeRunPipeline([_result(_REFERENCE, warnings=warning)] * 3)

    report = evaluate_samples(
        _power(tmp_path, timeout=300),
        pipeline,
        timeout_increment=60,
        max_timeout_retries=2,
        default_flow=Flow.HYBRID,
    )

    sample = report.samples[0]
    assert sample.timed_out
    assert sample.fallback
    assert sample.timeout == 420.0
    assert "agotado (2 reintentos)" in report_markdown(report)
    assert report.as_dict()["samples_timed_out"] == 1


def test_failed_samples_count_as_empty_outputs_in_the_corpus(tmp_path):
    samples = _samples(
        tmp_path,
        [
            {"id": "good", "latex": "x^{2}", "reference_mathml": _REFERENCE},
            {"id": "bad", "latex": "x^{2}", "reference_mathml": _REFERENCE},
        ],
    )
    pipeline = FakeRunPipeline([_result(_REFERENCE), RuntimeError("boom")])

    report = evaluate_samples(samples, pipeline)

    assert len(report.failed_samples) == 1
    corpus = report.structure_bleu
    assert corpus.reference_length == 2 * corpus.candidate_length
    assert corpus.score < 1.0


def test_weighting_is_per_equation(tmp_path):
    different = (
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>'
        "<msqrt><mi>y</mi></msqrt></mrow></math>"
    )
    records = [
        {
            "id": f"easy{index}",
            "latex": "x^{2}",
            "reference_mathml": _REFERENCE,
            "eqn_complexity": "simple",
        }
        for index in range(3)
    ]
    records.append(
        {
            "id": "hard",
            "latex": "x^{2}",
            "reference_mathml": _REFERENCE,
            "eqn_complexity": "media",
        }
    )
    pipeline = FakeRunPipeline([_result(_REFERENCE)] * 3 + [_result(different)])

    report = evaluate_samples(
        _samples(tmp_path, records),
        pipeline,
        complexity_weights={"simple": 1.0, "media": 1.5},
    )

    simple = report.by_complexity["simple"].structure_bleu.score
    media = report.by_complexity["media"].structure_bleu.score
    expected = (1.0 * 3 * simple + 1.5 * 1 * media) / (1.0 * 3 + 1.5 * 1)
    assert report.weighted_structure_bleu == pytest.approx(expected)


def test_classes_without_a_weight_are_reported(tmp_path):
    pipeline = FakeRunPipeline([_result(_REFERENCE)])
    report = evaluate_samples(
        _power(tmp_path, eqn_complexity="dificil"),
        pipeline,
        complexity_weights={"simple": 1.0},
    )
    assert any("dificil" in note for note in report.notes)


def test_reference_latex_matches_identical_offline_output(tmp_path):
    cases = [
        r"\frac{a+b}{c}",
        r"x^{2}+{y}",
        r"\begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}",
    ]
    samples = _samples(
        tmp_path,
        [
            {
                "id": str(index),
                "latex": latex,
                "reference_latex": latex,
                "flow": "offline",
            }
            for index, latex in enumerate(cases)
        ],
    )

    report = evaluate_samples(samples, EquationPipeline(Settings(), llm=_NoLLM()))

    assert [sample.structure.score for sample in report.samples] == [1.0, 1.0, 1.0]
    assert all(sample.exact_structure for sample in report.samples)
    assert not any(sample.fallback for sample in report.samples)
    assert any("reference_latex" in note for note in report.notes)


def test_intent_f1_and_exact_match_are_reported(tmp_path):
    candidate = (
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>'
        '<msup intent="power($b,$e)"><mi arg="b">x</mi><mn arg="e">2</mn></msup>'
        "</mrow></math>"
    )
    reference = (
        candidate.replace("$b,$e", "$base,$exp")
        .replace('arg="b"', 'arg="base"')
        .replace('arg="e"', 'arg="exp"')
    )
    samples = _samples(
        tmp_path, [{"id": "p", "latex": "x^{2}", "reference_mathml": reference}]
    )

    report = evaluate_samples(samples, FakeRunPipeline([_result(candidate)]))

    sample = report.samples[0]
    assert sample.intent_f1.f1 == 1.0
    assert sample.exact_intent
    assert report.as_dict()["rates"]["mean_intent_f1"] == 1.0


def test_yaml_paths_expand_environment_variables_and_overrides(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "imgs").mkdir(parents=True)
    (data / "labels").mkdir()
    (data / "labels" / "power.mathml").write_text(_REFERENCE, encoding="utf-8")
    (tmp_path / "dataset.csv").write_text(
        "id,img_path,mathml_path\npower,power.png,power.mathml\n", encoding="utf-8"
    )
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "csv: dataset.csv\n"
        "images_dir: ${MY_DATA:-/nowhere}/imgs\n"
        "mathml_dir: ${MY_DATA:-/nowhere}/labels\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("MY_DATA", str(data))
    assert load_evaluation_config(config_path).image_dir == (data / "imgs").resolve()

    monkeypatch.delenv("MY_DATA")
    assert "nowhere" in str(load_evaluation_config(config_path).image_dir)

    overridden = load_evaluation_config(
        config_path, images_dir=data / "imgs", mathml_dir=data / "labels"
    )
    samples = load_evaluation_samples(config_path, config=overridden)
    assert samples[0].reference_mathml == _REFERENCE


def test_invalid_csv_timeout_names_the_sample(tmp_path):
    (tmp_path / "power.mathml").write_text(_REFERENCE, encoding="utf-8")
    dataset = tmp_path / "samples.csv"
    dataset.write_text(
        "id,img_path,mathml_path,timeout\npower,p.png,power.mathml,abc\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="'power'.*invalid timeout"):
        load_evaluation_samples(dataset)


def test_image_dataset_defaults_respect_environment(monkeypatch):
    args = argparse.Namespace(ocr=None, model=None, vision_model=None)
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")
    monkeypatch.delenv("IMAGE_TO_MATHML_OCR", raising=False)
    monkeypatch.delenv("IMAGE_TO_MATHML_OLLAMA_VISION_MODEL", raising=False)

    defaults = _image_dataset_defaults(args)

    assert "ollama_model" not in defaults
    assert defaults["ocr_backend"] == "ollama_vision"
    assert defaults["ollama_vision_model"] == "qwen3-vl:4b-instruct"


def test_report_csv_mirrors_the_sample_table_and_adds_timings(tmp_path):
    report = evaluate_samples(
        _power(tmp_path, eqn_complexity="simple"),
        FakeRunPipeline([_result(_REFERENCE)]),
    )
    sample = report.samples[0]
    sample.elapsed_seconds = 2.5
    sample.attempts_total = 2

    rows = list(csv.reader(io.StringIO(report_csv(report))))

    assert rows[0] == [
        "muestra",
        "complejidad",
        "bleu_estructura",
        "bleu_intent",
        "f1_intent",
        "exacta_estructura",
        "exacta_intent",
        "exacta_flexible_estructura",
        "exacta_flexible_intent",
        "llm",
        "validacion",
        "timeout",
        "error",
        "tiempo_total_s",
        "tiempo_por_intento_s",
        "tiempo_vision_s",
        "tiempo_llm_s",
        "tiempo_otros_s",
        "carga_modelo_s",
        "lectura_prompt_s",
        "generacion_s",
        "tokens_generados",
        "llamadas_cortadas",
    ]
    assert rows[1][0] == "power"
    assert rows[1][1] == "simple"
    assert rows[1][2] == "1.000000"
    assert rows[1][5:9] == ["sí", "sí", "sí", "sí"]
    assert rows[1][13:15] == ["2.500", "1.250"]


def test_lenient_exact_match_is_reported_in_samples_rates_and_csv(tmp_path):
    reference = (
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>'
        "<mi>a</mi><mo>+</mo><mi>b</mi></mrow></math>"
    )
    candidate = (
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>'
        "<mrow><mi>a</mi><mo>+</mo></mrow><mi>b</mi><mo>.</mo></mrow></math>"
    )
    samples = _samples(
        tmp_path, [{"id": "s", "latex": "a+b", "reference_mathml": reference}]
    )

    report = evaluate_samples(samples, FakeRunPipeline([_result(candidate)]))

    sample = report.samples[0]
    assert not sample.exact_structure
    assert sample.lenient_exact_structure
    assert sample.lenient_exact_intent
    rates = report.as_dict()["rates"]
    assert rates["exact_structure"] == 0.0
    assert rates["lenient_exact_structure"] == 1.0
    assert report.as_dict()["samples"][0]["lenient_exact_structure"] is True
    rows = list(csv.reader(io.StringIO(report_csv(report))))
    assert rows[1][5:9] == ["no", "no", "sí", "sí"]
    assert "coincidencia exacta flexible de estructura" in report_markdown(report)


def test_seconds_per_attempt_divides_the_total_by_the_attempts_made():
    sample = SampleEvaluation(sample_id="x", input="")
    sample.elapsed_seconds = 3.0
    sample.attempts_total = 3

    assert sample.seconds_per_attempt == 1.0
    assert sample.as_dict()["elapsed_seconds"] == 3.0
    assert sample.as_dict()["seconds_per_attempt"] == 1.0


def test_timeout_retries_are_counted_as_timing_attempts(tmp_path):
    timeout = ["The local LLM failed: Read timed out"]
    pipeline = FakeRunPipeline(
        [
            _result(_REFERENCE, warnings=timeout),
            _result(_REFERENCE, warnings=timeout),
            _result(_REFERENCE),
        ]
    )

    report = evaluate_samples(
        _power(tmp_path, timeout=300),
        pipeline,
        timeout_increment=60,
        max_timeout_retries=2,
    )

    assert report.samples[0].timeout_retries == 2
    assert report.samples[0].attempts_total == 3
    assert report.samples[0].elapsed_seconds >= 0.0


def test_on_sample_reports_every_sample_as_it_finishes(tmp_path):
    samples = _samples(
        tmp_path,
        [
            {"id": "bad", "latex": "x", "reference_mathml": "<math><mrow>"},
            {"id": "power", "latex": "x^{2}", "reference_mathml": _REFERENCE},
        ],
    )
    seen = []

    evaluate_samples(
        samples,
        FakeRunPipeline([_result(_REFERENCE)]),
        on_sample=lambda sample, index, total: seen.append(
            (sample.sample_id, index, total, sample.ok)
        ),
    )

    assert seen == [("bad", 1, 2, False), ("power", 2, 2, True)]


def test_completed_samples_are_rescored_without_running_the_pipeline(tmp_path):
    saved = {
        "sample_id": "power",
        "output_mathml": _REFERENCE,
        "validation_ok": True,
        "llm_used": True,
        "fallback": False,
        "attempts": 1,
        "attempts_total": 2,
        "timeout": 360.0,
        "timeout_retries": 1,
        "timed_out": False,
        "elapsed_seconds": 12.5,
        "stage_times": {"vision_seconds": 4.0, "llm_seconds": 7.5, "calls": 2},
        "warnings": ["w"],
        "error": "Invalid output MathML: stale",
    }
    pipeline = FakeRunPipeline([])

    report = evaluate_samples(_power(tmp_path), pipeline, completed={"power": saved})

    sample = report.samples[0]
    assert pipeline.timeouts == []  # the pipeline was never called
    assert sample.resumed
    assert sample.ok and sample.error == ""
    assert sample.exact_structure
    assert (sample.elapsed_seconds, sample.timeout_retries) == (12.5, 1)
    assert (sample.stages.vision_seconds, sample.stages.llm_seconds) == (4.0, 7.5)
    assert sample.stages.calls == 2
    assert report.as_dict()["samples"][0]["structure_bleu"]["score"] == 1.0


def test_cli_evaluation_saves_partial_results_and_resumes(tmp_path, monkeypatch):
    from image_to_mathml import cli_commands
    from image_to_mathml.cli import main

    dataset = tmp_path / "samples.jsonl"
    _write_jsonl(
        dataset,
        [
            {"id": "a", "latex": "x^{2}", "reference_mathml": _REFERENCE},
            {"id": "b", "latex": "x^{2}", "reference_mathml": _REFERENCE},
        ],
    )
    output = tmp_path / "report.json"
    partial = tmp_path / "report.partial.jsonl"
    args = ["evaluate", "--dataset", str(dataset), "--output", str(output), "--json"]

    class Interrupted(FakeRunPipeline):
        def run(self, request):
            if self.timeouts:  # the second sample: the user stops the run
                raise KeyboardInterrupt
            return super().run(request)

    interrupted = Interrupted([_result(_REFERENCE)])
    monkeypatch.setattr(cli_commands, "EquationPipeline", lambda settings: interrupted)
    with pytest.raises(KeyboardInterrupt):
        main(args)
    saved = [
        json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines()
    ]
    assert [entry["sample_id"] for entry in saved] == ["a"]
    assert not output.exists()

    # Without --resume an existing partial file is never overwritten.
    assert main(args) == 1

    resumed = FakeRunPipeline([_result(_REFERENCE)])
    monkeypatch.setattr(cli_commands, "EquationPipeline", lambda settings: resumed)
    assert main([*args, "--resume"]) == 0

    assert len(resumed.timeouts) == 1  # only sample "b" ran again
    report = json.loads(output.read_text(encoding="utf-8"))
    assert [sample["sample_id"] for sample in report["samples"]] == ["a", "b"]
    assert not partial.exists()


class StagedPipeline(FakeRunPipeline):
    """Make one vision and one LLM call per run, the first run timing out."""

    def run(self, request):
        from image_to_mathml.llm.base import apply_ollama_stats, track_model_call

        with track_model_call("vision", "vl") as call:
            apply_ollama_stats(
                call,
                {
                    "load_duration": 2_000_000_000,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_duration": 4_000_000_000,
                    "eval_count": 20,
                },
            )
        with track_model_call("llm", "text"):
            return super().run(request)


def test_time_per_stage_adds_every_model_call_including_timeouts(tmp_path):
    pipeline = StagedPipeline(
        [RuntimeError("Read timed out. (read timeout=600)"), _result(_REFERENCE)]
    )

    report = evaluate_samples(_power(tmp_path), pipeline, max_timeout_retries=1)

    stages = report.samples[0].stages
    assert stages.calls == 4
    assert stages.cut_calls == 2  # the two LLM calls carried no Ollama stats
    assert stages.load_seconds == pytest.approx(4.0)
    assert stages.generate_seconds == pytest.approx(8.0)
    assert stages.output_tokens == 40
    assert stages.tokens_per_second == pytest.approx(5.0)
    total = stages.vision_seconds + stages.llm_seconds + stages.other_seconds
    assert total == pytest.approx(report.samples[0].elapsed_seconds)

    saved = report.as_dict()
    assert saved["samples"][0]["stage_times"]["output_tokens"] == 40
    assert saved["stage_times"]["calls"] == 4
    markdown = report_markdown(report)
    assert "## Tiempo por etapa" in markdown
    assert "carga del modelo en memoria" in markdown
    assert "2 llamadas cortadas por timeout" in markdown


class CutLLMPipeline(FakeRunPipeline):
    """The first run's LLM call is cut after writing part of its answer."""

    def run(self, request):
        from image_to_mathml.llm.base import track_model_call

        with track_model_call("llm", "text") as call:
            if not self.timeouts:
                call.partial_output = "times(times(times("
            return super().run(request)


def test_cut_answers_and_lowest_free_memory_are_kept_per_sample(tmp_path, monkeypatch):
    readings = iter([900.0, 640.0])
    monkeypatch.setattr(
        "image_to_mathml.llm.base.free_memory_mb", lambda: next(readings)
    )
    pipeline = CutLLMPipeline(
        [RuntimeError("Read timed out. (read timeout=600)"), _result(_REFERENCE)]
    )

    report = evaluate_samples(_power(tmp_path), pipeline, max_timeout_retries=1)

    sample = report.samples[0]
    assert sample.cut_outputs == ["llm: times(times(times("]
    assert sample.stages.min_free_memory_mb == 640.0
    saved = sample.as_dict()
    assert saved["cut_outputs"] == ["llm: times(times(times("]
    assert saved["stage_times"]["min_free_memory_mb"] == 640
    assert report.stage_totals.min_free_memory_mb == 640.0
    assert "RAM libre mínima al terminar una llamada: 640 MB" in report_markdown(report)


def test_a_sample_without_cut_calls_saves_no_cut_outputs(tmp_path):
    report = evaluate_samples(_power(tmp_path), FakeRunPipeline([_result(_REFERENCE)]))

    assert "cut_outputs" not in report.samples[0].as_dict()


def test_ocr_latex_and_raw_llm_answers_are_saved_and_restored(tmp_path):
    from image_to_mathml.evaluation import _restore_sample
    from image_to_mathml.models import LLMOutput

    result = _result(_REFERENCE, llm_used=True)
    result.latex = "x^{2}"
    result.llm_outputs = [
        LLMOutput(
            flow=Flow.HYBRID,
            attempt=1,
            mathml='{"annotations": []}',
            accepted=True,
            input_mathml="<math/>",
        )
    ]

    report = evaluate_samples(_power(tmp_path), FakeRunPipeline([result]))

    saved = report.samples[0].as_dict()
    assert saved["output_latex"] == "x^{2}"
    assert saved["llm_outputs"] == [
        {
            "flow": "hybrid",
            "attempt": 1,
            "accepted": True,
            "input_mathml": "<math/>",
            "mathml": '{"annotations": []}',
        }
    ]
    restored = SampleEvaluation(sample_id="a", input="")
    _restore_sample(restored, saved)
    assert restored.output_latex == "x^{2}"
    assert restored.llm_outputs == saved["llm_outputs"]
