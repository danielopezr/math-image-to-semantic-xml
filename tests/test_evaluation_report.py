"""Tests for the evaluation report: fallbacks, timeouts, weighting and metrics."""

import argparse
import json

import pytest

from image_to_mathml.cli_commands import _image_dataset_defaults
from image_to_mathml.config import Settings
from image_to_mathml.evaluation import (
    evaluate_samples,
    load_evaluation_config,
    load_evaluation_samples,
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

    report = evaluate_samples(_power(tmp_path), pipeline)

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
