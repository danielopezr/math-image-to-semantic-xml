import json

import pytest

from image_to_mathml.config import Settings
from image_to_mathml.evaluation import (
    evaluate_samples,
    load_evaluation_config,
    load_evaluation_samples,
    report_markdown,
)
from image_to_mathml.models import (
    EquationRequest,
    Flow,
    PipelineResult,
    ValidationReport,
)
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


def test_load_evaluation_samples_from_jsonl(tmp_path):
    dataset = tmp_path / "samples.jsonl"
    _write_jsonl(
        dataset,
        [
            {
                "id": "power",
                "latex": "x^{2}",
                "reference_mathml": _REFERENCE,
                "area": "álgebra",
                "topic": "potencias",
            }
        ],
    )

    samples = load_evaluation_samples(dataset)

    assert len(samples) == 1
    assert samples[0].sample_id == "power"
    assert samples[0].latex == "x^{2}"
    assert samples[0].context.area == "álgebra"
    assert samples[0].context.topic == "potencias"


def test_load_evaluation_samples_from_json_array_with_reference_latex(tmp_path):
    dataset = tmp_path / "samples.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "fraction",
                    "latex": r"\frac{a}{b}",
                    "reference_latex": r"\frac{a}{b}",
                }
            ]
        ),
        encoding="utf-8",
    )

    samples = load_evaluation_samples(dataset)

    assert len(samples) == 1
    assert "<math" in samples[0].reference_mathml


def test_load_evaluation_samples_can_reference_a_mathml_file(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "power.mathml").write_text(_REFERENCE, encoding="utf-8")
    dataset = tmp_path / "samples.jsonl"
    _write_jsonl(
        dataset,
        [
            {
                "id": "power",
                "latex": "x^{2}",
                "reference_mathml_file": "refs/power.mathml",
            }
        ],
    )

    samples = load_evaluation_samples(dataset)

    assert samples[0].reference_mathml == _REFERENCE


def test_load_evaluation_samples_reports_a_missing_reference_file(tmp_path):
    dataset = tmp_path / "samples.jsonl"
    _write_jsonl(
        dataset,
        [
            {
                "id": "power",
                "latex": "x^{2}",
                "reference_mathml_file": "refs/missing.mathml",
            }
        ],
    )

    with pytest.raises(ValueError, match="Could not read reference MathML file"):
        load_evaluation_samples(dataset)


def test_load_evaluation_samples_from_csv(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "power.mathml").write_text(_REFERENCE, encoding="utf-8")
    dataset = tmp_path / "samples.csv"
    dataset.write_text(
        "id,img_path,mathml_path,area,topic,timeout,eqn_complexity\n"
        "power,images/power.png,refs/power.mathml,álgebra,potencias,300,muy simple\n",
        encoding="utf-8",
    )

    samples = load_evaluation_samples(dataset)

    assert len(samples) == 1
    assert samples[0].sample_id == "power"
    assert samples[0].image_path == str(tmp_path / "images" / "power.png")
    assert samples[0].reference_mathml == _REFERENCE
    assert samples[0].context.area == "álgebra"
    assert samples[0].context.topic == "potencias"
    assert samples[0].timeout == 300.0
    assert samples[0].eqn_complexity == "muy simple"


def test_load_evaluation_samples_from_yaml_config(tmp_path):
    images = tmp_path / "images"
    refs = tmp_path / "labels"
    images.mkdir()
    refs.mkdir()
    (refs / "power.mathml").write_text(_REFERENCE, encoding="utf-8")
    (tmp_path / "dataset.csv").write_text(
        "id,img_path,mathml_path,area,topic,timeout,eqn_complexity\n"
        "power,power.png,power.mathml,álgebra,potencias,300,media\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "data_eval.yaml"
    config_path.write_text(
        "csv: dataset.csv\n"
        "images_dir: images\n"
        "mathml_dir: labels\n"
        "complexity_weights:\n"
        "  media: 2.0\n",
        encoding="utf-8",
    )

    config = load_evaluation_config(config_path)
    samples = load_evaluation_samples(config_path)

    assert config.csv_path == tmp_path / "dataset.csv"
    assert config.image_dir == images
    assert config.mathml_dir == refs
    assert config.complexity_weights == {"media": 2.0}
    assert samples[0].image_path == str(images / "power.png")
    assert samples[0].reference_mathml == _REFERENCE
    assert samples[0].eqn_complexity == "media"


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


def _result(mathml: str, warnings=None) -> PipelineResult:
    return PipelineResult(
        latex="",
        mathml=mathml,
        flow=Flow.HYBRID,
        validation=ValidationReport(),
        warnings=warnings or [],
    )


def _timeout_sample(tmp_path):
    dataset = tmp_path / "samples.jsonl"
    _write_jsonl(
        dataset,
        [
            {
                "id": "power",
                "latex": "x^{2}",
                "reference_mathml": _REFERENCE,
                "timeout": 300,
            }
        ],
    )
    return load_evaluation_samples(dataset)


def test_evaluate_samples_retries_after_a_timeout_exception(tmp_path):
    pipeline = FakeRunPipeline(
        [
            RuntimeError("Read timed out"),
            _result(_REFERENCE),
        ]
    )

    report = evaluate_samples(
        _timeout_sample(tmp_path),
        pipeline,
        timeout_increment=60,
        max_timeout_retries=2,
    )

    assert pipeline.timeouts == [300.0, 360.0]
    assert report.samples[0].ok
    assert report.samples[0].timeout == 360.0
    assert report.samples[0].timeout_retries == 1
    assert report.samples[0].timed_out is False


def test_evaluate_samples_retries_a_fallback_after_llm_timeout(tmp_path):
    pipeline = FakeRunPipeline(
        [
            _result(
                _REFERENCE,
                warnings=["The local LLM failed: Read timed out"],
            ),
            _result(_REFERENCE),
        ]
    )

    report = evaluate_samples(
        _timeout_sample(tmp_path),
        pipeline,
        timeout_increment=60,
        max_timeout_retries=2,
    )

    assert pipeline.timeouts == [300.0, 360.0]
    assert report.samples[0].ok
    assert report.samples[0].timeout_retries == 1


def test_evaluate_samples_groups_and_weights_by_complexity(tmp_path):
    different = (
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>'
        "<msqrt><mi>x</mi></msqrt>"
        "</mrow></math>"
    )
    dataset = tmp_path / "samples.jsonl"
    _write_jsonl(
        dataset,
        [
            {
                "id": "easy",
                "latex": "x^{2}",
                "reference_mathml": _REFERENCE,
                "eqn_complexity": "simple",
            },
            {
                "id": "hard",
                "latex": "x^{2}",
                "reference_mathml": _REFERENCE,
                "eqn_complexity": "compleja",
            },
        ],
    )
    pipeline = FakeRunPipeline([_result(_REFERENCE), _result(different)])

    report = evaluate_samples(
        load_evaluation_samples(dataset),
        pipeline,
        complexity_weights={"simple": 1.0, "compleja": 3.0},
    )

    assert set(report.by_complexity) == {"simple", "compleja"}
    assert report.by_complexity["simple"].structure_bleu.score == 1.0
    expected = (
        report.by_complexity["simple"].structure_bleu.score * 1.0
        + report.by_complexity["compleja"].structure_bleu.score * 3.0
    ) / 4.0
    assert report.weighted_structure_bleu == expected
    assert report.complexity_weights == {"simple": 1.0, "compleja": 3.0}
    assert "Por clasificación" in report_markdown(report)


def test_pipeline_applies_a_request_timeout_temporarily():
    pipeline = EquationPipeline()
    original = pipeline.settings.ollama_timeout

    with pipeline._request_timeout(123.0):
        assert pipeline.settings.ollama_timeout == 123.0
        assert pipeline._active_timeout == 123.0

    assert pipeline.settings.ollama_timeout == original
    assert pipeline._active_timeout is None


def test_evaluate_samples_scores_an_identical_offline_output(tmp_path):
    pipeline = EquationPipeline()
    reference = pipeline.run(EquationRequest(latex=r"\frac{a}{b}", flow=Flow.OFFLINE))
    dataset = tmp_path / "samples.jsonl"
    _write_jsonl(
        dataset,
        [
            {
                "id": "fraction",
                "latex": r"\frac{a}{b}",
                "reference_mathml": reference.mathml,
                "flow": "offline",
            }
        ],
    )

    report = evaluate_samples(
        load_evaluation_samples(dataset),
        pipeline,
        default_flow=Flow.OFFLINE,
    )

    assert len(report.samples) == 1
    assert report.samples[0].ok
    assert report.samples[0].structure.score == 1.0
    assert report.samples[0].intent.score == 1.0
    assert report.structure_bleu.score == 1.0
    assert report.intent_bleu.score == 1.0
    assert "BLEU estructura" in report_markdown(report)


def test_evaluate_samples_records_pipeline_errors(tmp_path):
    pipeline = EquationPipeline()
    dataset = tmp_path / "samples.jsonl"
    _write_jsonl(
        dataset,
        [
            {
                "id": "invalid",
                "latex": r"\frac{",
                "reference_mathml": _REFERENCE,
                "flow": "offline",
            }
        ],
    )

    report = evaluate_samples(
        load_evaluation_samples(dataset),
        pipeline,
        default_flow=Flow.OFFLINE,
    )

    assert report.failed_samples
    assert report.failed_samples[0].sample_id == "invalid"
    assert report.failed_samples[0].error
