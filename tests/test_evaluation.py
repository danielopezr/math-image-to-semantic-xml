import json

from image_to_mathml.evaluation import (
    evaluate_samples,
    load_evaluation_samples,
    report_markdown,
)
from image_to_mathml.models import EquationRequest, Flow
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
