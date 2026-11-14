"""Evaluate pipeline outputs against a labelled MathML dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .mathml.converter import LatexToMathML
from .metrics.bleu import (
    BleuScore,
    MathMLBleuScore,
    MathMLNormalizationError,
    mathml_bleu,
    mathml_corpus_bleu,
)
from .models import EquationRequest, Flow, MathContext
from .pipeline import EquationPipeline


@dataclass(frozen=True)
class EvaluationSample:
    sample_id: str
    reference_mathml: str
    latex: str | None = None
    image_path: str | None = None
    context: MathContext = field(default_factory=MathContext)
    flow: Flow | None = None
    ocr_backend: str | None = None
    allow_open_concepts: bool | None = None
    max_repair_attempts: int | None = None

    @property
    def input_description(self) -> str:
        if self.latex:
            return self.latex
        return self.image_path or ""

    def to_request(self, default_flow: Flow) -> EquationRequest:
        return EquationRequest(
            latex=self.latex,
            image_path=self.image_path,
            context=self.context,
            flow=self.flow or default_flow,
            ocr_backend=self.ocr_backend,
            allow_open_concepts=self.allow_open_concepts,
            max_repair_attempts=self.max_repair_attempts,
        )


@dataclass
class SampleEvaluation:
    sample_id: str
    input: str
    reference_mathml: str = ""
    output_mathml: str = ""
    structure: BleuScore | None = None
    intent: BleuScore | None = None
    validation_ok: bool | None = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.structure is not None and self.intent is not None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sample_id": self.sample_id,
            "input": self.input,
            "reference_mathml": self.reference_mathml,
            "output_mathml": self.output_mathml,
            "validation_ok": self.validation_ok,
            "error": self.error,
        }
        if self.structure is not None:
            payload["structure_bleu"] = self.structure.as_dict()
        if self.intent is not None:
            payload["intent_bleu"] = self.intent.as_dict()
        return payload


@dataclass
class EvaluationReport:
    flow: str
    samples: list[SampleEvaluation]
    structure_bleu: BleuScore
    intent_bleu: BleuScore
    max_n: int = 4

    @property
    def successful_samples(self) -> list[SampleEvaluation]:
        return [sample for sample in self.samples if sample.ok]

    @property
    def failed_samples(self) -> list[SampleEvaluation]:
        return [sample for sample in self.samples if not sample.ok]

    def as_dict(self) -> dict[str, Any]:
        return {
            "flow": self.flow,
            "max_n": self.max_n,
            "samples_total": len(self.samples),
            "samples_ok": len(self.successful_samples),
            "samples_failed": len(self.failed_samples),
            "corpus": {
                "structure_bleu": self.structure_bleu.as_dict(),
                "intent_bleu": self.intent_bleu.as_dict(),
            },
            "samples": [sample.as_dict() for sample in self.samples],
        }


def _reference_from_record(record: dict[str, Any]) -> str:
    for key in ("reference_mathml", "reference", "mathml", "reference_mathml_string"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    reference_latex = record.get("reference_latex")
    if isinstance(reference_latex, str) and reference_latex.strip():
        return LatexToMathML().convert(reference_latex)
    raise ValueError(
        "Each sample needs 'reference_mathml', 'reference' or 'reference_latex'"
    )


def _context_from_record(record: dict[str, Any]) -> MathContext:
    return MathContext(
        area=str(record.get("area") or ""),
        topic=str(record.get("topic") or ""),
        surrounding_text=str(
            record.get("context_text") or record.get("surrounding_text") or ""
        ),
        language=str(record.get("lang") or record.get("language") or "es"),
    )


def _sample_from_record(
    record: dict[str, Any],
    index: int,
    base_dir: Path,
) -> EvaluationSample:
    if not isinstance(record, dict):
        raise TypeError(f"Dataset sample {index} must be a JSON object")

    latex = record.get("latex")
    image_path = record.get("image") or record.get("image_path")
    if not latex and not image_path:
        raise ValueError(
            f"Dataset sample {index} needs either 'latex' or an image path"
        )

    flow_value = record.get("flow")
    flow = Flow(str(flow_value).strip().lower()) if flow_value else None
    if image_path:
        image_path = str(image_path)
        path = Path(image_path)
        if not path.is_absolute():
            path = base_dir / path
        image_path = str(path)

    sample_id = str(
        record.get("id")
        or record.get("sample_id")
        or record.get("name")
        or f"sample-{index}"
    )
    allow_open = record.get("allow_open_concepts")
    max_repairs = record.get("max_repair_attempts")
    return EvaluationSample(
        sample_id=sample_id,
        reference_mathml=_reference_from_record(record),
        latex=str(latex) if latex else None,
        image_path=str(image_path) if image_path else None,
        context=_context_from_record(record),
        flow=flow,
        ocr_backend=record.get("ocr") or record.get("ocr_backend"),
        allow_open_concepts=None if allow_open is None else bool(allow_open),
        max_repair_attempts=(None if max_repairs is None else int(max_repairs)),
    )


def load_evaluation_samples(path: str | Path) -> list[EvaluationSample]:
    """Load a JSON array or JSONL dataset of labelled equations."""

    dataset_path = Path(path)
    text = dataset_path.read_text(encoding="utf-8")
    if dataset_path.suffix.lower() in {".jsonl", ".jsonlines"}:
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        raw = payload.get("samples") if isinstance(payload, dict) else payload
        if not isinstance(raw, list):
            raise ValueError(
                "The dataset must be a JSON array or an object with samples"
            )

    base_dir = dataset_path.parent
    return [
        _sample_from_record(record, index, base_dir)
        for index, record in enumerate(raw, 1)
    ]


def evaluate_samples(
    samples: list[EvaluationSample],
    pipeline: EquationPipeline,
    *,
    default_flow: Flow = Flow.HYBRID,
    max_n: int = 4,
    smooth: bool = True,
) -> EvaluationReport:
    """Run every sample through the pipeline and compute both BLEU metrics."""

    evaluations: list[SampleEvaluation] = []
    references: list[str] = []
    candidates: list[str] = []

    for sample in samples:
        evaluation = SampleEvaluation(
            sample_id=sample.sample_id,
            input=sample.input_description,
            reference_mathml=sample.reference_mathml,
        )
        try:
            result = pipeline.run(sample.to_request(default_flow))
            scores = mathml_bleu(
                sample.reference_mathml,
                result.mathml,
                max_n=max_n,
                smooth=smooth,
            )
        except (ValueError, RuntimeError, OSError, MathMLNormalizationError) as exc:
            evaluation.error = str(exc)
            evaluations.append(evaluation)
            continue

        evaluation.output_mathml = result.mathml
        evaluation.structure = scores.structure
        evaluation.intent = scores.intent
        evaluation.validation_ok = result.validation.ok
        evaluations.append(evaluation)
        references.append(sample.reference_mathml)
        candidates.append(result.mathml)

    corpus = mathml_corpus_bleu(
        references,
        candidates,
        max_n=max_n,
        smooth=smooth,
    )
    effective_flows = {(sample.flow or default_flow).value for sample in samples}
    if not effective_flows:
        flow_name = default_flow.value
    elif len(effective_flows) == 1:
        flow_name = next(iter(effective_flows))
    else:
        flow_name = "mixed"
    return EvaluationReport(
        flow=flow_name,
        samples=evaluations,
        structure_bleu=corpus.structure,
        intent_bleu=corpus.intent,
        max_n=max_n,
    )


def report_markdown(report: EvaluationReport) -> str:
    lines = [
        "# Evaluación BLEU",
        "",
        f"- flujo: {report.flow}",
        (
            f"- muestras: {len(report.samples)} "
            f"({len(report.successful_samples)} correctas, "
            f"{len(report.failed_samples)} con error)"
        ),
        f"- BLEU estructura (corpus): {report.structure_bleu.score:.6f}",
        f"- BLEU estructura + intent (corpus): {report.intent_bleu.score:.6f}",
        "",
        "| muestra | BLEU estructura | BLEU intent | validación | error |",
        "|---|---|---|---|---|",
    ]
    for sample in report.samples:
        structure = (
            f"{sample.structure.score:.6f}" if sample.structure is not None else "-"
        )
        intent = f"{sample.intent.score:.6f}" if sample.intent is not None else "-"
        validation = (
            "-"
            if sample.validation_ok is None
            else ("sí" if sample.validation_ok else "no")
        )
        sample_id = sample.sample_id.replace("|", "\\|")
        error = (
            " ".join(sample.error.split()).replace("|", "\\|") if sample.error else ""
        )
        lines.append(
            f"| {sample_id} | {structure} | {intent} | {validation} | {error} |"
        )
    return "\n".join(lines)


def mathml_bleu_result(
    reference: str, candidate: str, max_n: int = 4
) -> MathMLBleuScore:
    """Convenience wrapper used by tests and external scripts."""

    return mathml_bleu(reference, candidate, max_n=max_n)
