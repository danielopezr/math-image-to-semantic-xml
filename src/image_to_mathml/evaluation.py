"""Evaluate pipeline outputs against a labelled MathML dataset."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

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
    timeout: float | None = None
    eqn_complexity: str = "unclassified"

    @property
    def input_description(self) -> str:
        if self.latex:
            return self.latex
        return self.image_path or ""

    def to_request(
        self,
        default_flow: Flow,
        *,
        timeout: float | None = None,
    ) -> EquationRequest:
        effective_timeout = self.timeout if timeout is None else timeout
        return EquationRequest(
            latex=self.latex,
            image_path=self.image_path,
            context=self.context,
            flow=self.flow or default_flow,
            ocr_backend=self.ocr_backend,
            allow_open_concepts=self.allow_open_concepts,
            max_repair_attempts=self.max_repair_attempts,
            timeout=effective_timeout,
        )


@dataclass(frozen=True)
class EvaluationConfig:
    path: Path
    csv_path: Path
    image_dir: Path | None = None
    mathml_dir: Path | None = None
    complexity_weights: dict[str, float] = field(default_factory=dict)


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
    timeout: float | None = None
    timeout_retries: int = 0
    timed_out: bool = False
    eqn_complexity: str = "unclassified"

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
            "timeout": self.timeout,
            "timeout_retries": self.timeout_retries,
            "timed_out": self.timed_out,
            "eqn_complexity": self.eqn_complexity,
        }
        if self.structure is not None:
            payload["structure_bleu"] = self.structure.as_dict()
        if self.intent is not None:
            payload["intent_bleu"] = self.intent.as_dict()
        return payload


@dataclass
class ComplexityMetrics:
    eqn_complexity: str
    samples: int
    weight: float
    structure_bleu: BleuScore
    intent_bleu: BleuScore

    def as_dict(self) -> dict[str, Any]:
        return {
            "eqn_complexity": self.eqn_complexity,
            "samples": self.samples,
            "weight": self.weight,
            "structure_bleu": self.structure_bleu.as_dict(),
            "intent_bleu": self.intent_bleu.as_dict(),
        }


@dataclass
class EvaluationReport:
    flow: str
    samples: list[SampleEvaluation]
    structure_bleu: BleuScore
    intent_bleu: BleuScore
    max_n: int = 4
    by_complexity: dict[str, ComplexityMetrics] = field(default_factory=dict)
    weighted_structure_bleu: float = 0.0
    weighted_intent_bleu: float = 0.0
    complexity_weights: dict[str, float] = field(default_factory=dict)

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
            "weighted_corpus": {
                "weights": self.complexity_weights,
                "structure_bleu": round(self.weighted_structure_bleu, 6),
                "intent_bleu": round(self.weighted_intent_bleu, 6),
            },
            "by_complexity": {
                name: metrics.as_dict() for name, metrics in self.by_complexity.items()
            },
            "samples": [sample.as_dict() for sample in self.samples],
        }


def _resolve_reference_file(value: str, base_dir: Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    try:
        markup = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"Could not read reference MathML file {path}: {exc}") from exc
    if not markup:
        raise ValueError(f"Reference MathML file {path} is empty")
    return markup


def _reference_from_record(record: dict[str, Any], base_dir: Path) -> str:
    for key in ("reference_mathml", "reference", "mathml", "reference_mathml_string"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for key in (
        "reference_mathml_file",
        "reference_file",
        "mathml_file",
        "reference_path",
    ):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return _resolve_reference_file(value, base_dir)
    reference_latex = record.get("reference_latex")
    if isinstance(reference_latex, str) and reference_latex.strip():
        return LatexToMathML().convert(reference_latex)
    raise ValueError(
        "Each sample needs 'reference_mathml', 'reference_mathml_file', "
        "'reference' or 'reference_latex'"
    )


def _normalise_complexity(value: object) -> str:
    text = str(value or "").strip().casefold()
    return text or "unclassified"


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
    timeout = record.get("timeout")
    if timeout is not None and str(timeout).strip():
        timeout = float(timeout)
    else:
        timeout = None
    return EvaluationSample(
        sample_id=sample_id,
        reference_mathml=_reference_from_record(record, base_dir),
        latex=str(latex) if latex else None,
        image_path=str(image_path) if image_path else None,
        context=_context_from_record(record),
        flow=flow,
        ocr_backend=record.get("ocr") or record.get("ocr_backend"),
        allow_open_concepts=None if allow_open is None else bool(allow_open),
        max_repair_attempts=(None if max_repairs is None else int(max_repairs)),
        timeout=timeout,
        eqn_complexity=_normalise_complexity(record.get("eqn_complexity")),
    )


def _first_value(data: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in data and data[name] not in (None, ""):
            return data[name]
    return None


def _resolve_config_path(value: object, base_dir: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    """Load a YAML file that points to the CSV and label/image folders."""

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("The evaluation YAML file must contain an object")

    base_dir = config_path.parent
    csv_value = _first_value(
        payload,
        "csv",
        "csv_file",
        "dataset_csv",
        "evaluation_csv",
    )
    if not csv_value:
        raise ValueError("The evaluation YAML file needs a 'csv' entry")
    csv_path = _resolve_config_path(csv_value, base_dir)
    if csv_path is None:
        raise ValueError("The evaluation YAML file needs a valid 'csv' entry")

    image_dir = _resolve_config_path(
        _first_value(payload, "images_dir", "image_dir", "images_path"),
        base_dir,
    )
    mathml_dir = _resolve_config_path(
        _first_value(
            payload,
            "mathml_dir",
            "labels_dir",
            "mathml_path",
            "labels_path",
        ),
        base_dir,
    )

    raw_weights = payload.get("complexity_weights") or {}
    if not isinstance(raw_weights, dict):
        raise TypeError("'complexity_weights' must be a mapping")
    complexity_weights: dict[str, float] = {}
    for name, value in raw_weights.items():
        try:
            complexity_weights[_normalise_complexity(name)] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid complexity weight for {name!r}: {value!r}"
            ) from exc

    return EvaluationConfig(
        path=config_path,
        csv_path=csv_path,
        image_dir=image_dir,
        mathml_dir=mathml_dir,
        complexity_weights=complexity_weights,
    )


def _csv_value(row: dict[str, str], header: dict[str, str], *names: str) -> str:
    for name in names:
        column = header.get(name)
        if column is None:
            continue
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _resolve_member_path(
    value: str,
    directory: Path | None,
    base_dir: Path,
) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    root = directory if directory is not None else base_dir
    return str((root / path).resolve())


def _load_csv_samples(
    path: Path,
    base_dir: Path,
    *,
    image_dir: Path | None = None,
    mathml_dir: Path | None = None,
) -> list[EvaluationSample]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("The CSV dataset needs a header row")
        header = {
            name.strip().lower(): name
            for name in reader.fieldnames
            if name and name.strip()
        }
        records: list[dict[str, Any]] = []
        for index, row in enumerate(reader, 1):
            if not any(
                isinstance(value, str) and value.strip() for value in row.values()
            ):
                continue
            image = _csv_value(
                row,
                header,
                "img_path",
                "image",
                "image_path",
            )
            mathml_file = _csv_value(
                row,
                header,
                "mathml_path",
                "reference_mathml_file",
                "mathml_file",
                "reference_file",
            )
            if not image:
                raise ValueError(f"CSV row {index} is missing 'img_path'")
            if not mathml_file:
                raise ValueError(f"CSV row {index} is missing 'mathml_path'")
            records.append(
                {
                    "id": _csv_value(row, header, "id", "sample_id", "name"),
                    "image": _resolve_member_path(image, image_dir, base_dir),
                    "reference_mathml_file": _resolve_member_path(
                        mathml_file, mathml_dir, base_dir
                    ),
                    "area": _csv_value(row, header, "area"),
                    "topic": _csv_value(row, header, "topic"),
                    "timeout": _csv_value(row, header, "timeout"),
                    "eqn_complexity": _csv_value(
                        row,
                        header,
                        "eqn_complexity",
                        "complexity",
                        "classification",
                    ),
                }
            )
    return [
        _sample_from_record(record, index, base_dir)
        for index, record in enumerate(records, 1)
    ]


def load_evaluation_samples(path: str | Path) -> list[EvaluationSample]:
    """Load a YAML, JSON, JSONL or CSV dataset of labelled equations."""

    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()
    base_dir = dataset_path.parent
    if suffix in {".yaml", ".yml"}:
        config = load_evaluation_config(dataset_path)
        return _load_csv_samples(
            config.csv_path,
            config.csv_path.parent,
            image_dir=config.image_dir,
            mathml_dir=config.mathml_dir,
        )
    if suffix == ".csv":
        return _load_csv_samples(dataset_path, base_dir)

    text = dataset_path.read_text(encoding="utf-8")
    if suffix in {".jsonl", ".jsonlines"}:
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        raw = payload.get("samples") if isinstance(payload, dict) else payload
        if not isinstance(raw, list):
            raise ValueError(
                "The dataset must be a JSON array or an object with samples"
            )

    return [
        _sample_from_record(record, index, base_dir)
        for index, record in enumerate(raw, 1)
    ]


_TIMEOUT_RE = re.compile(
    r"\b(?:timed?\s*out|timeout|read\s+timeout|connect\s+timeout)\b",
    re.IGNORECASE,
)
_TIMEOUT_VALUE_RE = re.compile(
    r"(?:read|connect)?\s*timeout\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


def _is_timeout_text(text: str) -> bool:
    return bool(_TIMEOUT_RE.search(text or ""))


def _result_timeout_text(result: object) -> str:
    warnings = getattr(result, "warnings", None) or []
    return " ".join(str(warning) for warning in warnings)


def _result_timed_out(result: object) -> bool:
    return _is_timeout_text(_result_timeout_text(result))


def _timeout_retry_base(
    current_timeout: float | None,
    text: str,
    fallback: float,
) -> float:
    if current_timeout is not None:
        return current_timeout
    match = _TIMEOUT_VALUE_RE.search(text or "")
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return fallback


def evaluate_samples(
    samples: list[EvaluationSample],
    pipeline: EquationPipeline,
    *,
    default_flow: Flow = Flow.HYBRID,
    max_n: int = 4,
    smooth: bool = True,
    timeout_increment: float = 60.0,
    max_timeout_retries: int = 3,
    complexity_weights: dict[str, float] | None = None,
) -> EvaluationReport:
    """Run every sample through the pipeline and compute both BLEU metrics.

    When a sample defines an initial ``timeout`` and the pipeline reports a
    timeout, the sample is retried with ``timeout + timeout_increment`` up to
    ``max_timeout_retries`` times.
    """

    evaluations: list[SampleEvaluation] = []
    references: list[str] = []
    candidates: list[str] = []
    class_references: dict[str, list[str]] = {}
    class_candidates: dict[str, list[str]] = {}
    retry_limit = max(0, max_timeout_retries)
    increment = max(0.0, timeout_increment)

    for sample in samples:
        current_timeout = sample.timeout
        retries = 0
        while True:
            evaluation = SampleEvaluation(
                sample_id=sample.sample_id,
                input=sample.input_description,
                reference_mathml=sample.reference_mathml,
                timeout=current_timeout,
                timeout_retries=retries,
                eqn_complexity=sample.eqn_complexity,
            )
            try:
                result = pipeline.run(
                    sample.to_request(default_flow, timeout=current_timeout)
                )
            except (
                ValueError,
                RuntimeError,
                OSError,
                MathMLNormalizationError,
            ) as exc:
                message = str(exc)
                timed_out = _is_timeout_text(message)
                if timed_out and retries < retry_limit:
                    current_timeout = (
                        _timeout_retry_base(
                            current_timeout,
                            message,
                            pipeline.settings.ollama_timeout,
                        )
                        + increment
                    )
                    retries += 1
                    continue
                evaluation.error = message
                evaluation.timed_out = timed_out
                evaluations.append(evaluation)
                break

            timed_out = _result_timed_out(result)
            if timed_out and retries < retry_limit:
                current_timeout = (
                    _timeout_retry_base(
                        current_timeout,
                        _result_timeout_text(result),
                        pipeline.settings.ollama_timeout,
                    )
                    + increment
                )
                retries += 1
                continue

            scores = mathml_bleu(
                sample.reference_mathml,
                result.mathml,
                max_n=max_n,
                smooth=smooth,
            )
            evaluation.output_mathml = result.mathml
            evaluation.structure = scores.structure
            evaluation.intent = scores.intent
            evaluation.validation_ok = result.validation.ok
            evaluation.timeout = current_timeout
            evaluation.timeout_retries = retries
            evaluation.timed_out = timed_out
            evaluations.append(evaluation)
            references.append(sample.reference_mathml)
            candidates.append(result.mathml)
            class_references.setdefault(sample.eqn_complexity, []).append(
                sample.reference_mathml
            )
            class_candidates.setdefault(sample.eqn_complexity, []).append(result.mathml)
            break

    corpus = mathml_corpus_bleu(
        references,
        candidates,
        max_n=max_n,
        smooth=smooth,
    )
    supplied_weights = {
        _normalise_complexity(name): float(value)
        for name, value in (complexity_weights or {}).items()
    }
    by_complexity: dict[str, ComplexityMetrics] = {}
    used_weights: dict[str, float] = {}
    for complexity in sorted(class_references):
        class_corpus = mathml_corpus_bleu(
            class_references[complexity],
            class_candidates[complexity],
            max_n=max_n,
            smooth=smooth,
        )
        weight = (
            supplied_weights.get(complexity, 1.0)
            if supplied_weights
            else float(len(class_references[complexity]))
        )
        used_weights[complexity] = weight
        by_complexity[complexity] = ComplexityMetrics(
            eqn_complexity=complexity,
            samples=len(class_references[complexity]),
            weight=weight,
            structure_bleu=class_corpus.structure,
            intent_bleu=class_corpus.intent,
        )

    total_weight = sum(used_weights.values())
    if total_weight > 0:
        weighted_structure = (
            sum(
                metrics.weight * metrics.structure_bleu.score
                for metrics in by_complexity.values()
            )
            / total_weight
        )
        weighted_intent = (
            sum(
                metrics.weight * metrics.intent_bleu.score
                for metrics in by_complexity.values()
            )
            / total_weight
        )
    else:
        weighted_structure = 0.0
        weighted_intent = 0.0

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
        by_complexity=by_complexity,
        weighted_structure_bleu=weighted_structure,
        weighted_intent_bleu=weighted_intent,
        complexity_weights=used_weights,
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
        f"- BLEU estructura ponderado: {report.weighted_structure_bleu:.6f}",
        f"- BLEU estructura + intent ponderado: {report.weighted_intent_bleu:.6f}",
        "",
        "| muestra | complejidad | BLEU estructura | BLEU intent | validación | error |",
        "|---|---|---|---|---|---|",
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
            f"| {sample_id} | {sample.eqn_complexity} | {structure} | {intent} | "
            f"{validation} | {error} |"
        )
    if report.by_complexity:
        lines.extend(
            [
                "",
                "## Por clasificación",
                "",
                "| clasificación | muestras | peso | BLEU estructura | BLEU intent |",
                "|---|---|---|---|---|",
            ]
        )
        for complexity, metrics in report.by_complexity.items():
            lines.append(
                f"| {complexity} | {metrics.samples} | {metrics.weight:.3f} | "
                f"{metrics.structure_bleu.score:.6f} | "
                f"{metrics.intent_bleu.score:.6f} |"
            )
    return "\n".join(lines)


def mathml_bleu_result(
    reference: str, candidate: str, max_n: int = 4
) -> MathMLBleuScore:
    """Convenience wrapper used by tests and external scripts."""

    return mathml_bleu(reference, candidate, max_n=max_n)
