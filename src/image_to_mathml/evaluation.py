"""Evaluate pipeline outputs against a labelled MathML dataset."""

from __future__ import annotations

import csv
import io
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm.base import ModelCall, record_model_calls
from .mathml.converter import LatexToMathML
from .mathml.normalizer import normalize_mathml
from .metrics.bleu import (
    BleuScore,
    IntentF1,
    MathMLBleuScore,
    MathMLNormalizationError,
    canonical_mathml_tokens,
    exact_match,
    intent_f1,
    lenient_exact_match,
    mathml_bleu,
    mathml_corpus_bleu,
)
from .models import EquationRequest, Flow, MathContext, PipelineResult
from .pipeline import EquationPipeline

# Flows whose result is supposed to come from a language model.  When such a
# run returns ``llm_used=False`` the output is the deterministic fallback.
_LLM_FLOWS = {Flow.HYBRID, Flow.DIRECT, Flow.AUTO, Flow.VISION}


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
    # "mathml" (hand-written or file) or "latex" (generated from reference_latex).
    reference_source: str = "mathml"

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
class StageTimes:
    """Where the time of one sample went, timeout retries included.

    ``vision`` and ``llm`` are the wall time of the requests to each model;
    ``other`` is the rest (local OCR, conversion, validation, scoring setup).
    ``load``, ``prompt`` and ``generate`` split the model requests as Ollama
    reports them: loading the model into memory, reading the prompt (and
    image) and writing the answer. A request cut by a timeout has no such
    split and is counted in ``cut_calls``. ``min_free_memory_mb`` is the
    lowest free RAM seen at the end of a model call (None if unknown).
    """

    vision_seconds: float = 0.0
    llm_seconds: float = 0.0
    other_seconds: float = 0.0
    load_seconds: float = 0.0
    prompt_seconds: float = 0.0
    generate_seconds: float = 0.0
    output_tokens: int = 0
    calls: int = 0
    cut_calls: int = 0
    min_free_memory_mb: float | None = None

    _FIELDS = (
        "vision_seconds",
        "llm_seconds",
        "other_seconds",
        "load_seconds",
        "prompt_seconds",
        "generate_seconds",
        "output_tokens",
        "calls",
        "cut_calls",
    )

    @classmethod
    def from_calls(cls, calls: list[ModelCall], elapsed: float) -> StageTimes:
        times = cls()
        for call in calls:
            if call.stage == "vision":
                times.vision_seconds += call.seconds
            else:
                times.llm_seconds += call.seconds
            times.load_seconds += call.load_seconds
            times.prompt_seconds += call.prompt_seconds
            times.generate_seconds += call.generate_seconds
            times.output_tokens += call.output_tokens
            times.calls += 1
            if not call.completed:
                times.cut_calls += 1
            times._see_memory(call.free_memory_mb)
        times.other_seconds = max(
            0.0, elapsed - times.vision_seconds - times.llm_seconds
        )
        return times

    @classmethod
    def from_dict(cls, data: object) -> StageTimes:
        times = cls()
        if isinstance(data, dict):
            for name in cls._FIELDS:
                value = data.get(name)
                if isinstance(value, int | float):
                    current = getattr(times, name)
                    setattr(times, name, type(current)(value))
            memory = data.get("min_free_memory_mb")
            if isinstance(memory, int | float):
                times.min_free_memory_mb = float(memory)
        return times

    def _see_memory(self, value: float | None) -> None:
        if value is not None and (
            self.min_free_memory_mb is None or value < self.min_free_memory_mb
        ):
            self.min_free_memory_mb = value

    def add(self, other: StageTimes) -> None:
        for name in self._FIELDS:
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self._see_memory(other.min_free_memory_mb)

    @property
    def tokens_per_second(self) -> float:
        if self.generate_seconds <= 0:
            return 0.0
        return self.output_tokens / self.generate_seconds

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name in self._FIELDS:
            value = getattr(self, name)
            payload[name] = round(value, 3) if isinstance(value, float) else value
        payload["tokens_per_second"] = round(self.tokens_per_second, 3)
        if self.min_free_memory_mb is not None:
            payload["min_free_memory_mb"] = round(self.min_free_memory_mb)
        return payload


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
    reference_source: str = "mathml"
    llm_used: bool | None = None
    fallback: bool = False
    structure_source: str = ""
    attempts: int = 0
    warnings: list[str] = field(default_factory=list)
    intent_f1: IntentF1 | None = None
    exact_structure: bool | None = None
    exact_intent: bool | None = None
    # Exact match ignoring bare <mrow> grouping and trailing punctuation.
    lenient_exact_structure: bool | None = None
    lenient_exact_intent: bool | None = None
    elapsed_seconds: float = 0.0
    # Attempts counted for the per-attempt time: every pipeline run (including
    # the timeout retries) and its internal repair attempts.
    attempts_total: int = 0
    stages: StageTimes = field(default_factory=StageTimes)
    # What each streamed model call cut by a timeout had written (to tell a
    # model stuck in a loop from one that is only slow).
    cut_outputs: list[str] = field(default_factory=list)
    # The LaTeX the structure was built from (the OCR reading for an image),
    # so the semantic stage can be evaluated again from it without the OCR.
    output_latex: str = ""
    # Every raw answer of the LLM (LLMOutput.as_dict): what it was given, what
    # it wrote and whether it was accepted.
    llm_outputs: list[dict[str, Any]] = field(default_factory=list)
    # Restored from a partial results file instead of being run again.
    resumed: bool = False

    @property
    def ok(self) -> bool:
        """The pipeline produced an output that could be scored."""

        return not self.error and self.structure is not None and self.intent is not None

    @property
    def seconds_per_attempt(self) -> float:
        """Total execution time split across the attempts that were made."""

        return self.elapsed_seconds / max(1, self.attempts_total)

    @property
    def scored_with_llm(self) -> bool:
        return self.ok and bool(self.llm_used) and not self.fallback

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sample_id": self.sample_id,
            "input": self.input,
            "reference_mathml": self.reference_mathml,
            "reference_source": self.reference_source,
            "output_mathml": self.output_mathml,
            "output_latex": self.output_latex,
            "validation_ok": self.validation_ok,
            "error": self.error,
            "timeout": self.timeout,
            "timeout_retries": self.timeout_retries,
            "timed_out": self.timed_out,
            "eqn_complexity": self.eqn_complexity,
            "llm_used": self.llm_used,
            "fallback": self.fallback,
            "structure_source": self.structure_source,
            "attempts": self.attempts,
            "attempts_total": self.attempts_total,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "seconds_per_attempt": round(self.seconds_per_attempt, 3),
            "stage_times": self.stages.as_dict(),
            "warnings": self.warnings,
            "exact_structure": self.exact_structure,
            "exact_intent": self.exact_intent,
            "lenient_exact_structure": self.lenient_exact_structure,
            "lenient_exact_intent": self.lenient_exact_intent,
        }
        if self.cut_outputs:
            payload["cut_outputs"] = self.cut_outputs
        if self.llm_outputs:
            payload["llm_outputs"] = self.llm_outputs
        if self.structure is not None:
            payload["structure_bleu"] = self.structure.as_dict()
        if self.intent is not None:
            payload["intent_bleu"] = self.intent.as_dict()
        if self.intent_f1 is not None:
            payload["intent_f1"] = self.intent_f1.as_dict()
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
    llm_structure_bleu: BleuScore | None = None
    llm_intent_bleu: BleuScore | None = None
    configuration: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def successful_samples(self) -> list[SampleEvaluation]:
        return [sample for sample in self.samples if sample.ok]

    @property
    def failed_samples(self) -> list[SampleEvaluation]:
        return [sample for sample in self.samples if not sample.ok]

    @property
    def llm_samples(self) -> list[SampleEvaluation]:
        return [sample for sample in self.samples if sample.scored_with_llm]

    @property
    def fallback_samples(self) -> list[SampleEvaluation]:
        return [sample for sample in self.samples if sample.ok and sample.fallback]

    @property
    def timed_out_samples(self) -> list[SampleEvaluation]:
        return [sample for sample in self.samples if sample.timed_out]

    def _rate(self, predicate) -> float:
        if not self.samples:
            return 0.0
        return sum(1 for sample in self.samples if predicate(sample)) / len(
            self.samples
        )

    @property
    def validation_rate(self) -> float:
        return self._rate(lambda sample: bool(sample.validation_ok))

    @property
    def exact_structure_rate(self) -> float:
        return self._rate(lambda sample: bool(sample.exact_structure))

    @property
    def exact_intent_rate(self) -> float:
        return self._rate(lambda sample: bool(sample.exact_intent))

    @property
    def lenient_exact_structure_rate(self) -> float:
        return self._rate(lambda sample: bool(sample.lenient_exact_structure))

    @property
    def lenient_exact_intent_rate(self) -> float:
        return self._rate(lambda sample: bool(sample.lenient_exact_intent))

    @property
    def stage_totals(self) -> StageTimes:
        """Time per stage added over every sample (resumed ones included)."""

        totals = StageTimes()
        for sample in self.samples:
            totals.add(sample.stages)
        return totals

    @property
    def mean_intent_f1(self) -> float:
        if not self.samples:
            return 0.0
        return sum(
            sample.intent_f1.f1 if sample.intent_f1 is not None else 0.0
            for sample in self.samples
        ) / len(self.samples)

    def as_dict(self) -> dict[str, Any]:
        llm_corpus: dict[str, Any] | None = None
        if self.llm_structure_bleu is not None and self.llm_intent_bleu is not None:
            llm_corpus = {
                "samples": len(self.llm_samples),
                "structure_bleu": self.llm_structure_bleu.as_dict(),
                "intent_bleu": self.llm_intent_bleu.as_dict(),
            }
        return {
            "flow": self.flow,
            "max_n": self.max_n,
            "configuration": self.configuration,
            "samples_total": len(self.samples),
            "samples_ok": len(self.successful_samples),
            "samples_failed": len(self.failed_samples),
            "samples_llm": len(self.llm_samples),
            "samples_fallback": len(self.fallback_samples),
            "samples_timed_out": len(self.timed_out_samples),
            "corpus": {
                "structure_bleu": self.structure_bleu.as_dict(),
                "intent_bleu": self.intent_bleu.as_dict(),
            },
            "llm_only_corpus": llm_corpus,
            "weighted_corpus": {
                "weighting": "per equation (class weight x samples in the class)",
                "weights": self.complexity_weights,
                "structure_bleu": round(self.weighted_structure_bleu, 6),
                "intent_bleu": round(self.weighted_intent_bleu, 6),
            },
            "rates": {
                "validation": round(self.validation_rate, 6),
                "exact_structure": round(self.exact_structure_rate, 6),
                "exact_intent": round(self.exact_intent_rate, 6),
                "lenient_exact_structure": round(self.lenient_exact_structure_rate, 6),
                "lenient_exact_intent": round(self.lenient_exact_intent_rate, 6),
                "mean_intent_f1": round(self.mean_intent_f1, 6),
            },
            "by_complexity": {
                name: metrics.as_dict() for name, metrics in self.by_complexity.items()
            },
            "stage_times": self.stage_totals.as_dict(),
            "notes": self.notes,
            "samples": [sample.as_dict() for sample in self.samples],
        }


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
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


def _reference_from_record(record: dict[str, Any], base_dir: Path) -> tuple[str, str]:
    """Return ``(reference_mathml, source)``; source is "mathml" or "latex"."""

    for key in ("reference_mathml", "reference", "mathml", "reference_mathml_string"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value, "mathml"
    for key in (
        "reference_mathml_file",
        "reference_file",
        "mathml_file",
        "reference_path",
    ):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return _resolve_reference_file(value, base_dir), "mathml"
    reference_latex = record.get("reference_latex")
    if isinstance(reference_latex, str) and reference_latex.strip():
        # Normalise with the same code as the pipeline output, so the score
        # does not penalise purely cosmetic differences of latex2mathml.
        return normalize_mathml(LatexToMathML().convert(reference_latex)), "latex"
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


def _parse_timeout(value: Any, sample_id: str) -> float | None:
    if value is None or not str(value).strip():
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Sample {sample_id!r}: invalid timeout {value!r}; use a number of seconds"
        ) from exc


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
    reference, reference_source = _reference_from_record(record, base_dir)
    return EvaluationSample(
        sample_id=sample_id,
        reference_mathml=reference,
        latex=str(latex) if latex else None,
        image_path=str(image_path) if image_path else None,
        context=_context_from_record(record),
        flow=flow,
        ocr_backend=record.get("ocr") or record.get("ocr_backend"),
        allow_open_concepts=None if allow_open is None else bool(allow_open),
        max_repair_attempts=(None if max_repairs is None else int(max_repairs)),
        timeout=_parse_timeout(record.get("timeout"), sample_id),
        eqn_complexity=_normalise_complexity(record.get("eqn_complexity")),
        reference_source=reference_source,
    )


def _first_value(data: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in data and data[name] not in (None, ""):
            return data[name]
    return None


_ENV_DEFAULT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: str) -> str:
    """Expand ``$VAR``, ``${VAR}`` and ``${VAR:-default}`` in a config path.

    This lets a shared YAML point to a dataset folder that lives in a
    different place on each machine.
    """

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        resolved = os.environ.get(name)
        if resolved:
            return resolved
        if default is not None:
            return default
        return match.group(0)

    return os.path.expandvars(_ENV_DEFAULT_RE.sub(replace, value))


def _resolve_config_path(value: object, base_dir: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(_expand_env(str(value))).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_evaluation_config(
    path: str | Path,
    *,
    images_dir: str | Path | None = None,
    mathml_dir: str | Path | None = None,
) -> EvaluationConfig:
    """Load a YAML file that points to the CSV and label/image folders.

    ``images_dir`` and ``mathml_dir`` override the YAML values (for example
    from the command line) so the same YAML works on several machines.
    """

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is a dependency
        raise ValueError("PyYAML is required to read YAML evaluation files") from exc

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
        images_dir
        if images_dir is not None
        else _first_value(payload, "images_dir", "image_dir", "images_path"),
        Path.cwd() if images_dir is not None else base_dir,
    )
    labels_dir = _resolve_config_path(
        mathml_dir
        if mathml_dir is not None
        else _first_value(
            payload,
            "mathml_dir",
            "labels_dir",
            "mathml_path",
            "labels_path",
        ),
        Path.cwd() if mathml_dir is not None else base_dir,
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
        mathml_dir=labels_dir,
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
            sample_id = _csv_value(row, header, "id", "sample_id", "name")
            records.append(
                {
                    "id": sample_id or f"row-{index}",
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


def load_evaluation_samples(
    path: str | Path,
    *,
    config: EvaluationConfig | None = None,
) -> list[EvaluationSample]:
    """Load a YAML, JSON, JSONL or CSV dataset of labelled equations.

    Pass an already loaded YAML ``config`` to avoid reading it twice.
    """

    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()
    base_dir = dataset_path.parent
    if suffix in {".yaml", ".yml"}:
        config = config or load_evaluation_config(dataset_path)
        return _load_csv_samples(
            config.csv_path,
            config.csv_path.parent,
            image_dir=config.image_dir,
            mathml_dir=config.mathml_dir,
        )
    if suffix == ".csv":
        return _load_csv_samples(dataset_path, base_dir)

    text = dataset_path.read_text(encoding="utf-8")
    raw: Any
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


# ---------------------------------------------------------------------------
# timeouts
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def _configuration_summary(pipeline: object) -> dict[str, Any]:
    settings = getattr(pipeline, "settings", None)
    if settings is None:
        return {}
    llm = getattr(pipeline, "llm", None)
    return {
        "llm_backend": getattr(settings, "llm_backend", ""),
        "model": getattr(llm, "model", getattr(settings, "ollama_model", "")),
        "vision_model": getattr(settings, "ollama_vision_model", ""),
        "ocr_backend": getattr(settings, "ocr_backend", ""),
        "max_repair_attempts": getattr(settings, "max_repair_attempts", None),
    }


def _weighted(
    by_complexity: dict[str, ComplexityMetrics],
    attribute: str,
) -> float:
    total = sum(metrics.weight * metrics.samples for metrics in by_complexity.values())
    if total <= 0:
        return 0.0
    return (
        sum(
            metrics.weight * metrics.samples * getattr(metrics, attribute).score
            for metrics in by_complexity.values()
        )
        / total
    )


# Tail of a cut answer kept per call: enough to see a loop, small in the JSON.
_CUT_OUTPUT_CHARS = 2000


def _run_sample(
    evaluation: SampleEvaluation,
    sample: EvaluationSample,
    pipeline: EquationPipeline,
    *,
    default_flow: Flow,
    retry_limit: int,
    increment: float,
    default_timeout: float,
) -> str | None:
    """Run one sample (with timeout retries) and return its MathML output.

    ``None`` means the pipeline raised and produced nothing to score.
    """

    flow = sample.flow or default_flow
    sample_started = time.perf_counter()
    with record_model_calls() as calls:
        result = _run_with_retries(
            evaluation,
            sample,
            pipeline,
            default_flow=default_flow,
            retry_limit=retry_limit,
            increment=increment,
            default_timeout=default_timeout,
        )
    evaluation.elapsed_seconds = time.perf_counter() - sample_started
    evaluation.stages = StageTimes.from_calls(calls, evaluation.elapsed_seconds)
    evaluation.cut_outputs = [
        f"{call.stage}: {call.partial_output[-_CUT_OUTPUT_CHARS:]}"
        for call in calls
        if not call.completed and call.partial_output
    ]
    if result is None:
        return None
    evaluation.output_mathml = result.mathml
    evaluation.output_latex = str(getattr(result, "latex", "") or "")
    evaluation.llm_outputs = [
        output.as_dict() for output in getattr(result, "llm_outputs", None) or []
    ]
    evaluation.validation_ok = result.validation.ok
    evaluation.llm_used = bool(result.llm_used)
    evaluation.structure_source = result.structure_source
    evaluation.attempts = result.attempts
    evaluation.warnings = list(result.warnings)
    evaluation.fallback = flow in _LLM_FLOWS and not result.llm_used
    return result.mathml


def _run_with_retries(
    evaluation: SampleEvaluation,
    sample: EvaluationSample,
    pipeline: EquationPipeline,
    *,
    default_flow: Flow,
    retry_limit: int,
    increment: float,
    default_timeout: float,
) -> PipelineResult | None:
    """Run the pipeline, retrying timeouts with a longer timeout each time."""

    attempts_total = 0
    current_timeout = sample.timeout
    retries = 0
    result = None
    while True:
        try:
            result = pipeline.run(
                sample.to_request(default_flow, timeout=current_timeout)
            )
        except (ValueError, RuntimeError, OSError) as exc:
            attempts_total += 1
            message = str(exc)
            timed_out = _is_timeout_text(message)
            if timed_out and retries < retry_limit:
                current_timeout = (
                    _timeout_retry_base(current_timeout, message, default_timeout)
                    + increment
                )
                retries += 1
                continue
            evaluation.error = message
            evaluation.timed_out = timed_out
            # The LaTeX the OCR read is kept even when a later step failed.
            evaluation.output_latex = str(getattr(exc, "latex", "") or "")
            result = None
            break

        attempts_total += max(1, int(getattr(result, "attempts", 0) or 0))
        timed_out = _result_timed_out(result)
        if timed_out and retries < retry_limit:
            current_timeout = (
                _timeout_retry_base(
                    current_timeout,
                    _result_timeout_text(result),
                    default_timeout,
                )
                + increment
            )
            retries += 1
            continue
        evaluation.timed_out = timed_out
        break

    evaluation.attempts_total = attempts_total
    evaluation.timeout = current_timeout
    evaluation.timeout_retries = retries
    return result


def _restore_sample(evaluation: SampleEvaluation, data: dict[str, Any]) -> str | None:
    """Fill ``evaluation`` from a saved :meth:`SampleEvaluation.as_dict`.

    Only what the pipeline produced is restored; the scores are recomputed, so
    a resumed run is scored with the current references and metrics.
    """

    produced = data.get("validation_ok") is not None
    evaluation.output_mathml = str(data.get("output_mathml") or "")
    evaluation.validation_ok = data.get("validation_ok")
    evaluation.llm_used = data.get("llm_used")
    evaluation.fallback = bool(data.get("fallback", False))
    evaluation.structure_source = str(data.get("structure_source") or "")
    evaluation.attempts = int(data.get("attempts") or 0)
    evaluation.attempts_total = int(data.get("attempts_total") or 0)
    evaluation.warnings = [str(item) for item in data.get("warnings") or []]
    evaluation.timeout = data.get("timeout")
    evaluation.timeout_retries = int(data.get("timeout_retries") or 0)
    evaluation.timed_out = bool(data.get("timed_out", False))
    evaluation.elapsed_seconds = float(data.get("elapsed_seconds") or 0.0)
    evaluation.stages = StageTimes.from_dict(data.get("stage_times"))
    evaluation.cut_outputs = [str(item) for item in data.get("cut_outputs") or []]
    evaluation.output_latex = str(data.get("output_latex") or "")
    evaluation.llm_outputs = [
        item for item in data.get("llm_outputs") or [] if isinstance(item, dict)
    ]
    evaluation.resumed = True
    # A pipeline error is kept; a scoring error is recomputed when scoring.
    evaluation.error = "" if produced else str(data.get("error") or "")
    return evaluation.output_mathml if produced else None


def _score_sample(
    evaluation: SampleEvaluation,
    reference: str,
    output: str | None,
    *,
    max_n: int,
    smooth: bool,
) -> str:
    """Score ``output`` against ``reference``; return the scored candidate."""

    candidate = ""
    if output is not None:
        try:
            scores = mathml_bleu(reference, output, max_n=max_n, smooth=smooth)
        except MathMLNormalizationError as exc:
            evaluation.error = f"Invalid output MathML: {exc}"
        else:
            evaluation.structure = scores.structure
            evaluation.intent = scores.intent
            candidate = output
            evaluation.intent_f1 = intent_f1(reference, candidate)
            evaluation.exact_structure = exact_match(
                reference, candidate, include_intent=False
            )
            evaluation.exact_intent = exact_match(
                reference, candidate, include_intent=True
            )
            evaluation.lenient_exact_structure = lenient_exact_match(
                reference, candidate, include_intent=False
            )
            evaluation.lenient_exact_intent = lenient_exact_match(
                reference, candidate, include_intent=True
            )

    if evaluation.intent_f1 is None:
        evaluation.intent_f1 = intent_f1(reference, None)
        evaluation.exact_structure = False
        evaluation.exact_intent = False
        evaluation.lenient_exact_structure = False
        evaluation.lenient_exact_intent = False
    return candidate


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
    on_sample: Callable[[SampleEvaluation, int, int], None] | None = None,
    completed: dict[str, dict[str, Any]] | None = None,
) -> EvaluationReport:
    """Run every sample through the pipeline and compute the metrics.

    ``on_sample(evaluation, index, total)`` is called as soon as each sample
    is scored, so callers can report progress or save partial results.
    ``completed`` maps sample ids to saved :meth:`SampleEvaluation.as_dict`
    entries: those samples are not run again, their saved output is scored
    instead (used to resume an interrupted evaluation).

    When the pipeline reports a timeout (as an exception or as a fallback
    warning) the sample is retried with ``timeout + timeout_increment`` up to
    ``max_timeout_retries`` times.  The starting value is the sample's
    ``timeout`` or, if it has none, the timeout of the failed request.  The
    timeout applies to each HTTP request, not to the whole sample.

    Every sample with a valid reference counts in the corpus scores: a sample
    whose run failed counts as an empty output.  Samples whose output is the
    deterministic fallback of an LLM flow are flagged; ``llm_structure_bleu``
    and ``llm_intent_bleu`` cover only the samples the LLM really produced.
    """

    evaluations: list[SampleEvaluation] = []
    references: list[str] = []
    candidates: list[str] = []
    llm_references: list[str] = []
    llm_candidates: list[str] = []
    class_references: dict[str, list[str]] = {}
    class_candidates: dict[str, list[str]] = {}
    retry_limit = max(0, max_timeout_retries)
    increment = max(0.0, timeout_increment)
    settings = getattr(pipeline, "settings", None)
    default_timeout = float(getattr(settings, "ollama_timeout", 180.0))

    total = len(samples)
    for index, sample in enumerate(samples, start=1):
        evaluation = SampleEvaluation(
            sample_id=sample.sample_id,
            input=sample.input_description,
            reference_mathml=sample.reference_mathml,
            timeout=sample.timeout,
            eqn_complexity=sample.eqn_complexity,
            reference_source=sample.reference_source,
        )
        evaluations.append(evaluation)

        try:
            canonical_mathml_tokens(sample.reference_mathml, include_intent=True)
        except MathMLNormalizationError as exc:
            # Without a valid reference the sample cannot be scored at all.
            evaluation.error = f"Invalid reference MathML: {exc}"
            if on_sample is not None:
                on_sample(evaluation, index, total)
            continue

        previous = (completed or {}).get(sample.sample_id)
        if previous is not None:
            output = _restore_sample(evaluation, previous)
        else:
            output = _run_sample(
                evaluation,
                sample,
                pipeline,
                default_flow=default_flow,
                retry_limit=retry_limit,
                increment=increment,
                default_timeout=default_timeout,
            )
        candidate = _score_sample(
            evaluation, sample.reference_mathml, output, max_n=max_n, smooth=smooth
        )

        # Failed samples count as an empty output instead of being excluded.
        references.append(sample.reference_mathml)
        candidates.append(candidate)
        class_references.setdefault(sample.eqn_complexity, []).append(
            sample.reference_mathml
        )
        class_candidates.setdefault(sample.eqn_complexity, []).append(candidate)
        if evaluation.scored_with_llm:
            llm_references.append(sample.reference_mathml)
            llm_candidates.append(candidate)
        if on_sample is not None:
            on_sample(evaluation, index, total)

    corpus = mathml_corpus_bleu(
        references,
        candidates,
        max_n=max_n,
        smooth=smooth,
    )
    llm_corpus = (
        mathml_corpus_bleu(llm_references, llm_candidates, max_n=max_n, smooth=smooth)
        if llm_references
        else None
    )

    supplied_weights = {
        _normalise_complexity(name): float(value)
        for name, value in (complexity_weights or {}).items()
    }
    notes: list[str] = []
    by_complexity: dict[str, ComplexityMetrics] = {}
    used_weights: dict[str, float] = {}
    for complexity in sorted(class_references):
        class_corpus = mathml_corpus_bleu(
            class_references[complexity],
            class_candidates[complexity],
            max_n=max_n,
            smooth=smooth,
        )
        if supplied_weights and complexity not in supplied_weights:
            notes.append(
                f"La clasificación {complexity!r} no tiene peso en "
                "complexity_weights; se usa 1.0."
            )
        weight = supplied_weights.get(complexity, 1.0)
        used_weights[complexity] = weight
        by_complexity[complexity] = ComplexityMetrics(
            eqn_complexity=complexity,
            samples=len(class_references[complexity]),
            weight=weight,
            structure_bleu=class_corpus.structure,
            intent_bleu=class_corpus.intent,
        )

    if any(sample.reference_source == "latex" for sample in samples):
        notes.append(
            "Algunas referencias se generaron desde reference_latex y no tienen "
            "intents: su BLEU de intent no mide la anotación semántica."
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
        by_complexity=by_complexity,
        weighted_structure_bleu=_weighted(by_complexity, "structure_bleu"),
        weighted_intent_bleu=_weighted(by_complexity, "intent_bleu"),
        complexity_weights=used_weights,
        llm_structure_bleu=llm_corpus.structure if llm_corpus else None,
        llm_intent_bleu=llm_corpus.intent if llm_corpus else None,
        configuration=_configuration_summary(pipeline),
        notes=notes,
    )


def _llm_cell(sample: SampleEvaluation) -> str:
    if not sample.ok:
        return "-"
    if sample.fallback:
        return "respaldo"
    if sample.llm_used:
        return "sí"
    return "no aplica"


def _timeout_cell(sample: SampleEvaluation) -> str:
    if sample.timed_out:
        return f"agotado ({sample.timeout_retries} reintentos)"
    if sample.timeout_retries:
        return f"{sample.timeout_retries} reintentos"
    return ""


_CSV_HEADERS = (
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
)


def _yes_no(value: bool | None) -> str:
    if value is None:
        return ""
    return "sí" if value else "no"


def report_csv(report: EvaluationReport) -> str:
    """Return the per-sample report table as CSV.

    It mirrors the table printed by :func:`report_markdown` and adds the exact
    and lenient exact matches, the total execution time of each equation,
    the time per attempt (total time divided by the attempts made, including
    timeout retries) and the time per stage (see :class:`StageTimes`).
    """

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(_CSV_HEADERS)
    for sample in report.samples:
        structure = (
            f"{sample.structure.score:.6f}" if sample.structure is not None else ""
        )
        intent = f"{sample.intent.score:.6f}" if sample.intent is not None else ""
        f1 = f"{sample.intent_f1.f1:.6f}" if sample.intent_f1 is not None else ""
        writer.writerow(
            [
                sample.sample_id,
                sample.eqn_complexity,
                structure,
                intent,
                f1,
                _yes_no(sample.exact_structure),
                _yes_no(sample.exact_intent),
                _yes_no(sample.lenient_exact_structure),
                _yes_no(sample.lenient_exact_intent),
                _llm_cell(sample),
                _yes_no(sample.validation_ok),
                _timeout_cell(sample),
                " ".join(sample.error.split()),
                f"{sample.elapsed_seconds:.3f}",
                f"{sample.seconds_per_attempt:.3f}",
                f"{sample.stages.vision_seconds:.3f}",
                f"{sample.stages.llm_seconds:.3f}",
                f"{sample.stages.other_seconds:.3f}",
                f"{sample.stages.load_seconds:.3f}",
                f"{sample.stages.prompt_seconds:.3f}",
                f"{sample.stages.generate_seconds:.3f}",
                sample.stages.output_tokens,
                sample.stages.cut_calls,
            ]
        )
    return buffer.getvalue()


def _share(part: float, total: float) -> str:
    return f"{part / total:.0%}" if total > 0 else "-"


def _stage_lines(report: EvaluationReport) -> list[str]:
    totals = report.stage_totals
    elapsed = sum(sample.elapsed_seconds for sample in report.samples)
    if not report.samples or elapsed <= 0:
        return []
    count = len(report.samples)
    model_seconds = totals.vision_seconds + totals.llm_seconds
    rows = [
        ("visión (OCR con el modelo multimodal)", totals.vision_seconds, elapsed),
        ("LLM de texto", totals.llm_seconds, elapsed),
        ("resto (OCR local, conversión, validación)", totals.other_seconds, elapsed),
    ]
    lines = [
        "",
        "## Tiempo por etapa",
        "",
        "| etapa | total | media por muestra | parte del total |",
        "|---|---|---|---|",
    ]
    for name, seconds, whole in rows:
        lines.append(
            f"| {name} | {seconds:.0f} s | {seconds / count:.1f} s | "
            f"{_share(seconds, whole)} |"
        )
    lines.extend(
        [
            "",
            "Dentro de las llamadas a los modelos (según Ollama):",
            "",
            "| parte | total | parte de las llamadas |",
            "|---|---|---|",
            (
                f"| carga del modelo en memoria | {totals.load_seconds:.0f} s | "
                f"{_share(totals.load_seconds, model_seconds)} |"
            ),
            (
                f"| lectura del prompt e imagen | {totals.prompt_seconds:.0f} s | "
                f"{_share(totals.prompt_seconds, model_seconds)} |"
            ),
            (
                f"| generación de la respuesta | {totals.generate_seconds:.0f} s | "
                f"{_share(totals.generate_seconds, model_seconds)} |"
            ),
            "",
            (
                f"- {totals.calls} llamadas a modelos, {totals.output_tokens} tokens "
                f"generados ({totals.tokens_per_second:.1f} tokens/s)"
            ),
        ]
    )
    if totals.cut_calls:
        lines.append(
            f"- {totals.cut_calls} llamadas cortadas por timeout: su tiempo cuenta "
            "en la etapa pero no en el reparto de carga, lectura y generación"
        )
    if totals.min_free_memory_mb is not None:
        lines.append(
            f"- RAM libre mínima al terminar una llamada: "
            f"{totals.min_free_memory_mb:.0f} MB"
        )
    return lines


def report_markdown(report: EvaluationReport) -> str:
    configuration = report.configuration
    lines = [
        "# Evaluación BLEU",
        "",
        f"- flujo: {report.flow}",
    ]
    if configuration:
        lines.append(
            f"- modelo: {configuration.get('model', '')} · visión: "
            f"{configuration.get('vision_model', '')} · OCR: "
            f"{configuration.get('ocr_backend', '')}"
        )
    lines.extend(
        [
            (
                f"- muestras: {len(report.samples)} "
                f"({len(report.llm_samples)} con LLM, "
                f"{len(report.fallback_samples)} con respaldo determinista, "
                f"{len(report.failed_samples)} con error; "
                f"{len(report.timed_out_samples)} terminaron en timeout)"
            ),
            f"- BLEU estructura (corpus): {report.structure_bleu.score:.6f}",
            f"- BLEU estructura + intent (corpus): {report.intent_bleu.score:.6f}",
        ]
    )
    if report.llm_structure_bleu is not None and report.llm_intent_bleu is not None:
        lines.extend(
            [
                f"- BLEU estructura (solo LLM): {report.llm_structure_bleu.score:.6f}",
                (
                    "- BLEU estructura + intent (solo LLM): "
                    f"{report.llm_intent_bleu.score:.6f}"
                ),
            ]
        )
    lines.extend(
        [
            (
                "- BLEU estructura ponderado por ecuación: "
                f"{report.weighted_structure_bleu:.6f}"
            ),
            (
                "- BLEU estructura + intent ponderado por ecuación: "
                f"{report.weighted_intent_bleu:.6f}"
            ),
            f"- validación correcta: {report.validation_rate:.1%}",
            f"- coincidencia exacta de estructura: {report.exact_structure_rate:.1%}",
            f"- coincidencia exacta con intents: {report.exact_intent_rate:.1%}",
            (
                "- coincidencia exacta flexible de estructura (sin agrupación "
                f"<mrow> ni puntuación final): {report.lenient_exact_structure_rate:.1%}"
            ),
            (
                "- coincidencia exacta flexible con intents: "
                f"{report.lenient_exact_intent_rate:.1%}"
            ),
            f"- F1 de intents (media): {report.mean_intent_f1:.6f}",
            "",
            (
                "| muestra | complejidad | BLEU estructura | BLEU intent | F1 intent "
                "| LLM | validación | timeout | error |"
            ),
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for sample in report.samples:
        structure = (
            f"{sample.structure.score:.6f}" if sample.structure is not None else "-"
        )
        intent = f"{sample.intent.score:.6f}" if sample.intent is not None else "-"
        f1 = f"{sample.intent_f1.f1:.3f}" if sample.intent_f1 is not None else "-"
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
            f"{f1} | {_llm_cell(sample)} | {validation} | {_timeout_cell(sample)} | "
            f"{error} |"
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
        lines.extend(
            [
                "",
                (
                    "El ponderado multiplica el peso de cada clasificación por su "
                    "número de ecuaciones."
                ),
            ]
        )
    lines.extend(_stage_lines(report))
    if report.notes:
        lines.extend(["", "## Notas", ""])
        lines.extend(f"- {note}" for note in report.notes)
    return "\n".join(lines)


def mathml_bleu_result(
    reference: str, candidate: str, max_n: int = 4
) -> MathMLBleuScore:
    """Convenience wrapper used by tests and external scripts."""

    return mathml_bleu(reference, candidate, max_n=max_n)
