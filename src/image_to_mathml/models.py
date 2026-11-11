"""Data objects exchanged by the pipeline agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .timing import PhaseTiming


class Flow(str, Enum):
    """Supported generation strategies."""

    DIRECT = "direct"
    HYBRID = "hybrid"
    AUTO = "auto"
    OFFLINE = "offline"
    VISION = "vision"


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class MathContext:
    """Context recovered by the previous stage of the accessibility pipeline."""

    area: str = ""
    topic: str = ""
    surrounding_text: str = ""
    language: str = "es"

    def prompt_lines(self) -> str:
        lines = []
        if self.area:
            lines.append(f"math area: {self.area}")
        if self.topic:
            lines.append(f"topic: {self.topic}")
        if self.surrounding_text:
            text = " ".join(self.surrounding_text.split())
            if len(text) > 700:
                text = text[:697] + "..."
            lines.append(f"surrounding textbook text: {text}")
        lines.append(f"reader language: {self.language}")
        return "\n".join(lines)


@dataclass
class EquationRequest:
    """One equation to recognise.

    Either ``latex`` or ``image_path`` must be supplied.  If ``latex`` is
    present the OCR stage is skipped.
    """

    latex: str | None = None
    image_path: str | None = None
    context: MathContext = field(default_factory=MathContext)
    flow: Flow = Flow.HYBRID
    ocr_backend: str | None = None
    allow_open_concepts: bool | None = None
    max_repair_attempts: int | None = None
    direct_vision: bool | None = None


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    path: str = ""

    def format(self) -> str:
        location = f" at {self.path}" if self.path else ""
        return f"[{self.severity.value}] {self.code}{location}: {self.message}"


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    intent_count: int = 0
    arg_count: int = 0
    element_count: int = 0
    redundant_mrows: int = 0

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def acceptable(self) -> bool:
        return self.ok and self.intent_count > 0

    def add(
        self,
        severity: Severity,
        code: str,
        message: str,
        path: str = "",
    ) -> None:
        self.issues.append(ValidationIssue(severity, code, message, path))

    def error(self, code: str, message: str, path: str = "") -> None:
        self.add(Severity.ERROR, code, message, path)

    def warning(self, code: str, message: str, path: str = "") -> None:
        self.add(Severity.WARNING, code, message, path)

    def error_messages(self) -> list[str]:
        return [issue.format() for issue in self.errors]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "acceptable": self.acceptable,
            "intent_count": self.intent_count,
            "arg_count": self.arg_count,
            "element_count": self.element_count,
            "redundant_mrows": self.redundant_mrows,
            "issues": [
                {
                    "severity": issue.severity.value,
                    "code": issue.code,
                    "message": issue.message,
                    "path": issue.path,
                }
                for issue in self.issues
            ],
        }


@dataclass
class LLMOutput:
    """One MathML structure returned by the LLM during a pipeline run."""

    flow: Flow
    attempt: int
    mathml: str
    accepted: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "flow": self.flow.value,
            "attempt": self.attempt,
            "accepted": self.accepted,
            "mathml": self.mathml,
        }


@dataclass
class PipelineResult:
    latex: str
    mathml: str
    flow: Flow
    validation: ValidationReport
    attempts: int = 0
    llm_used: bool = False
    ocr_backend: str = "none"
    warnings: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    trace: list[str] = field(default_factory=list)
    llm_outputs: list[LLMOutput] = field(default_factory=list)
    timings: list[PhaseTiming] = field(default_factory=list)

    @property
    def structure_source(self) -> str:
        """Return which component produced the MathML structure."""

        if self.flow in {Flow.DIRECT, Flow.VISION} and self.llm_used:
            return "llm"
        return "latex2mathml"

    def as_dict(self) -> dict[str, Any]:
        return {
            "latex": self.latex,
            "mathml": self.mathml,
            "flow": self.flow.value,
            "structure_source": self.structure_source,
            "attempts": self.attempts,
            "llm_used": self.llm_used,
            "ocr_backend": self.ocr_backend,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "warnings": self.warnings,
            "trace": self.trace,
            "llm_outputs": [output.as_dict() for output in self.llm_outputs],
            "timings": [timing.as_dict() for timing in self.timings],
            "timing_summary": self.timing_summary(),
            "validation": self.validation.as_dict(),
        }

    def timing_summary(self) -> dict[str, Any]:
        attempts = [timing for timing in self.timings if timing.phase == "attempt"]
        attempts_total = round(sum(timing.duration_ms for timing in attempts), 3)
        phase_totals: dict[str, float] = {}
        for timing in self.timings:
            if timing.attempt is not None or timing.phase == "attempt":
                continue
            phase_totals[timing.phase] = round(
                phase_totals.get(timing.phase, 0.0) + timing.duration_ms, 3
            )
        return {
            "total_ms": round(self.elapsed_seconds * 1000, 3),
            "phase_totals_ms": phase_totals,
            "attempt_count": len(attempts),
            "attempts_total_ms": attempts_total,
            "attempts_average_ms": (
                round(attempts_total / len(attempts), 3) if attempts else 0.0
            ),
        }
