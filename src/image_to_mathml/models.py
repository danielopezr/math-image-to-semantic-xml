"""Data objects exchanged by the pipeline agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Flow(str, Enum):
    """Supported generation strategies."""

    DIRECT = "direct"
    HYBRID = "hybrid"
    AUTO = "auto"
    OFFLINE = "offline"


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

    @property
    def structure_source(self) -> str:
        """Return which component produced the MathML structure."""

        if self.flow is Flow.DIRECT and self.llm_used:
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
            "validation": self.validation.as_dict(),
        }
