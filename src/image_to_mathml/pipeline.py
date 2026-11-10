"""Agentic orchestrator for equation recognition and intent annotation."""

from __future__ import annotations

import time
from dataclasses import replace

from .agents import DirectMathMLAgent, HybridIntentAgent
from .config import Settings
from .llm.base import LLMClient, LLMError
from .llm.factory import build_llm
from .mathml.annotator import annotate_tree
from .mathml.concepts import ConceptRegistry, get_registry
from .mathml.converter import LatexToMathML
from .mathml.latex import clean_latex
from .mathml.normalizer import (
    NormalizationError,
    normalize_mathml,
    parse_and_normalize,
    serialize_mathml,
)
from .mathml.signature import same_structure
from .mathml.validator import (
    prune_invalid_intents,
    prune_orphan_args,
    validate_mathml,
)
from .models import EquationRequest, Flow, MathContext, PipelineResult
from .ocr.base import OCREngine
from .ocr.factory import build_ocr
from .ocr.preprocess import PreprocessConfig


class EquationPipeline:
    """Run OCR, MathML generation, intent annotation, validation and repair.

    The pipeline is deliberately explicit about its agents:

    ``OCR`` -> ``LaTeX cleaner`` -> ``MathML generator/annotator`` ->
    ``deterministic normaliser`` -> ``validator`` -> ``repair loop``.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        ocr: OCREngine | None = None,
        registry: ConceptRegistry | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.llm = llm or build_llm(self.settings)
        self.ocr = ocr
        self.converter = LatexToMathML()
        self.registry = registry or get_registry()
        self.direct_agent = DirectMathMLAgent(self.llm, self.settings)
        self.hybrid_agent = HybridIntentAgent(self.llm, self.settings)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def run(self, request: EquationRequest) -> PipelineResult:
        started = time.perf_counter()
        latex, ocr_backend = self._resolve_latex(request)
        allow_open = (
            self.settings.allow_open_concepts
            if request.allow_open_concepts is None
            else request.allow_open_concepts
        )
        max_repairs = (
            self.settings.max_repair_attempts
            if request.max_repair_attempts is None
            else request.max_repair_attempts
        )
        max_repairs = max(0, max_repairs)

        if request.flow is Flow.AUTO:
            hybrid = self._run_flow(
                Flow.HYBRID, request, latex, ocr_backend, allow_open, max_repairs
            )
            if hybrid.validation.acceptable:
                hybrid.trace.insert(
                    0, "auto: hybrid accepted; skipped the slower direct flow"
                )
                hybrid.elapsed_seconds = time.perf_counter() - started
                return hybrid

            direct = self._run_flow(
                Flow.DIRECT, request, latex, ocr_backend, allow_open, max_repairs
            )
            if direct.validation.acceptable:
                result = direct
                chosen = "direct"
            else:
                result = hybrid if hybrid.validation.ok else direct
                chosen = result.flow.value
            result.trace.insert(
                0, f"auto: evaluated hybrid and direct; selected {chosen}"
            )
            result.elapsed_seconds = time.perf_counter() - started
            return result

        result = self._run_flow(
            request.flow, request, latex, ocr_backend, allow_open, max_repairs
        )
        result.elapsed_seconds = time.perf_counter() - started
        return result

    # ------------------------------------------------------------------
    # input resolution
    # ------------------------------------------------------------------
    def _resolve_latex(self, request: EquationRequest) -> tuple[str, str]:
        if request.latex and request.latex.strip():
            return clean_latex(request.latex), "none"
        if not request.image_path:
            raise ValueError("Provide either latex=... or image_path=...")

        engine = self.ocr
        if engine is None:
            backend = request.ocr_backend or self.settings.ocr_backend
            engine = build_ocr(
                backend,
                preprocess=PreprocessConfig.from_env(
                    enabled=self.settings.ocr_preprocess,
                    profile=self.settings.ocr_preprocess_profile,
                ),
            )
        if engine is None:
            raise ValueError(
                "An image was provided but OCR is disabled. Install a supported "
                "OCR backend (texify, pix2tex, rapid_latex_ocr, pix2text or "
                "paddleocr) or pass the recognised LaTeX directly."
            )
        latex = clean_latex(engine.image_to_latex(request.image_path))
        if not latex:
            raise ValueError("The OCR backend returned an empty LaTeX string")
        return latex, engine.name

    # ------------------------------------------------------------------
    # flow dispatch
    # ------------------------------------------------------------------
    def _run_flow(
        self,
        flow: Flow,
        request: EquationRequest,
        latex: str,
        ocr_backend: str,
        allow_open: bool,
        max_repairs: int,
    ) -> PipelineResult:
        if flow is Flow.OFFLINE:
            return self._offline(latex, request.context, ocr_backend, allow_open)
        if flow is Flow.DIRECT:
            return self._direct(
                latex, request.context, ocr_backend, allow_open, max_repairs
            )
        if flow is Flow.HYBRID:
            return self._hybrid(
                latex, request.context, ocr_backend, allow_open, max_repairs
            )
        raise ValueError(f"Unsupported flow: {flow}")

    def _allowed_open(self, allow_open: bool) -> set[str] | None:
        if not allow_open or self.settings.allow_all_open_concepts:
            return None
        return set(self.settings.allowed_open_concepts)

    # ------------------------------------------------------------------
    # deterministic fallback
    # ------------------------------------------------------------------
    def _offline(
        self,
        latex: str,
        context: MathContext,
        ocr_backend: str,
        allow_open: bool,
    ) -> PipelineResult:
        base = self.converter.convert(latex)
        root = parse_and_normalize(base)
        count = annotate_tree(root)
        prune_orphan_args(root)
        mathml = serialize_mathml(root)
        report = validate_mathml(
            mathml,
            allow_open_concepts=allow_open,
            allowed_open_concepts=self._allowed_open(allow_open),
            registry=self.registry,
        )
        warnings: list[str] = []
        if count == 0:
            warnings.append("The deterministic annotator found no safe intent to add")
        return PipelineResult(
            latex=latex,
            mathml=mathml,
            flow=Flow.OFFLINE,
            validation=report,
            attempts=0,
            llm_used=False,
            ocr_backend=ocr_backend,
            warnings=warnings,
            trace=["latex2mathml", "deterministic-intent-annotation", "validation"],
        )

    # ------------------------------------------------------------------
    # flow 1: direct LaTeX -> MathML
    # ------------------------------------------------------------------
    def _direct(
        self,
        latex: str,
        context: MathContext,
        ocr_backend: str,
        allow_open: bool,
        max_repairs: int,
    ) -> PipelineResult:
        warnings: list[str] = []
        trace = ["direct-flow", f"model={self.llm.model}"]

        if not self._llm_ready(warnings):
            result = self._offline(latex, context, ocr_backend, allow_open)
            result.flow = Flow.DIRECT
            result.trace = trace + result.trace
            result.warnings.extend(warnings)
            return result

        feedback: list[str] | None = None
        attempts = 0
        for attempt in range(max_repairs + 1):
            attempts = attempt + 1
            try:
                raw = self.direct_agent.run(
                    latex,
                    context,
                    feedback,
                    allow_open_concepts=allow_open,
                    allow_all_open_concepts=(
                        allow_open and self.settings.allow_all_open_concepts
                    ),
                )
            except LLMError as exc:
                warnings.append(f"The local LLM failed during the direct flow: {exc}")
                trace.append(f"attempt {attempts}: llm-error")
                break

            try:
                candidate = normalize_mathml(raw)
                candidate_root = parse_and_normalize(candidate)
                if prune_orphan_args(candidate_root):
                    candidate = serialize_mathml(candidate_root)
            except NormalizationError as exc:
                feedback = [f"Invalid MathML: {exc}"]
                trace.append(f"attempt {attempts}: normalization-error: {exc}")
                continue

            report = validate_mathml(
                candidate,
                allow_open_concepts=allow_open,
                allowed_open_concepts=self._allowed_open(allow_open),
                registry=self.registry,
            )
            if report.acceptable:
                trace.append(f"attempt {attempts}: accepted")
                return PipelineResult(
                    latex=latex,
                    mathml=candidate,
                    flow=Flow.DIRECT,
                    validation=report,
                    attempts=attempts,
                    llm_used=True,
                    ocr_backend=ocr_backend,
                    warnings=warnings,
                    trace=trace,
                )

            feedback = report.error_messages()
            if not feedback:
                feedback = [
                    "The MathML is valid XML but has no usable intent attributes."
                ]
            trace.append(
                f"attempt {attempts}: rejected ({len(report.errors)} errors, "
                f"{report.intent_count} intents)"
            )

        fallback = self._offline(latex, context, ocr_backend, allow_open)
        fallback.flow = Flow.DIRECT
        fallback.attempts = attempts
        fallback.warnings.extend(warnings)
        fallback.warnings.append(
            "The direct flow did not produce an acceptable result; deterministic "
            "fallback was used."
        )
        fallback.trace = trace + fallback.trace
        return fallback

    # ------------------------------------------------------------------
    # flow 2: latex2mathml skeleton -> Ollama intent annotation
    # ------------------------------------------------------------------
    def _hybrid(
        self,
        latex: str,
        context: MathContext,
        ocr_backend: str,
        allow_open: bool,
        max_repairs: int,
    ) -> PipelineResult:
        warnings: list[str] = []
        trace = ["hybrid-flow", "latex2mathml", f"model={self.llm.model}"]
        base_markup = self.converter.convert(latex)
        base_mathml = normalize_mathml(base_markup)
        base_root = parse_and_normalize(base_mathml)

        if not self._llm_ready(warnings):
            result = self._annotate_base(
                latex, base_markup, ocr_backend, allow_open, warnings
            )
            result.flow = Flow.HYBRID
            result.trace = trace + result.trace
            return result

        feedback: list[str] | None = None
        attempts = 0
        for attempt in range(max_repairs + 1):
            attempts = attempt + 1
            try:
                raw = self.hybrid_agent.run(
                    latex,
                    base_mathml,
                    context,
                    feedback,
                    allow_open_concepts=allow_open,
                    allow_all_open_concepts=(
                        allow_open and self.settings.allow_all_open_concepts
                    ),
                )
            except LLMError as exc:
                warnings.append(f"The local LLM failed during the hybrid flow: {exc}")
                trace.append(f"attempt {attempts}: llm-error")
                break

            try:
                candidate = normalize_mathml(raw)
                candidate_root = parse_and_normalize(candidate)
            except NormalizationError as exc:
                feedback = [f"Invalid MathML: {exc}"]
                trace.append(f"attempt {attempts}: normalization-error: {exc}")
                continue

            if not same_structure(base_root, candidate_root):
                feedback = [
                    (
                        "The mathematical structure changed. Keep every element, its "
                        "order and its text identical; only add intent and arg "
                        "attributes."
                    )
                ]
                trace.append(f"attempt {attempts}: structure-changed")
                continue

            prune_orphan_args(candidate_root)
            candidate = serialize_mathml(candidate_root)
            report = validate_mathml(
                candidate,
                allow_open_concepts=allow_open,
                allowed_open_concepts=self._allowed_open(allow_open),
                registry=self.registry,
            )
            if report.acceptable:
                trace.append(f"attempt {attempts}: accepted")
                return PipelineResult(
                    latex=latex,
                    mathml=candidate,
                    flow=Flow.HYBRID,
                    validation=report,
                    attempts=attempts,
                    llm_used=True,
                    ocr_backend=ocr_backend,
                    warnings=warnings,
                    trace=trace,
                )

            if report.errors:
                # Keep the model's valid annotations and let the deterministic
                # annotator rebuild only the invalid/missing ones.
                merged_root = parse_and_normalize(candidate)
                removed = prune_invalid_intents(
                    merged_root,
                    allow_open_concepts=allow_open,
                    allowed_open_concepts=self._allowed_open(allow_open),
                    registry=self.registry,
                )
                annotate_tree(merged_root)
                prune_orphan_args(merged_root)
                merged = serialize_mathml(merged_root)
                merged_report = validate_mathml(
                    merged,
                    allow_open_concepts=allow_open,
                    allowed_open_concepts=self._allowed_open(allow_open),
                    registry=self.registry,
                )
                if merged_report.acceptable:
                    trace.append(
                        f"attempt {attempts}: merged after pruning {removed} "
                        "invalid intent(s)"
                    )
                    return PipelineResult(
                        latex=latex,
                        mathml=merged,
                        flow=Flow.HYBRID,
                        validation=merged_report,
                        attempts=attempts,
                        llm_used=True,
                        ocr_backend=ocr_backend,
                        warnings=warnings,
                        trace=trace,
                    )
                candidate = merged
                report = merged_report

            feedback = report.error_messages()
            if not feedback:
                feedback = ["No intent attributes were added; annotate the expression."]
            trace.append(
                f"attempt {attempts}: rejected ({len(report.errors)} errors, "
                f"{report.intent_count} intents)"
            )

        fallback = self._annotate_base(
            latex, base_markup, ocr_backend, allow_open, warnings
        )
        fallback.flow = Flow.HYBRID
        fallback.attempts = attempts
        fallback.warnings.append(
            "The model annotation did not produce an acceptable result; "
            "deterministic annotation was used on the latex2mathml skeleton."
        )
        fallback.trace = trace + fallback.trace
        return fallback

    def _annotate_base(
        self,
        latex: str,
        base_markup: str,
        ocr_backend: str,
        allow_open: bool,
        warnings: list[str],
    ) -> PipelineResult:
        root = parse_and_normalize(base_markup)
        count = annotate_tree(root)
        prune_orphan_args(root)
        mathml = serialize_mathml(root)
        report = validate_mathml(
            mathml,
            allow_open_concepts=allow_open,
            allowed_open_concepts=self._allowed_open(allow_open),
            registry=self.registry,
        )
        if count == 0:
            warnings.append("The deterministic annotator found no safe intent to add")
        return PipelineResult(
            latex=latex,
            mathml=mathml,
            flow=Flow.HYBRID,
            validation=report,
            attempts=0,
            llm_used=False,
            ocr_backend=ocr_backend,
            warnings=warnings,
            trace=["deterministic-intent-annotation", "validation"],
        )

    def _llm_ready(self, warnings: list[str]) -> bool:
        base_url = getattr(self.llm, "base_url", None) or self.settings.ollama_base_url
        backend = self.settings.llm_backend
        if not self.llm.is_available():
            warnings.append(
                f"The local LLM server ({backend}) is not reachable at {base_url}; "
                "using the deterministic fallback."
            )
            return False
        if not self.llm.has_model():
            warnings.append(
                f"Model {self.llm.model!r} is not installed on the {backend} "
                "server; using the deterministic fallback."
            )
            return False
        return True


def compare_flows(
    request: EquationRequest,
    pipeline: EquationPipeline | None = None,
) -> dict[str, PipelineResult]:
    """Run direct, hybrid and offline flows for the same input.

    This is the helper used by ``image-to-mathml compare`` to evaluate the two
    generation strategies requested for this module.
    """

    pipeline = pipeline or EquationPipeline()
    latex, ocr_backend = pipeline._resolve_latex(request)
    allow_open = (
        pipeline.settings.allow_open_concepts
        if request.allow_open_concepts is None
        else request.allow_open_concepts
    )
    max_repairs = (
        pipeline.settings.max_repair_attempts
        if request.max_repair_attempts is None
        else request.max_repair_attempts
    )
    max_repairs = max(0, max_repairs)

    results: dict[str, PipelineResult] = {}
    for flow in (Flow.HYBRID, Flow.DIRECT, Flow.OFFLINE):
        started = time.perf_counter()
        results[flow.value] = pipeline._run_flow(
            flow,
            replace(request, flow=flow),
            latex,
            ocr_backend,
            allow_open,
            max_repairs,
        )
        results[flow.value].elapsed_seconds = time.perf_counter() - started
    return results
