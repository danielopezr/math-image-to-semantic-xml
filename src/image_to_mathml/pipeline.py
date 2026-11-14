"""Agentic orchestrator for equation recognition and intent annotation."""

from __future__ import annotations

import copy
import time
from dataclasses import replace

from lxml import etree

from .agents import DirectMathMLAgent, HybridIntentAgent, VisionMathMLAgent
from .config import Settings
from .llm.base import LLMClient, LLMError
from .llm.factory import build_llm
from .mathml.annotator import annotate_tree
from .mathml.concept_rag import ConceptRAG
from .mathml.concepts import ConceptRegistry, get_registry
from .mathml.converter import LatexToMathML
from .mathml.latex import clean_latex
from .mathml.normalizer import (
    NormalizationError,
    normalize_mathml,
    parse_and_normalize,
    serialize_mathml,
)
from .mathml.open_concept_rag import OpenConceptRAG
from .mathml.semantic import (
    SemanticError,
    apply_semantic_payload,
    assign_node_ids,
    parse_semantic_payload,
    semantic_tree_markup,
    strip_node_ids,
    strip_semantic_attributes,
)
from .mathml.signature import same_structure
from .mathml.validator import (
    prune_invalid_intents,
    prune_orphan_args,
    validate_mathml,
)
from .models import (
    EquationRequest,
    Flow,
    LLMOutput,
    MathContext,
    PipelineResult,
    ValidationReport,
)
from .ocr.base import OCREngine, OCRError
from .ocr.factory import build_ocr
from .ocr.preprocess import PreprocessConfig
from .timing import TimingRecorder


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
        self.concept_rag = (
            ConceptRAG(self.registry) if self.settings.concept_rag else None
        )
        self.open_concept_rag = (
            OpenConceptRAG(self.registry) if self.settings.open_concept_rag else None
        )
        self.direct_agent = DirectMathMLAgent(self.llm, self.settings)
        self.hybrid_agent = HybridIntentAgent(self.llm, self.settings)
        self.vision_agent = VisionMathMLAgent(self.settings)

    def _finish(
        self,
        result: PipelineResult,
        started: float,
        timings: TimingRecorder,
    ) -> PipelineResult:
        result.timings = timings.as_list()
        result.elapsed_seconds = time.perf_counter() - started
        return result

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def run(self, request: EquationRequest) -> PipelineResult:
        started = time.perf_counter()
        timings = TimingRecorder()
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

        if request.flow is Flow.VISION:
            with timings.group("vision"):
                result = self._vision(request, allow_open, max_repairs, timings=timings)
            return self._finish(result, started, timings)

        if request.flow is Flow.DIRECT and self._direct_vision_enabled(request):
            with timings.group("direct"):
                result = self._vision(
                    request,
                    allow_open,
                    max_repairs,
                    flow=Flow.DIRECT,
                    allow_fallback=True,
                    timings=timings,
                )
            return self._finish(result, started, timings)

        latex, ocr_backend = self._resolve_latex(request, timings=timings)

        if request.flow is Flow.AUTO:
            hybrid = self._run_flow(
                Flow.HYBRID,
                request,
                latex,
                ocr_backend,
                allow_open,
                max_repairs,
                timings,
            )
            if hybrid.validation.acceptable:
                hybrid.trace.insert(
                    0, "auto: hybrid accepted; skipped the slower direct flow"
                )
                return self._finish(hybrid, started, timings)

            direct = self._run_flow(
                Flow.DIRECT,
                request,
                latex,
                ocr_backend,
                allow_open,
                max_repairs,
                timings,
            )
            if direct.validation.acceptable:
                result = direct
                chosen = "direct"
            else:
                result = hybrid if hybrid.validation.ok else direct
                chosen = result.flow.value
            result.llm_outputs = hybrid.llm_outputs + direct.llm_outputs
            result.trace.insert(
                0, f"auto: evaluated hybrid and direct; selected {chosen}"
            )
            return self._finish(result, started, timings)

        result = self._run_flow(
            request.flow,
            request,
            latex,
            ocr_backend,
            allow_open,
            max_repairs,
            timings,
        )
        return self._finish(result, started, timings)

    # ------------------------------------------------------------------
    # input resolution
    # ------------------------------------------------------------------
    def _resolve_latex(
        self,
        request: EquationRequest,
        *,
        timings: TimingRecorder | None = None,
        note: str = "",
    ) -> tuple[str, str]:
        recorder = timings or TimingRecorder()
        if request.latex and request.latex.strip():
            with recorder.measure("latex-clean", "clean_latex", note=note):
                latex = clean_latex(request.latex)
            return latex, "none"
        if not request.image_path:
            raise ValueError("Provide either latex=... or image_path=...")

        engine = self.ocr
        if engine is None:
            backend = request.ocr_backend or self.settings.ocr_backend
            with recorder.measure("ocr-setup", backend, note=note):
                engine = build_ocr(
                    backend,
                    preprocess=PreprocessConfig.from_env(
                        enabled=self.settings.ocr_preprocess,
                        profile=self.settings.ocr_preprocess_profile,
                    ),
                    stream=self.settings.ollama_stream,
                )
        if engine is None:
            raise ValueError(
                "An image was provided but OCR is disabled. Install a supported "
                "OCR backend (texify, pix2tex, rapid_latex_ocr, pix2text or "
                "paddleocr) or pass the recognised LaTeX directly."
            )
        with recorder.measure("ocr", engine.name, note=note):
            raw_latex = engine.image_to_latex(request.image_path)
        with recorder.measure("latex-clean", "clean_latex", note=note):
            latex = clean_latex(raw_latex)
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
        timings: TimingRecorder,
    ) -> PipelineResult:
        with timings.group(flow.value):
            if flow is Flow.OFFLINE:
                return self._offline(
                    latex, request.context, ocr_backend, allow_open, timings
                )
            if flow is Flow.DIRECT:
                if self._direct_vision_enabled(request):
                    return self._vision(
                        request,
                        allow_open,
                        max_repairs,
                        flow=Flow.DIRECT,
                        allow_fallback=True,
                        latex=latex,
                        ocr_backend=ocr_backend,
                        timings=timings,
                    )
                return self._direct(
                    latex,
                    request.context,
                    ocr_backend,
                    allow_open,
                    max_repairs,
                    timings,
                )
            if flow is Flow.HYBRID:
                return self._hybrid(
                    latex,
                    request.context,
                    ocr_backend,
                    allow_open,
                    max_repairs,
                    timings,
                )
            if flow is Flow.VISION:
                return self._vision(request, allow_open, max_repairs, timings=timings)
            raise ValueError(f"Unsupported flow: {flow}")

    def _allowed_open(self, allow_open: bool) -> set[str] | None:
        if not allow_open or self.settings.allow_all_open_concepts:
            return None
        return set(self.settings.allowed_open_concepts)

    def _direct_vision_enabled(self, request: EquationRequest) -> bool:
        enabled = (
            self.settings.direct_vision
            if request.direct_vision is None
            else request.direct_vision
        )
        return bool(enabled and request.image_path)

    # ------------------------------------------------------------------
    # deterministic fallback
    # ------------------------------------------------------------------
    def _offline(
        self,
        latex: str,
        context: MathContext,
        ocr_backend: str,
        allow_open: bool,
        timings: TimingRecorder,
        *,
        note: str = "",
    ) -> PipelineResult:
        with timings.measure("latex2mathml", "latex2mathml", note=note):
            base = self.converter.convert(latex)
        with timings.measure("normalize", "normalizer", note=note):
            root = parse_and_normalize(base)
        with timings.measure("deterministic-annotation", "annotator", note=note):
            count = annotate_tree(root)
            prune_orphan_args(root)
            mathml = serialize_mathml(root)
        with timings.measure("validate", "validator", note=note):
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
        timings: TimingRecorder,
    ) -> PipelineResult:
        warnings: list[str] = []
        trace = ["direct-flow", f"model={self.llm.model}"]
        llm_outputs: list[LLMOutput] = []
        tool = f"{self.direct_agent.role}[{self.llm.model}]"

        with timings.measure("llm-check", "ollama"):
            llm_ready = self._llm_ready(warnings)
        if not llm_ready:
            result = self._offline(
                latex,
                context,
                ocr_backend,
                allow_open,
                timings,
                note="LLM unavailable; deterministic fallback",
            )
            result.flow = Flow.DIRECT
            result.trace = trace + result.trace
            result.warnings.extend(warnings)
            return result

        feedback: list[str] | None = None
        attempts = 0
        for attempt in range(max_repairs + 1):
            attempts = attempt + 1
            with timings.measure("attempt", tool, attempt=attempts):
                try:
                    with timings.measure("llm", tool, attempt=attempts):
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
                    warnings.append(
                        f"The local LLM failed during the direct flow: {exc}"
                    )
                    trace.append(f"attempt {attempts}: llm-error")
                    break

                output = LLMOutput(flow=Flow.DIRECT, attempt=attempts, mathml=raw)
                llm_outputs.append(output)

                try:
                    with timings.measure("normalize", "normalizer", attempt=attempts):
                        candidate = normalize_mathml(raw)
                        candidate_root = parse_and_normalize(candidate)
                        if prune_orphan_args(candidate_root):
                            candidate = serialize_mathml(candidate_root)
                except NormalizationError as exc:
                    feedback = [f"Invalid MathML: {exc}"]
                    trace.append(f"attempt {attempts}: normalization-error: {exc}")
                    continue

                with timings.measure("validate", "validator", attempt=attempts):
                    report = validate_mathml(
                        candidate,
                        allow_open_concepts=allow_open,
                        allowed_open_concepts=self._allowed_open(allow_open),
                        registry=self.registry,
                    )
                if report.acceptable:
                    output.accepted = True
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
                        llm_outputs=llm_outputs,
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

        fallback = self._offline(
            latex,
            context,
            ocr_backend,
            allow_open,
            timings,
            note="direct fallback after LLM attempts",
        )
        fallback.flow = Flow.DIRECT
        fallback.attempts = attempts
        fallback.warnings.extend(warnings)
        fallback.warnings.append(
            "The direct flow did not produce an acceptable result; deterministic "
            "fallback was used."
        )
        fallback.trace = trace + fallback.trace
        fallback.llm_outputs = llm_outputs
        return fallback

    # ------------------------------------------------------------------
    # flow 3: image directly -> annotated MathML via a vision model
    # ------------------------------------------------------------------
    def _vision(
        self,
        request: EquationRequest,
        allow_open: bool,
        max_repairs: int,
        *,
        flow: Flow = Flow.VISION,
        allow_fallback: bool = False,
        latex: str | None = None,
        ocr_backend: str = "none",
        timings: TimingRecorder,
    ) -> PipelineResult:
        if not request.image_path:
            raise ValueError(
                "The vision flow requires an image and does not use an "
                "intermediate LaTeX representation"
            )

        warnings: list[str] = []
        trace = [
            f"{flow.value}-flow",
            "vision-agent",
            f"model={self.vision_agent.model}",
        ]
        role = getattr(self.vision_agent, "role", "vision-mathml-generator")
        tool = f"{role}[{self.vision_agent.model}]"

        with timings.measure("vision-availability", "ollama", note=flow.value):
            available = self.vision_agent.is_available()
        if not available:
            raise RuntimeError(
                "The Ollama vision server is not reachable at "
                f"{self.vision_agent.client.base_url}"
            )
        with timings.measure(
            "vision-model-check", "ollama", note=self.vision_agent.model
        ):
            has_model = self.vision_agent.has_model()
        if not has_model:
            raise RuntimeError(
                f"Vision model {self.vision_agent.model!r} is not installed on "
                "the Ollama server; run "
                f"`ollama pull {self.vision_agent.model}`"
            )

        feedback: list[str] | None = None
        attempts = 0
        last_mathml: str | None = None
        last_report = None
        llm_outputs: list[LLMOutput] = []

        for attempt in range(max_repairs + 1):
            attempts = attempt + 1
            with timings.measure("attempt", tool, attempt=attempts):
                try:
                    with timings.measure("llm", tool, attempt=attempts):
                        raw = self.vision_agent.run(
                            request.image_path,
                            request.context,
                            feedback,
                            allow_open_concepts=allow_open,
                            allow_all_open_concepts=(
                                allow_open and self.settings.allow_all_open_concepts
                            ),
                        )
                except (LLMError, OCRError) as exc:
                    warnings.append(f"The vision LLM failed: {exc}")
                    trace.append(f"attempt {attempts}: llm-error: {exc}")
                    break

                output = LLMOutput(flow=flow, attempt=attempts, mathml=raw)
                llm_outputs.append(output)

                try:
                    with timings.measure("normalize", "normalizer", attempt=attempts):
                        candidate = normalize_mathml(raw)
                        candidate_root = parse_and_normalize(candidate)
                        if prune_orphan_args(candidate_root):
                            candidate = serialize_mathml(candidate_root)
                except NormalizationError as exc:
                    last_mathml = raw
                    with timings.measure("validate", "validator", attempt=attempts):
                        last_report = validate_mathml(
                            raw,
                            allow_open_concepts=allow_open,
                            allowed_open_concepts=self._allowed_open(allow_open),
                            registry=self.registry,
                        )
                    feedback = [f"Invalid MathML: {exc}"]
                    trace.append(f"attempt {attempts}: normalization-error: {exc}")
                    continue

                last_mathml = candidate
                with timings.measure("validate", "validator", attempt=attempts):
                    report = validate_mathml(
                        candidate,
                        allow_open_concepts=allow_open,
                        allowed_open_concepts=self._allowed_open(allow_open),
                        registry=self.registry,
                    )
                last_report = report
                if report.acceptable:
                    output.accepted = True
                    trace.append(f"attempt {attempts}: accepted")
                    return PipelineResult(
                        latex=latex or "",
                        mathml=candidate,
                        flow=flow,
                        validation=report,
                        attempts=attempts,
                        llm_used=True,
                        ocr_backend=ocr_backend if latex else "none",
                        warnings=warnings,
                        trace=trace,
                        llm_outputs=llm_outputs,
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

        if allow_fallback:
            fallback_latex = latex
            fallback_backend = ocr_backend
            if not fallback_latex:
                try:
                    fallback_latex, fallback_backend = self._resolve_latex(
                        request,
                        timings=timings,
                        note="fallback after vision attempts",
                    )
                except (ValueError, OCRError, LLMError) as exc:
                    warnings.append(f"Could not obtain a LaTeX fallback: {exc}")
                    fallback_latex = None
            if fallback_latex:
                fallback = self._offline(
                    fallback_latex,
                    request.context,
                    fallback_backend,
                    allow_open,
                    timings,
                    note="vision fallback after LLM attempts",
                )
                fallback.flow = flow
                fallback.attempts = attempts
                fallback.warnings.extend(warnings)
                fallback.warnings.append(
                    "The vision agent did not produce acceptable MathML; "
                    "deterministic fallback was used."
                )
                fallback.trace = trace + fallback.trace
                fallback.llm_outputs = llm_outputs
                return fallback

        if last_mathml is None:
            detail = warnings[-1] if warnings else "no MathML was produced"
            raise RuntimeError(
                f"The {flow.value} vision flow did not produce any MathML: {detail}"
            )

        with timings.measure("validate", "validator", note="final candidate"):
            report = last_report or validate_mathml(
                last_mathml,
                allow_open_concepts=allow_open,
                allowed_open_concepts=self._allowed_open(allow_open),
                registry=self.registry,
            )
        if allow_fallback:
            warning = (
                "The vision agent did not produce acceptable MathML and no LaTeX "
                "fallback was available; the last LLM candidate is returned."
            )
        else:
            warning = (
                "The vision flow did not produce acceptable MathML; the last LLM "
                "candidate is returned without deterministic fallback."
            )
        warnings.append(warning)
        trace.append("returning last LLM candidate")
        return PipelineResult(
            latex=latex or "",
            mathml=last_mathml,
            flow=flow,
            validation=report,
            attempts=attempts,
            llm_used=True,
            ocr_backend=ocr_backend if latex else "none",
            warnings=warnings,
            trace=trace,
            llm_outputs=llm_outputs,
        )

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
        timings: TimingRecorder,
    ) -> PipelineResult:
        warnings: list[str] = []
        trace = [
            "hybrid-flow",
            "latex2mathml",
            "semantic-json",
            f"model={self.llm.model}",
        ]
        llm_outputs: list[LLMOutput] = []
        tool = f"{self.hybrid_agent.role}[{self.llm.model}]"

        with timings.measure("latex2mathml", "latex2mathml"):
            base_markup = self.converter.convert(latex)
        with timings.measure("normalize", "normalizer", note="skeleton"):
            base_mathml = normalize_mathml(base_markup)
            base_root = parse_and_normalize(base_mathml)
        with timings.measure("clean", "semantic-cleaner", note="drop intent/arg/id"):
            strip_semantic_attributes(base_root)

        with timings.measure("llm-check", "ollama"):
            llm_ready = self._llm_ready(warnings)
        if not llm_ready:
            result = self._annotate_base(
                latex,
                base_markup,
                ocr_backend,
                allow_open,
                warnings,
                timings,
                note="LLM unavailable; deterministic fallback",
            )
            result.flow = Flow.HYBRID
            result.trace = trace + result.trace
            return result

        feedback: list[str] | None = None
        attempts = 0
        for attempt in range(max_repairs + 1):
            attempts = attempt + 1
            with timings.measure("attempt", tool, attempt=attempts):
                attempt_root = copy.deepcopy(base_root)
                with timings.measure("assign-ids", "semantic-json", attempt=attempts):
                    assign_node_ids(attempt_root)
                    semantic_tree = semantic_tree_markup(attempt_root)
                concept_cards = None
                if self.concept_rag is not None:
                    with timings.measure(
                        "concept-rag", "concept-rag", attempt=attempts
                    ):
                        concept_cards = self.concept_rag.retrieve(
                            latex,
                            attempt_root,
                            context,
                            allow_open=allow_open,
                            allowed_open_concepts=self._allowed_open(allow_open),
                            top_k=self.settings.concept_rag_top_k,
                        )
                    trace.append(
                        f"attempt {attempts}: concept-rag ({len(concept_cards)} cards)"
                    )
                    if self.open_concept_rag is not None:
                        with timings.measure(
                            "open-concept-rag",
                            "open-concept-rag",
                            attempt=attempts,
                        ):
                            open_cards = self.open_concept_rag.retrieve(
                                latex,
                                attempt_root,
                                context,
                                allow_open=allow_open,
                                allowed_open_concepts=self._allowed_open(allow_open),
                                top_k=self.settings.open_concept_rag_top_k,
                            )
                        seen = {card.name for card in concept_cards}
                        added = 0
                        for card in open_cards:
                            if card.name in seen:
                                continue
                            concept_cards.append(card)
                            seen.add(card.name)
                            added += 1
                        trace.append(
                            f"attempt {attempts}: open-concept-rag "
                            f"({len(open_cards)} cards, {added} added)"
                        )
                try:
                    with timings.measure("llm", tool, attempt=attempts):
                        raw = self.hybrid_agent.run(
                            latex,
                            semantic_tree,
                            context,
                            feedback,
                            allow_open_concepts=allow_open,
                            allow_all_open_concepts=(
                                allow_open and self.settings.allow_all_open_concepts
                            ),
                            concept_cards=concept_cards,
                        )
                except LLMError as exc:
                    warnings.append(
                        f"The local LLM failed during the hybrid flow: {exc}"
                    )
                    trace.append(f"attempt {attempts}: llm-error")
                    break

                output = LLMOutput(
                    flow=Flow.HYBRID,
                    attempt=attempts,
                    mathml=raw,
                    input_mathml=semantic_tree,
                )
                llm_outputs.append(output)

                if self._looks_like_mathml(raw):
                    with timings.measure(
                        "legacy-mathml", "hybrid-compat", attempt=attempts
                    ):
                        candidate, report, errors = self._legacy_hybrid_candidate(
                            raw, base_root, allow_open
                        )
                    if candidate is not None and report is not None:
                        output.accepted = True
                        trace.append(
                            f"attempt {attempts}: accepted legacy MathML response"
                        )
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
                            llm_outputs=llm_outputs,
                        )
                    feedback = errors or [
                        (
                            "The MathML response was not acceptable; annotate the "
                            "expression or return the semantic JSON contract."
                        )
                    ]
                    trace.append(f"attempt {attempts}: rejected legacy MathML response")
                    continue

                with timings.measure("parse-json", "semantic-json", attempt=attempts):
                    try:
                        annotations = parse_semantic_payload(raw)
                    except SemanticError as exc:
                        feedback = [str(exc)]
                        trace.append(f"attempt {attempts}: json-error: {exc}")
                        continue

                with timings.measure("apply-semantics", "annotator", attempt=attempts):
                    applied = apply_semantic_payload(
                        attempt_root,
                        annotations,
                        registry=self.registry,
                        allow_open_concepts=allow_open,
                        allowed_open_concepts=self._allowed_open(allow_open),
                    )
                if applied.warnings:
                    warnings.extend(applied.warnings)
                    trace.append(
                        f"attempt {attempts}: semantic-warning "
                        f"({len(applied.warnings)} warning(s))"
                    )
                if applied.errors:
                    feedback = applied.errors
                    trace.append(
                        f"attempt {attempts}: semantic-error "
                        f"({len(applied.errors)} problem(s))"
                    )
                    continue
                if applied.applied == 0:
                    feedback = list(applied.warnings) or [
                        (
                            "No annotations were applied; choose the semantic "
                            "intents for the expression."
                        )
                    ]
                    trace.append(f"attempt {attempts}: no annotations")
                    continue

                with timings.measure("strip-ids", "semantic-json", attempt=attempts):
                    strip_node_ids(attempt_root)
                with timings.measure("normalize", "normalizer", attempt=attempts):
                    candidate = normalize_mathml(serialize_mathml(attempt_root))
                    candidate_root = parse_and_normalize(candidate)
                    if prune_orphan_args(candidate_root):
                        candidate = serialize_mathml(candidate_root)
                with timings.measure("validate", "validator", attempt=attempts):
                    report = validate_mathml(
                        candidate,
                        allow_open_concepts=allow_open,
                        allowed_open_concepts=self._allowed_open(allow_open),
                        registry=self.registry,
                    )
                if report.acceptable:
                    output.accepted = True
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
                        llm_outputs=llm_outputs,
                    )

                feedback = report.error_messages()
                if not feedback:
                    feedback = [
                        "No intent attributes were added; annotate the expression."
                    ]
                trace.append(
                    f"attempt {attempts}: rejected ({len(report.errors)} errors, "
                    f"{report.intent_count} intents)"
                )

        fallback = self._annotate_base(
            latex,
            base_markup,
            ocr_backend,
            allow_open,
            warnings,
            timings,
            note="deterministic fallback after LLM attempts",
        )
        fallback.flow = Flow.HYBRID
        fallback.attempts = attempts
        fallback.warnings.append(
            "The semantic JSON annotation did not produce an acceptable result; "
            "deterministic annotation was used on the latex2mathml skeleton."
        )
        fallback.trace = trace + fallback.trace
        fallback.llm_outputs = llm_outputs
        return fallback

    def _looks_like_mathml(self, text: str) -> bool:
        lowered = (text or "").lower()
        return "<math" in lowered or "<mrow" in lowered

    def _legacy_hybrid_candidate(
        self,
        raw: str,
        base_root: etree._Element,
        allow_open: bool,
    ) -> tuple[str | None, ValidationReport | None, list[str]]:
        try:
            candidate = normalize_mathml(raw)
            candidate_root = parse_and_normalize(candidate)
        except NormalizationError as exc:
            return None, None, [f"Invalid MathML: {exc}"]

        strip_node_ids(candidate_root)
        if not same_structure(base_root, candidate_root):
            return (
                None,
                None,
                [
                    (
                        "The mathematical structure changed. Keep every element, "
                        "its order and its text identical; only add intent and arg "
                        "attributes."
                    )
                ],
            )

        prune_orphan_args(candidate_root)
        candidate = serialize_mathml(candidate_root)
        report = validate_mathml(
            candidate,
            allow_open_concepts=allow_open,
            allowed_open_concepts=self._allowed_open(allow_open),
            registry=self.registry,
        )
        if report.acceptable:
            return candidate, report, []

        if report.errors:
            merged_root = parse_and_normalize(candidate)
            prune_invalid_intents(
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
                return merged, merged_report, []
            return None, merged_report, merged_report.error_messages()

        return (
            None,
            report,
            ["No intent attributes were added; annotate the expression."],
        )

    def _annotate_base(
        self,
        latex: str,
        base_markup: str,
        ocr_backend: str,
        allow_open: bool,
        warnings: list[str],
        timings: TimingRecorder,
        *,
        note: str = "",
    ) -> PipelineResult:
        with timings.measure("normalize", "normalizer", note=note):
            root = parse_and_normalize(base_markup)
        with timings.measure("deterministic-annotation", "annotator", note=note):
            count = annotate_tree(root)
            prune_orphan_args(root)
            mathml = serialize_mathml(root)
        with timings.measure("validate", "validator", note=note):
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
    resolve_timings = TimingRecorder()
    resolve_started = time.perf_counter()
    latex, ocr_backend = pipeline._resolve_latex(request, timings=resolve_timings)
    resolve_elapsed = time.perf_counter() - resolve_started
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
        timings = TimingRecorder()
        timings.entries.extend(resolve_timings.entries)
        started = time.perf_counter()
        results[flow.value] = pipeline._run_flow(
            flow,
            replace(request, flow=flow),
            latex,
            ocr_backend,
            allow_open,
            max_repairs,
            timings,
        )
        results[flow.value].timings = timings.as_list()
        results[flow.value].elapsed_seconds = (
            time.perf_counter() - started + resolve_elapsed
        )
    return results
