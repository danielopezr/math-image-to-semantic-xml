"""Agentic orchestrator for equation recognition and intent annotation."""

from __future__ import annotations

import copy
import os
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

from lxml import etree

from .agents import DirectMathMLAgent, HybridIntentAgent, VisionMathMLAgent
from .config import Settings
from .llm.base import LLMClient, LLMError, current_request_timeout, request_timeout
from .llm.factory import build_llm
from .mathml.annotator import (
    annotate_tree,
    enforce_derivatives,
    enforce_structures,
    strip_default_intents,
)
from .mathml.calculus import enforce_partial_derivatives
from .mathml.concept_rag import ConceptRAG
from .mathml.concepts import ConceptRegistry, get_registry
from .mathml.converter import LatexToMathML
from .mathml.dom import element_children, is_element
from .mathml.grouping import group_operator_scopes
from .mathml.latex import clean_latex
from .mathml.normalizer import (
    NormalizationError,
    normalize_mathml,
    parse_and_normalize,
    serialize_mathml,
)
from .mathml.open_concept_rag import OpenConceptRAG
from .mathml.semantic import (
    ID_ATTR,
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
from .ocr.factory import build_ocr, canonical_backend_name
from .ocr.preprocess import PreprocessConfig
from .timing import TimingRecorder

# Recognised LaTeX kept per (backend, image, mtime) so repeated runs of the
# same image (evaluation retries, compare) do not repeat the OCR.
_OCR_CACHE_SIZE = 256


_CALCULUS_CUES = (
    "calculo",
    "calculus",
    "ecuaciones diferenciales",
    "ecuacion diferencial",
    "differential",
    "derivad",
    "derivative",
)
_INNER_PRODUCT_CUES = (
    "producto interno",
    "producto escalar",
    "inner product",
    "ortogonal",
    "orthogonal",
    "ortonormal",
    "orthonormal",
    "gram-schmidt",
    "gram schmidt",
)


def _fold(text: str) -> str:
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


_ORIGINAL_ATTR = "data-original-index"

# Concepts with an unambiguous notation that the deterministic rules detect
# (primes, ∂, ∇, ‖·‖, |·|, ⟨·,·⟩, n!, …, tables, mixed numbers).  With the
# rules as the base, the rules decide them: when they did not annotate one,
# the notation is not there, so the LLM may not add it (measured 18/11/2026:
# the LLM added derivative to f'(a; y), inner-product to u·v, ...).
_RULE_OWNED_CONCEPTS = frozenset(
    {
        "absolute-value",
        "curl",
        "derivative",
        "determinant",
        "divergence",
        "ellipsis",
        "factorial",
        "gradient",
        "inner-product",
        "matrix",
        "mixed-fraction",
        "norm",
        "partial-derivative",
        "transpose",
        "vector",
    }
)


def _split_rule_owned(
    annotations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Separate the LLM annotations whose concept the rules decide."""

    kept: list[dict[str, Any]] = []
    owned: list[str] = []
    for index, item in enumerate(annotations, 1):
        concept = str(item.get("concept") or "").strip()
        if concept in _RULE_OWNED_CONCEPTS:
            owned.append(
                f"annotation {index}: {concept!r} is decided by the deterministic rules"
            )
        else:
            kept.append(item)
    return kept, owned


def _skeleton_with_llm_ids(
    skeleton_root: etree._Element,
) -> tuple[etree._Element, list[tuple[etree._Element, str]], dict[str, list[str]]]:
    """Copy of the ungrouped skeleton and the ids the LLM saw on it.

    The LLM sees the skeleton after operator grouping, which only adds
    ``<mrow>`` wrappers.  Grouping a marked copy and numbering it exactly as
    the hybrid flow does tells which id each original element got.  The ids
    are returned apart (element, id) instead of set as attributes, because the
    rules must run on a tree without them: an extra attribute changes some of
    their decisions (y''' read as a power).  Also returns, for every wrapper
    id, its children's ids.
    """

    root = copy.deepcopy(skeleton_root)
    originals = [element for element in root.iter() if is_element(element)]
    for index, element in enumerate(originals):
        element.set(_ORIGINAL_ATTR, str(index))
    grouped = copy.deepcopy(root)
    group_operator_scopes(grouped)
    assign_node_ids(grouped)
    by_original: dict[str, str] = {}
    wrappers: dict[str, list[str]] = {}
    for element in grouped.iter():
        if not is_element(element):
            continue
        node_id = str(element.get(ID_ATTR))
        original = element.get(_ORIGINAL_ATTR)
        if original is None:
            wrappers[node_id] = [
                str(child.get(ID_ATTR)) for child in element_children(element)
            ]
        else:
            by_original[original] = node_id
    assignments: list[tuple[etree._Element, str]] = []
    for element in originals:
        assignments.append((element, by_original[str(element.get(_ORIGINAL_ATTR))]))
        del element.attrib[_ORIGINAL_ATTR]
    return root, assignments, wrappers


def _expand_wrapper_ids(
    annotation: dict[str, Any], wrappers: dict[str, list[str]]
) -> dict[str, Any]:
    """Rewrite references to grouping wrappers as groups of their children.

    The wrappers do not exist in the ungrouped skeleton: a wrapper target
    becomes a ``wrap`` of its children (keeping the wrapper id for later
    references) and a wrapper argument becomes a group.
    """

    if not wrappers:
        return annotation

    def flat(node_id: str) -> list[str]:
        if node_id not in wrappers:
            return [node_id]
        return [leaf for child in wrappers[node_id] for leaf in flat(child)]

    def expand(value: Any) -> Any:
        if isinstance(value, str) and value in wrappers:
            return flat(value)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return [leaf for item in value for leaf in flat(item)]
        return value

    result = dict(annotation)
    target = result.get("target")
    if isinstance(target, str) and target in wrappers and "wrap" not in result:
        del result["target"]
        result["wrap"] = flat(target)
        result["new_id"] = target
    if "wrap" in result:
        result["wrap"] = expand(result["wrap"])
    for key in ("args", "groups"):
        value = result.get(key)
        if isinstance(value, dict):
            result[key] = {slot: expand(item) for slot, item in value.items()}
        elif isinstance(value, list):
            result[key] = [expand(item) for item in value]
    return result


def _cached_timing(tool: str, note: str):
    from .timing import PhaseTiming

    return PhaseTiming(
        phase="ocr",
        tool=tool,
        duration_ms=0.0,
        note=(f"{note}; " if note else "") + "cached result",
    )


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
        self.registry = registry or get_registry(self.settings.nary_associative)
        self.concept_rag = (
            ConceptRAG(self.registry) if self.settings.concept_rag else None
        )
        self.open_concept_rag = (
            OpenConceptRAG(self.registry) if self.settings.open_concept_rag else None
        )
        self.direct_agent = DirectMathMLAgent(self.llm, self.settings)
        self.hybrid_agent = HybridIntentAgent(self.llm, self.settings)
        self.vision_agent = VisionMathMLAgent(self.settings)
        # OCR engines are expensive to load (texify, pix2text...): build each
        # backend once per pipeline and reuse it.
        self._ocr_engines: dict[str, OCREngine | None] = {}
        self._ocr_latex: OrderedDict[tuple[str, str, float], str] = OrderedDict()

    def _finish(
        self,
        result: PipelineResult,
        started: float,
        timings: TimingRecorder,
    ) -> PipelineResult:
        result.timings = timings.as_list()
        result.elapsed_seconds = time.perf_counter() - started
        result.warnings = _dedupe(result.warnings)
        if not self.settings.annotate_default_structures and result.mathml:
            try:
                root = parse_and_normalize(result.mathml)
            except NormalizationError:
                return result
            removed = strip_default_intents(root)
            if removed:
                prune_orphan_args(root)
                result.mathml = serialize_mathml(root)
                result.validation = validate_mathml(
                    result.mathml,
                    allow_open_concepts=self.settings.allow_open_concepts,
                    allowed_open_concepts=self._allowed_open(
                        self.settings.allow_open_concepts
                    ),
                    registry=self.registry,
                )
                result.trace.append(f"removed {removed} default-reading intent(s)")
        return result

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def run(self, request: EquationRequest) -> PipelineResult:
        started = time.perf_counter()
        timings = TimingRecorder()
        self._run_context = request.context
        with self._request_timeout(request.timeout):
            result = self._run_request(request, timings)
        return self._finish(result, started, timings)

    @contextmanager
    def _request_timeout(self, timeout: float | None) -> Iterator[None]:
        """Apply ``timeout`` to every LLM, vision and OCR request of one run.

        The value travels in a context variable read by the HTTP clients, so
        the pipeline, its settings and its clients are never mutated and
        concurrent runs in other threads keep their own timeout.
        """

        with request_timeout(timeout):
            yield

    @property
    def _active_timeout(self) -> float | None:
        return current_request_timeout()

    def _run_request(
        self,
        request: EquationRequest,
        timings: TimingRecorder,
    ) -> PipelineResult:
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
                return self._vision(request, allow_open, max_repairs, timings=timings)

        if request.flow is Flow.DIRECT and self._direct_vision_enabled(request):
            with timings.group("direct"):
                return self._vision(
                    request,
                    allow_open,
                    max_repairs,
                    flow=Flow.DIRECT,
                    allow_fallback=True,
                    timings=timings,
                )

        latex, ocr_backend = self._resolve_latex(request, timings=timings)
        try:
            return self._run_latex(
                request, latex, ocr_backend, allow_open, max_repairs, timings
            )
        except Exception as exc:
            # Keep what the OCR read, so a failed sample can still be replayed
            # or turned into a semantic-only dataset.
            if getattr(exc, "latex", None) is None:
                exc.latex = latex  # type: ignore[attr-defined]
            raise

    def _run_latex(
        self,
        request: EquationRequest,
        latex: str,
        ocr_backend: str,
        allow_open: bool,
        max_repairs: int,
        timings: TimingRecorder,
    ) -> PipelineResult:
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
            if self._accepts(hybrid.validation):
                hybrid.trace.insert(
                    0, "auto: hybrid accepted; skipped the slower direct flow"
                )
                return hybrid

            direct = self._run_flow(
                Flow.DIRECT,
                request,
                latex,
                ocr_backend,
                allow_open,
                max_repairs,
                timings,
            )
            if self._accepts(direct.validation):
                result = direct
                chosen = "direct"
            else:
                result = hybrid if hybrid.validation.ok else direct
                chosen = result.flow.value
            result.llm_outputs = hybrid.llm_outputs + direct.llm_outputs
            result.trace.insert(
                0, f"auto: evaluated hybrid and direct; selected {chosen}"
            )
            return result

        return self._run_flow(
            request.flow,
            request,
            latex,
            ocr_backend,
            allow_open,
            max_repairs,
            timings,
        )

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
            engine = self._ocr_engine(backend, recorder, note)
        if engine is None:
            raise ValueError(
                "An image was provided but OCR is disabled. Install a supported "
                "OCR backend (texify, pix2tex, rapid_latex_ocr, pix2text or "
                "paddleocr) or pass the recognised LaTeX directly."
            )

        cache_key = self._ocr_cache_key(engine, request.image_path)
        cached = self._ocr_latex.get(cache_key) if cache_key is not None else None
        if cached is not None:
            self._ocr_latex.move_to_end(cache_key)  # type: ignore[arg-type]
            recorder.entries.append(
                replace(
                    _cached_timing(engine.name, note),
                    group=recorder._group,
                )
            )
            return cached, engine.name

        with recorder.measure("ocr", engine.name, note=note):
            raw_latex = engine.image_to_latex(request.image_path)
        with recorder.measure("latex-clean", "clean_latex", note=note):
            latex = clean_latex(raw_latex)
        if not latex:
            raise ValueError("The OCR backend returned an empty LaTeX string")
        if cache_key is not None:
            self._ocr_latex[cache_key] = latex
            while len(self._ocr_latex) > _OCR_CACHE_SIZE:
                self._ocr_latex.popitem(last=False)
        return latex, engine.name

    def _ocr_options(self) -> dict[str, object]:
        """Settings forwarded to the ``ollama_vision`` OCR backend."""

        return {
            "model": self.settings.ollama_vision_model,
            "base_url": self.settings.vision_base_url,
            "timeout": self.settings.ollama_vision_timeout,
            "num_ctx": self.settings.ollama_vision_num_ctx,
            "num_predict": self.settings.ollama_vision_num_predict,
            "max_side": self.settings.ollama_vision_max_side,
            "stream": self.settings.ollama_stream,
            "think": self.settings.ollama_think,
            "retries": self.settings.llm_retries,
        }

    def _ocr_engine(
        self, backend: str, recorder: TimingRecorder, note: str
    ) -> OCREngine | None:
        key = canonical_backend_name(backend)
        if key in self._ocr_engines:
            return self._ocr_engines[key]
        with recorder.measure("ocr-setup", backend, note=note):
            engine = build_ocr(
                backend,
                preprocess=PreprocessConfig.from_env(
                    enabled=self.settings.ocr_preprocess,
                    profile=self.settings.ocr_preprocess_profile,
                ),
                ollama_options=self._ocr_options(),
            )
        self._ocr_engines[key] = engine
        return engine

    @staticmethod
    def _ocr_cache_key(
        engine: OCREngine, image_path: str
    ) -> tuple[str, str, float] | None:
        try:
            return (
                engine.name,
                os.path.abspath(image_path),
                os.path.getmtime(image_path),
            )
        except OSError:
            return None

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
            count = annotate_tree(root, **self._annotator_options())
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

        with timings.measure("llm-check", self.settings.llm_backend):
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
                    output.rejection = list(feedback)
                    trace.append(f"attempt {attempts}: normalization-error: {exc}")
                    continue

                with timings.measure("validate", "validator", attempt=attempts):
                    report = validate_mathml(
                        candidate,
                        allow_open_concepts=allow_open,
                        allowed_open_concepts=self._allowed_open(allow_open),
                        registry=self.registry,
                    )
                if self._accepts(report):
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
                output.rejection = list(feedback)
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
        unavailable_reason: str | None = None
        if not available:
            unavailable_reason = (
                "The Ollama vision server is not reachable at "
                f"{self.vision_agent.client.base_url}"
            )
        else:
            with timings.measure(
                "vision-model-check", "ollama", note=self.vision_agent.model
            ):
                has_model = self.vision_agent.has_model()
            if not has_model:
                unavailable_reason = (
                    f"Vision model {self.vision_agent.model!r} is not installed on "
                    "the Ollama server; run "
                    f"`ollama pull {self.vision_agent.model}`"
                )
        if unavailable_reason is not None:
            if not allow_fallback:
                raise RuntimeError(unavailable_reason)
            # The direct flow promises a deterministic fallback: use it instead
            # of failing when the vision model cannot run at all.
            warnings.append(unavailable_reason)
            trace.append("vision unavailable; skipping LLM attempts")

        feedback: list[str] | None = None
        attempts = 0
        last_mathml: str | None = None
        last_report = None
        llm_outputs: list[LLMOutput] = []

        vision_attempts = 0 if unavailable_reason is not None else max_repairs + 1
        for attempt in range(vision_attempts):
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
                    output.rejection = list(feedback)
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
                if self._accepts(report):
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
                output.rejection = list(feedback)
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
        # The rules work best on the ungrouped skeleton (the offline flow's
        # input); rules-first maps the LLM ids back onto it.
        skeleton_root = copy.deepcopy(base_root)
        with timings.measure("group", "operator-grouping"):
            groups = group_operator_scopes(base_root)
        if groups:
            trace.append(f"operator-grouping ({groups} group(s))")

        with timings.measure("llm-check", self.settings.llm_backend):
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

        # The candidate concepts depend only on the expression and its context,
        # so they are retrieved once and reused by every repair attempt.
        concept_cards = self._retrieve_concept_cards(
            latex, base_root, context, allow_open, timings, trace
        )

        feedback: list[str] | None = None
        attempts = 0
        best_partial = None
        for attempt in range(max_repairs + 1):
            attempts = attempt + 1
            with timings.measure("attempt", tool, attempt=attempts):
                attempt_root = copy.deepcopy(base_root)
                with timings.measure("assign-ids", "semantic-json", attempt=attempts):
                    assign_node_ids(attempt_root)
                    semantic_tree = semantic_tree_markup(attempt_root)
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
                    output.rejection = list(feedback)
                    trace.append(f"attempt {attempts}: rejected legacy MathML response")
                    continue

                with timings.measure("parse-json", "semantic-json", attempt=attempts):
                    try:
                        annotations = parse_semantic_payload(raw)
                    except SemanticError as exc:
                        feedback = [str(exc)]
                        output.rejection = list(feedback)
                        trace.append(f"attempt {attempts}: json-error: {exc}")
                        continue
                if self.settings.hybrid_base == "rules":
                    # The rules decide these concepts anyway; a malformed one
                    # (partial-derivative with two arguments, 181 times on
                    # 18/11) must not sink the rest of the answer.
                    annotations, owned = _split_rule_owned(annotations)
                    output.discarded = owned

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
                    output.rejection = list(feedback)
                    trace.append(
                        f"attempt {attempts}: semantic-error "
                        f"({len(applied.errors)} problem(s))"
                    )
                    if applied.applied:
                        # The invalid annotations were rolled back; keep the
                        # valid part in case no later attempt is fully valid.
                        partial = self._partial_candidate(attempt_root, allow_open)
                        if partial is not None and (
                            best_partial is None
                            or partial[1].intent_count > best_partial[1].intent_count
                        ):
                            best_partial = (
                                partial[0],
                                partial[1],
                                attempts,
                                output,
                                len(applied.errors),
                                annotations,
                            )
                    continue
                if applied.applied == 0 and self.settings.annotate_default_structures:
                    feedback = list(applied.warnings) or [
                        (
                            "No annotations were applied; choose the semantic "
                            "intents for the expression."
                        )
                    ]
                    output.rejection = list(feedback)
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
                if self._accepts(report):
                    output.accepted = True
                    output.discarded += applied.warnings
                    trace.append(f"attempt {attempts}: accepted")
                    with timings.measure(
                        "deterministic-fill", "annotator", attempt=attempts
                    ):
                        candidate, report = self._combine_with_rules(
                            candidate,
                            report,
                            skeleton_root,
                            annotations,
                            output,
                            allow_open,
                            trace,
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

                feedback = report.error_messages()
                if not feedback:
                    feedback = [
                        "No intent attributes were added; annotate the expression."
                    ]
                output.rejection = list(feedback)
                trace.append(
                    f"attempt {attempts}: rejected ({len(report.errors)} errors, "
                    f"{report.intent_count} intents)"
                )

        if best_partial is not None:
            (
                candidate,
                report,
                partial_attempt,
                output,
                dropped,
                annotations,
            ) = best_partial
            output.accepted = True
            output.discarded += output.rejection
            output.rejection = []
            trace.append(
                f"attempt {partial_attempt}: accepted the valid annotations "
                f"({dropped} invalid one(s) discarded)"
            )
            with timings.measure("deterministic-fill", "annotator"):
                candidate, report = self._combine_with_rules(
                    candidate,
                    report,
                    skeleton_root,
                    annotations,
                    output,
                    allow_open,
                    trace,
                )
            warnings.append(
                f"{dropped} invalid LLM annotation(s) were discarded; the valid "
                "ones were kept."
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

    def _retrieve_concept_cards(
        self,
        latex: str,
        root: etree._Element,
        context: MathContext,
        allow_open: bool,
        timings: TimingRecorder,
        trace: list[str],
    ):
        if self.concept_rag is None:
            return None
        allowed = self._allowed_open(allow_open)
        with timings.measure("concept-rag", "concept-rag"):
            concept_cards = self.concept_rag.retrieve(
                latex,
                root,
                context,
                allow_open=allow_open,
                allowed_open_concepts=allowed,
                top_k=self.settings.concept_rag_top_k,
            )
        trace.append(f"concept-rag ({len(concept_cards)} cards)")
        if self.open_concept_rag is None:
            return concept_cards
        with timings.measure("open-concept-rag", "open-concept-rag"):
            open_cards = self.open_concept_rag.retrieve(
                latex,
                root,
                context,
                allow_open=allow_open,
                allowed_open_concepts=allowed,
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
        trace.append(f"open-concept-rag ({len(open_cards)} cards, {added} added)")
        return concept_cards

    def _partial_candidate(
        self, attempt_root: etree._Element, allow_open: bool
    ) -> tuple[str, ValidationReport] | None:
        """Validate the annotations that applied cleanly in a failed attempt."""

        root = copy.deepcopy(attempt_root)
        strip_node_ids(root)
        try:
            candidate = normalize_mathml(serialize_mathml(root))
            candidate_root = parse_and_normalize(candidate)
        except NormalizationError:
            return None
        if prune_orphan_args(candidate_root):
            candidate = serialize_mathml(candidate_root)
        report = validate_mathml(
            candidate,
            allow_open_concepts=allow_open,
            allowed_open_concepts=self._allowed_open(allow_open),
            registry=self.registry,
        )
        if not report.acceptable:
            return None
        return candidate, report

    def _accepts(self, report: ValidationReport) -> bool:
        """Whether a validated candidate can be returned as the result.

        When plain fractions, subscripts and roots are annotated, every useful
        result has at least one intent, so an intent-less candidate means the
        model did nothing.  Under the annotation guide (defaults not
        annotated) an expression such as ``a/b`` written with ``\\frac``
        legitimately has no intent, so a valid document is enough.
        """

        if self.settings.annotate_default_structures:
            return report.acceptable
        return report.ok

    def _annotator_options(self) -> dict[str, Any]:
        """Annotator options from the settings and the current run context."""

        context = getattr(self, "_run_context", None) or MathContext()
        text = _fold(
            " ".join((context.area, context.topic, context.surrounding_text[:2000]))
        )
        topic = _fold(" ".join((context.topic, context.surrounding_text[:2000])))

        def decide(mode: str, keywords: tuple[str, ...], haystack: str) -> bool:
            if mode == "on":
                return True
            if mode == "off":
                return False
            return any(keyword in haystack for keyword in keywords)

        primes = decide(self.settings.derivative_primes, _CALCULUS_CUES, text)
        return {
            "implicit_times": self.settings.annotate_implicit_times,
            "nary_associative": self.registry.nary_associative,
            "default_structures": self.settings.annotate_default_structures,
            "derivative_variable": self.settings.derivative_variable
            if primes
            else None,
            "inner_product_pairs": decide(
                self.settings.inner_product_pairs, _INNER_PRODUCT_CUES, topic
            ),
        }

    def _combine_with_rules(
        self,
        candidate: str,
        report: ValidationReport,
        skeleton_root: etree._Element,
        annotations: list[dict[str, Any]],
        output: LLMOutput,
        allow_open: bool,
        trace: list[str],
    ) -> tuple[str, ValidationReport]:
        """Combine an accepted LLM answer with the deterministic annotator.

        With ``hybrid_base="rules"`` the deterministic annotation is the base
        and the LLM only adds what the rules leave out; otherwise the LLM
        answer is the base and the rules correct and complete it.
        ``skeleton_root`` is the clean skeleton before operator grouping.
        """

        if self.settings.hybrid_base == "rules":
            combined = self._rules_first(skeleton_root, annotations, output, allow_open)
            if combined is not None:
                candidate, report, kept = combined
                trace.append(
                    f"rules-first: {kept} LLM annotation(s) added to the rules, "
                    f"{len(annotations) - kept} discarded"
                )
                return candidate, report
            rules_only = self._rules_first(skeleton_root, [], output, allow_open)
            if rules_only is not None:
                output.discarded = _dedupe(
                    list(output.discarded)
                    + ["the LLM annotations made the rules' annotation invalid"]
                )
                trace.append("rules-first: invalid combination; rules kept")
                return rules_only[0], rules_only[1]
            trace.append("rules-first: invalid combination; LLM answer kept")
        candidate, report, enforced = self._enforce_rules(candidate, report, allow_open)
        candidate, report, added = self._fill_unannotated(candidate, report, allow_open)
        if enforced:
            trace.append(f"deterministic rules ({enforced} corrected)")
        if added:
            trace.append(f"deterministic fill (+{added} intent(s))")
        return candidate, report

    def _rules_first(
        self,
        skeleton_root: etree._Element,
        annotations: list[dict[str, Any]],
        output: LLMOutput,
        allow_open: bool,
    ) -> tuple[str, ValidationReport, int] | None:
        """Apply the LLM annotations on top of the deterministic annotation.

        The rules annotate the ungrouped skeleton (as in the offline flow,
        where they work best), which first gets the temporary ids the LLM saw
        on the grouped tree.  Then every LLM annotation is applied as a
        transaction: one that targets an element the rules already annotated,
        or that would change an argument the rules set, is rolled back.
        ``None`` when the combination does not validate.
        """

        root, assignments, wrappers = _skeleton_with_llm_ids(skeleton_root)
        annotations, owned = _split_rule_owned(annotations)
        annotations = [_expand_wrapper_ids(item, wrappers) for item in annotations]
        annotate_tree(root, **self._annotator_options())
        # The ids go on after the rules (see _skeleton_with_llm_ids); an
        # element the rules replaced keeps no id, so the LLM cannot target it.
        for element, node_id in assignments:
            if element.getroottree().getroot() is root:
                element.set(ID_ATTR, node_id)
        applied = apply_semantic_payload(
            root,
            annotations,
            registry=self.registry,
            allow_open_concepts=allow_open,
            allowed_open_concepts=self._allowed_open(allow_open),
        )
        strip_node_ids(root)
        try:
            candidate = normalize_mathml(serialize_mathml(root))
            candidate_root = parse_and_normalize(candidate)
        except NormalizationError:
            return None
        if prune_orphan_args(candidate_root):
            candidate = serialize_mathml(candidate_root)
        report = validate_mathml(
            candidate,
            allow_open_concepts=allow_open,
            allowed_open_concepts=self._allowed_open(allow_open),
            registry=self.registry,
        )
        if not self._accepts(report):
            return None
        candidate, report, _ = self._enforce_rules(candidate, report, allow_open)
        output.discarded = _dedupe(
            list(output.discarded) + owned + applied.errors + applied.warnings
        )
        return candidate, report, applied.applied

    def _fill_unannotated(
        self,
        candidate: str,
        report: ValidationReport,
        allow_open: bool,
    ) -> tuple[str, ValidationReport, int]:
        """Complete an accepted LLM annotation with the deterministic annotator.

        Small models often annotate only part of an expression (the fraction
        but not f(x) or e^{-x}).  The deterministic annotator never touches an
        element that already has an intent, so it only fills the gaps.  The
        filled version is kept only if it still validates; otherwise the LLM
        result is returned unchanged.
        """

        root = parse_and_normalize(candidate)
        annotate_tree(root, **self._annotator_options())
        prune_orphan_args(root)
        filled = serialize_mathml(root)
        filled_report = validate_mathml(
            filled,
            allow_open_concepts=allow_open,
            allowed_open_concepts=self._allowed_open(allow_open),
            registry=self.registry,
        )
        added = filled_report.intent_count - report.intent_count
        if added <= 0 or not filled_report.acceptable:
            return candidate, report, 0
        return filled, filled_report, added

    def _enforce_rules(
        self,
        candidate: str,
        report: ValidationReport,
        allow_open: bool,
    ) -> tuple[str, ValidationReport, int]:
        """Override the LLM where an unambiguous deterministic rule applies.

        Matrices, column vectors, ``|A|`` determinants and mixed numbers
        always; primes and ``y^{(n)}`` derivatives in a calculus context.  The
        LLM often reads them otherwise (``y^{(4)}`` as a power, ``3 1/2`` as a
        product).  The corrected version is kept only if it still validates;
        otherwise the LLM result is returned unchanged.
        """

        root = parse_and_normalize(candidate)
        changed = enforce_structures(root) + enforce_partial_derivatives(root)
        variable = self._annotator_options()["derivative_variable"]
        if variable:
            changed += enforce_derivatives(root, variable)
        if not changed:
            return candidate, report, 0
        prune_orphan_args(root)
        enforced = serialize_mathml(root)
        enforced_report = validate_mathml(
            enforced,
            allow_open_concepts=allow_open,
            allowed_open_concepts=self._allowed_open(allow_open),
            registry=self.registry,
        )
        if not enforced_report.acceptable:
            return candidate, report, 0
        return enforced, enforced_report, changed

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
        if self._accepts(report):
            return candidate, report, []

        if report.errors:
            merged_root = parse_and_normalize(candidate)
            prune_invalid_intents(
                merged_root,
                allow_open_concepts=allow_open,
                allowed_open_concepts=self._allowed_open(allow_open),
                registry=self.registry,
            )
            annotate_tree(merged_root, **self._annotator_options())
            prune_orphan_args(merged_root)
            merged = serialize_mathml(merged_root)
            merged_report = validate_mathml(
                merged,
                allow_open_concepts=allow_open,
                allowed_open_concepts=self._allowed_open(allow_open),
                registry=self.registry,
            )
            if self._accepts(merged_report):
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
            count = annotate_tree(root, **self._annotator_options())
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
