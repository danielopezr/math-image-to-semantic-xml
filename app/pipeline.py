"""Orquestacion: imagen -> OCR -> LaTeX -> MathML -> interpretacion semantica -> reglas ->
enriquecimiento (opcional) -> MathCAT (braille, lectura) -> audio (opcional) -> XML -> validacion.

Reglas de diseno:
* Cada etapa devuelve su propio resultado; aqui solo se encadenan. CLI y futura web deben usar
  exclusivamente ``run_pipeline``.
* Dependencias REALES entre etapas (no es una cadena lineal):
    converter  <- ocr
    semantic   <- converter
    rules      <- converter (usa la interpretacion si existe)
    enrichment <- converter + semantic (opcional, desactivado por defecto)
    braille    <- converter (usa siempre el MathML ORIGINAL)
    speech     <- converter (usa el MathML enriquecido solo si MathCAT lo acepto)
    audio      <- speech (+ proveedor TTS)
* Si una etapa falla, sus dependientes se omiten (SKIPPED) con el motivo; los resultados previos se conservan.
* Etapas OBLIGATORIAS: ocr, converter, semantic_agent, rules, y braille/speech si se solicitaron.
  audio y enrichment son opcionales: omitirlos no impide VALIDATED, pero un fallo en cualquiera si.
* ``report.validated`` es True solo si el XML supera XSD y coherencia Y declara VALIDATED.
* Cada resultado intermedio se guarda en <output_dir>/<run_id>/ para depuracion.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Union

from app.audio import TTSProvider, mathml_to_speech, synthesize_audio
from app.braille import apply_braille_review, mathml_to_braille
from app.converter import Converter, latex_to_mathml, validate_mathml
from app.enrichment import enrich_mathml
from app.intent_rules import CatalogError, evaluate_intents, load_catalog
from app.mathcat import MathCAT, load_configuration
from app.models import (
    BrailleOutput, ConversionResult, EnrichmentResult, OcrResult, PipelineResult, StageIssue, StageResult, StageStatus, XmlResult,
)
from app.ocr import Engine, apply_manual_review, recognize_formula, retry_recognition
from app.semantic_agent import SemanticProvider, analyze_semantics
from app.xml_builder import ALL_STAGES, CORE_STAGES, build_xml_document
from app.xml_validator import overall_status

logger = logging.getLogger(__name__)


class ReviewRetry:
    """Valor que un revisor devuelve para pedir otro intento de OCR."""

    def __repr__(self) -> str:
        return "RETRY"


RETRY = ReviewRetry()
# Un revisor recibe el OcrResult y devuelve: None (aceptar), un str (LaTeX corregido) o RETRY.
Reviewer = Callable[[OcrResult], Union[None, str, ReviewRetry]]
# Un revisor de braille recibe el BrailleOutput y devuelve None (aceptar) o el texto corregido.
BrailleReviewer = Callable[[BrailleOutput], Optional[str]]


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def _save_text(out_dir: Optional[Path], name: str, text: str) -> None:
    if out_dir is None:
        return
    try:
        (out_dir / name).write_text(text, encoding="utf-8")
    except OSError as exc:
        logger.warning("No se pudo guardar %s: %s", name, exc)


def _save_model(out_dir: Optional[Path], name: str, model: object, **kwargs: object) -> None:
    if model is not None:
        _save_text(out_dir, name, model.model_dump_json(indent=2, **kwargs))


def run_pipeline(
    *,
    image_path: Optional[Union[str, Path]] = None,
    latex: Optional[str] = None,
    mathml: Optional[str] = None,
    provider: Optional[SemanticProvider] = None,
    ocr_engine: Optional[Engine] = None,
    converter: Optional[Converter] = None,
    semantic_analyzer: Callable = analyze_semantics,
    reviewer: Optional[Reviewer] = None,
    output_dir: Union[str, Path] = "outputs",
    run_id: Optional[str] = None,
    max_ocr_attempts: int = 3,
    catalog: Optional[dict] = None,
    # --- accesibilidad
    mathcat: Optional[MathCAT] = None,
    braille: bool = False,
    braille_code: Optional[str] = None,
    braille_reviewer: Optional[BrailleReviewer] = None,
    speech: bool = False,
    speech_language: Optional[str] = None,
    speech_style: Optional[str] = None,
    speech_verbosity: Optional[str] = None,
    audio: bool = False,
    tts_provider: Optional[TTSProvider] = None,
    audio_path: Optional[Union[str, Path]] = None,
    voice_id: Optional[str] = None,
    voice_rate: Optional[int] = None,
    enrich_intents: bool = False,
) -> PipelineResult:
    """Ejecuta el pipeline completo y devuelve un ``PipelineResult``.

    Con ``latex`` y sin imagen se omite el OCR (entrada manual). Con ``mathml`` se omiten OCR y
    conversion (el resultado nunca podra quedar VALIDATED: faltan etapas obligatorias). Sin ``provider``
    se omite la etapa semantica. ``braille``/``speech``/``audio`` activan las etapas de accesibilidad; ``audio`` implica ``speech``.
    """
    if image_path is None and latex is None and mathml is None:
        raise ValueError("Indica image_path, latex o mathml.")

    run_id = run_id or _new_run_id()
    report = PipelineResult(run_id=run_id, image_ref=Path(image_path).name if image_path else None)

    out_dir: Optional[Path] = Path(output_dir) / run_id
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        report.output_dir = str(out_dir)
    except OSError as exc:
        logger.warning("No se pudo crear %s: %s", out_dir, exc)
        out_dir = None

    file_handler: Optional[logging.Handler] = None
    if out_dir is not None:
        file_handler = logging.FileHandler(out_dir / "pipeline.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger("app").addHandler(file_handler)

    speech = speech or audio
    owns_mathcat = mathcat is None and (braille or speech)
    mc = MathCAT(load_configuration()) if owns_mathcat else mathcat
    try:
        _run_stages(report, out_dir, locals_=dict(
            image_path=image_path, latex=latex, mathml=mathml, provider=provider, ocr_engine=ocr_engine, converter=converter,
            semantic_analyzer=semantic_analyzer, reviewer=reviewer, max_ocr_attempts=max_ocr_attempts,
            catalog=catalog, mc=mc, braille=braille, braille_code=braille_code, braille_reviewer=braille_reviewer,
            speech=speech, speech_language=speech_language, speech_style=speech_style,
            speech_verbosity=speech_verbosity, audio=audio, tts_provider=tts_provider, audio_path=audio_path,
            voice_id=voice_id, voice_rate=voice_rate, enrich_intents=enrich_intents))
    finally:
        if owns_mathcat and mc is not None:
            mc.close()
        if file_handler is not None:
            logging.getLogger("app").removeHandler(file_handler)
            file_handler.close()
    return report


def _stage_summary(report: PipelineResult, required: set[str]) -> list[StageResult]:
    objects = {
        "ocr": report.ocr, "converter": report.conversion, "semantic_agent": report.semantic,
        "enrichment": report.enrichment, "braille": report.braille, "speech": report.speech, "audio": report.audio,
    }
    summary: list[StageResult] = []
    for name in ALL_STAGES:
        obj = objects.get(name)
        status = report.rules_status if name == "rules" else (obj.status if obj is not None else StageStatus.SKIPPED)
        review = False
        if name == "semantic_agent" and obj is not None and obj.output is not None:
            review = obj.output.review_required
        elif name in {"braille", "speech", "audio"} and obj is not None:
            review = obj.review_required
        summary.append(StageResult(
            stage=name, status=status, required=name in required, review_required=review,
            skipped_reason=report.skipped.get(name) if status is StageStatus.SKIPPED else None,
            error_count=len(getattr(obj, "errors", []) or []), warning_count=len(getattr(obj, "warnings", []) or [])))
    if report.xml is not None:
        summary.append(StageResult(stage="xml", status=report.xml.status, required=True,
                                   error_count=len(report.xml.errors), warning_count=len(report.xml.warnings)))
    return summary


def _run_stages(report: PipelineResult, out_dir: Optional[Path], locals_: dict) -> None:
    g = locals_.get
    image_path, latex, mc = g("image_path"), g("latex"), g("mc")

    ocr = None
    provided_mathml = g("mathml")
    if provided_mathml is not None:
        # Entrada MathML directa: no hay OCR ni conversion (etapas obligatorias -> nunca VALIDATED).
        report.skipped["ocr"] = "Se proporciono MathML directamente."
        report.skipped["converter"] = "Se proporciono MathML directamente."
        errors, warnings = validate_mathml(provided_mathml)
        report.conversion = ConversionResult(
            latex=latex or "", status=StageStatus.FAILED if errors else StageStatus.OK,
            mathml=provided_mathml, converter="provided", errors=errors, warnings=warnings)
        _save_model(out_dir, "conversion.json", report.conversion)
    else:
        # ---- Etapa 1: OCR (o entrada manual)
        if image_path is None:
            base = OcrResult(image_ref="(manual)", status=StageStatus.SKIPPED, engine="manual", attempt=0)
            ocr = apply_manual_review(base, latex)
        else:
            ocr = recognize_formula(image_path, engine=g("ocr_engine"))
            if latex is not None and latex.strip():
                ocr = apply_manual_review(ocr, latex)

        reviewer = g("reviewer")
        if reviewer is not None:
            attempts = 1
            while True:
                action = reviewer(ocr)
                if isinstance(action, ReviewRetry):
                    if image_path is None or attempts >= g("max_ocr_attempts"):
                        logger.warning("No se puede repetir el OCR (intentos=%s).", attempts)
                        break
                    ocr = retry_recognition(ocr, engine=g("ocr_engine"))
                    attempts += 1
                    continue
                if isinstance(action, str) and action.strip():
                    ocr = apply_manual_review(ocr, action)
                break
        report.ocr = ocr
        _save_model(out_dir, "ocr.json", ocr)

        # ---- Etapa 2: conversion a MathML
        if ocr.status is StageStatus.OK and ocr.latex:
            report.conversion = latex_to_mathml(ocr.latex, converter=g("converter"))
            _save_model(out_dir, "conversion.json", report.conversion)
        else:
            report.skipped["converter"] = "El OCR no produjo LaTeX utilizable."
    conversion_ok = report.conversion is not None and report.conversion.status is StageStatus.OK
    original_mathml = report.conversion.mathml if conversion_ok else None
    latex_text = (ocr.latex if ocr is not None else latex) or ""

    # ---- Etapa 3: interpretacion semantica
    if g("provider") is None:
        report.skipped["semantic_agent"] = "Ejecucion sin proveedor LLM."
    elif not conversion_ok:
        report.skipped["semantic_agent"] = "No hay MathML valido que interpretar."
    else:
        report.semantic = g("semantic_analyzer")(latex_text, original_mathml, g("provider"))
        _save_model(out_dir, "semantic.json", report.semantic)
    semantic_ok = report.semantic is not None and report.semantic.status is StageStatus.OK
    semantic_review = bool(semantic_ok and report.semantic.output.review_required)

    # ---- Etapa 5: evaluacion de intents (reglas)
    if report.conversion is not None and report.conversion.mathml:
        try:
            cat = g("catalog") or load_catalog()
            report.rules = evaluate_intents(report.conversion.mathml, report.semantic, catalog=cat)
            report.rules_catalog_version = cat["catalog_version"]
            report.rules_status = StageStatus.OK
            _save_text(out_dir, "rules.json",
                       json.dumps([json.loads(r.model_dump_json()) for r in report.rules], indent=2, ensure_ascii=False))
        except CatalogError as exc:
            report.rules_status = StageStatus.FAILED
            report.skipped["rules"] = f"Catalogo de reglas invalido: {exc}"
    else:
        report.skipped["rules"] = "No hay MathML sobre el que evaluar reglas."

    # ---- Enriquecimiento semantico (opcional, desactivado por defecto)
    speech_mathml = original_mathml
    if not g("enrich_intents"):
        report.skipped["enrichment"] = "Desactivado (opcional)."
    elif not (conversion_ok and semantic_ok):
        report.enrichment = EnrichmentResult(status=StageStatus.SKIPPED, enabled=True)
        report.skipped["enrichment"] = "Requiere MathML valido e interpretacion semantica valida."
    else:
        report.enrichment = enrich_mathml(original_mathml, report.semantic.output, mathcat=mc)
        if report.enrichment.status is StageStatus.OK and report.enrichment.mathml:
            speech_mathml = report.enrichment.mathml
        elif report.enrichment.status is StageStatus.SKIPPED:
            report.skipped["enrichment"] = "El enriquecimiento no se aplico (ver advertencias)."
        _save_model(out_dir, "enrichment.json", report.enrichment)

    required = set(CORE_STAGES)

    # ---- MathCAT: braille
    if not g("braille"):
        report.skipped["braille"] = "No solicitada."
    elif not conversion_ok:
        required.add("braille")
        report.skipped["braille"] = "No hay MathML valido."
    else:
        required.add("braille")
        report.braille = mathml_to_braille(original_mathml, mathcat=mc, braille_code=g("braille_code"),
                                           semantic_review_required=semantic_review)
        reviewer_b = g("braille_reviewer")
        if reviewer_b is not None and report.braille.status is StageStatus.OK:
            corrected = reviewer_b(report.braille)
            if isinstance(corrected, str) and corrected.strip():
                report.braille = apply_braille_review(report.braille, corrected)
        _save_model(out_dir, "braille.json", report.braille)

    # ---- MathCAT: lectura matematica (texto)
    if not g("speech"):
        report.skipped["speech"] = "No solicitada."
    elif not conversion_ok:
        required.add("speech")
        report.skipped["speech"] = "No hay MathML valido."
    else:
        required.add("speech")
        report.speech = mathml_to_speech(speech_mathml, mathcat=mc, language=g("speech_language"),
                                         style=g("speech_style"), verbosity=g("speech_verbosity"),
                                         semantic_review_required=semantic_review)
        if speech_mathml is not original_mathml and report.speech.status is StageStatus.OK:
            report.speech.warnings.append(StageIssue(stage="speech", code="ENRICHED_MATHML_USED",
                                                     message="La lectura uso el MathML enriquecido con intent."))
        _save_model(out_dir, "speech.json", report.speech)

    # ---- Sintesis de audio (opcional; depende de la lectura)
    if not g("audio"):
        report.skipped["audio"] = "No solicitada."
    elif report.speech is None or report.speech.status is not StageStatus.OK:
        report.skipped["audio"] = "No hay texto de lectura que sintetizar."
    else:
        tts = g("tts_provider")
        fmt = getattr(tts, "output_format", "wav")
        target = g("audio_path") or ((out_dir or Path(".")) / f"lectura.{fmt}")
        report.audio = synthesize_audio(report.speech, tts, target, voice_id=g("voice_id"), rate=g("voice_rate"))
        if report.audio.status is StageStatus.SKIPPED:
            report.skipped["audio"] = "Sin proveedor TTS configurado: solo hay texto de lectura."
        _save_model(out_dir, "audio.json", report.audio)

    # ---- XML (al final, para incluir todos los resultados) y validacion
    try:
        report.xml = build_xml_document(
            run_id=report.run_id, image_ref=report.image_ref, ocr=report.ocr, conversion=report.conversion,
            semantic=report.semantic, rules=report.rules, rules_catalog_version=report.rules_catalog_version,
            rules_status=report.rules_status, enrichment=report.enrichment, braille=report.braille,
            speech=report.speech, audio=report.audio, required_stages=required)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fallo inesperado al construir el XML")
        report.xml = XmlResult(
            status=StageStatus.FAILED,
            errors=[StageIssue(stage="xml_builder", code="BUILD_FAILED", message=f"{type(exc).__name__}: {exc}")])

    if report.xml.xml is not None:
        name = "result.xml" if report.xml.status is StageStatus.OK else "result.INVALID.xml"
        _save_text(out_dir, name, report.xml.xml.decode("utf-8"))

    report.validated = (
        report.xml.status is StageStatus.OK
        and report.xml.xml is not None
        and overall_status(report.xml.xml) == "VALIDATED"
    )
    report.stages = _stage_summary(report, required)
    _save_model(out_dir, "report.json", report, exclude={"xml": {"xml"}})
