"""Orquestacion: imagen -> OCR -> MathML -> LLM -> reglas -> XML -> informe.

Reglas de diseno:
* Cada etapa devuelve su propio resultado; aqui solo se encadenan.
* Si una etapa falla o no puede ejecutarse, las dependientes se omiten
  (SKIPPED) con el motivo registrado; nunca se aparenta que estan validadas.
* Cada resultado intermedio se guarda en <output_dir>/<run_id>/ para depuracion.
* ``report.validated`` es True solo si todas las etapas estan OK y el XML supera
  XSD y comprobaciones de coherencia.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Union

from app.converter import Converter, latex_to_mathml
from app.intent_rules import CatalogError, evaluate_intents, load_catalog
from app.models import OcrResult, PipelineReport, StageStatus, XmlResult, StageIssue
from app.ocr import Engine, apply_manual_review, recognize_formula, retry_recognition
from app.semantic_agent import SemanticProvider, analyze_semantics
from app.xml_builder import build_xml_document
from app.xml_validator import overall_status

logger = logging.getLogger(__name__)


class ReviewRetry:
    """Valor que un revisor devuelve para pedir otro intento de OCR."""

    def __repr__(self) -> str:
        return "RETRY"


RETRY = ReviewRetry()
# Un revisor recibe el OcrResult y devuelve: None (aceptar), un str (LaTeX corregido) o RETRY.
Reviewer = Callable[[OcrResult], Union[None, str, ReviewRetry]]


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
    provider: Optional[SemanticProvider] = None,
    ocr_engine: Optional[Engine] = None,
    converter: Optional[Converter] = None,
    semantic_analyzer: Callable = analyze_semantics,
    reviewer: Optional[Reviewer] = None,
    output_dir: Union[str, Path] = "outputs",
    run_id: Optional[str] = None,
    max_ocr_attempts: int = 3,
    catalog: Optional[dict] = None,
) -> PipelineReport:
    """Ejecuta el pipeline completo y devuelve un ``PipelineReport``.

    Con ``latex`` y sin imagen se omite el OCR (entrada manual). Sin ``provider``
    se omite la etapa semantica: el OCR y la conversion funcionan sin LLM.
    """
    if image_path is None and latex is None:
        raise ValueError("Indica image_path o latex.")

    run_id = run_id or _new_run_id()
    report = PipelineReport(run_id=run_id, image_ref=Path(image_path).name if image_path else None)

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

    try:
        _run_stages(report, out_dir, image_path, latex, provider, ocr_engine, converter,
                    semantic_analyzer, reviewer, max_ocr_attempts, catalog)
    finally:
        if file_handler is not None:
            logging.getLogger("app").removeHandler(file_handler)
            file_handler.close()
    return report


def _run_stages(report, out_dir, image_path, latex, provider, ocr_engine, converter,
                semantic_analyzer, reviewer, max_ocr_attempts, catalog) -> None:
    # ---- Etapa 1: OCR (o entrada manual)
    if image_path is None:
        base = OcrResult(image_ref="(manual)", status=StageStatus.SKIPPED, engine="manual", attempt=0)
        ocr = apply_manual_review(base, latex)
    else:
        ocr = recognize_formula(image_path, engine=ocr_engine)
        if latex is not None and latex.strip():
            ocr = apply_manual_review(ocr, latex)

    if reviewer is not None:
        attempts = 1
        while True:
            action = reviewer(ocr)
            if isinstance(action, ReviewRetry):
                if image_path is None or attempts >= max_ocr_attempts:
                    logger.warning("No se puede repetir el OCR (intentos=%s).", attempts)
                    break
                ocr = retry_recognition(ocr, engine=ocr_engine)
                attempts += 1
                continue
            if isinstance(action, str) and action.strip():
                ocr = apply_manual_review(ocr, action)
            break
    report.ocr = ocr
    _save_model(out_dir, "ocr.json", ocr)

    # ---- Etapa 2: conversion a MathML
    if ocr.status is StageStatus.OK and ocr.latex:
        report.conversion = latex_to_mathml(ocr.latex, converter=converter)
        _save_model(out_dir, "conversion.json", report.conversion)
    else:
        report.skipped["converter"] = "El OCR no produjo LaTeX utilizable."

    # ---- Etapa 3: agente semantico
    conversion_ok = report.conversion is not None and report.conversion.status is StageStatus.OK
    if provider is None:
        report.skipped["semantic_agent"] = "Ejecucion sin proveedor LLM."
    elif not conversion_ok:
        report.skipped["semantic_agent"] = "No hay MathML valido que interpretar."
    else:
        report.semantic = semantic_analyzer(ocr.latex, report.conversion.mathml, provider)
        _save_model(out_dir, "semantic.json", report.semantic)

    # ---- Etapa 5: reglas de intencion
    if report.conversion is not None and report.conversion.mathml:
        try:
            cat = catalog or load_catalog()
            report.rules = evaluate_intents(report.conversion.mathml, report.semantic, catalog=cat)
            report.rules_catalog_version = cat["catalog_version"]
            report.rules_status = StageStatus.OK
            _save_text(out_dir, "rules.json",
                       json.dumps([json.loads(r.model_dump_json()) for r in report.rules],
                                  indent=2, ensure_ascii=False))
        except CatalogError as exc:
            report.rules_status = StageStatus.FAILED
            report.skipped["rules"] = f"Catalogo de reglas invalido: {exc}"
    else:
        report.skipped["rules"] = "No hay MathML sobre el que evaluar reglas."

    # ---- Etapa 4: XML (despues de las reglas para incluirlas) y validacion
    try:
        report.xml = build_xml_document(
            run_id=report.run_id,
            image_ref=report.image_ref,
            ocr=report.ocr,
            conversion=report.conversion,
            semantic=report.semantic,
            rules=report.rules,
            rules_catalog_version=report.rules_catalog_version,
            rules_status=report.rules_status,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fallo inesperado al construir el XML")
        report.xml = XmlResult(
            status=StageStatus.FAILED,
            errors=[StageIssue(stage="xml_builder", code="BUILD_FAILED", message=f"{type(exc).__name__}: {exc}")],
        )

    if report.xml.xml is not None:
        name = "result.xml" if report.xml.status is StageStatus.OK else "result.INVALID.xml"
        _save_text(out_dir, name, report.xml.xml.decode("utf-8"))

    report.validated = (
        report.xml.status is StageStatus.OK
        and report.xml.xml is not None
        and overall_status(report.xml.xml) == "VALIDATED"
    )
    _save_model(out_dir, "report.json", report, exclude={"xml": {"xml"}})
