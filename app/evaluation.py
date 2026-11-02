"""Evaluacion POR ASPECTO. No existe ni se calcula una puntuacion agregada.

Aspectos (cada uno independiente): exactitud del OCR, validez de LaTeX, validez del MathML,
respuesta estructurada del LLM, calidad estructural de los intents, coherencia semantica,
cumplimiento del catalogo, validez del XML, braille, lectura matematica y audio.

Limites:
* La exactitud del OCR solo se puede medir si se conoce el LaTeX esperado. Se
  compara con una normalizacion minima (sin espacios): formas equivalentes de
  escribir lo mismo (\\frac frente a \\dfrac) cuentan como diferencias.
* La calidad semantica NO tiene metrica automatica fiable: se informan las
  reglas y la marca de revision, y se deja la valoracion a una persona.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from app.braille import non_braille_chars
from app.converter import validate_mathml
from app.models import PipelineResult, StageStatus
from app.semantic_tree import errors_only, tree_metrics, validate_semantic_output
from app.xml_validator import overall_status


def _normalize(text: str) -> str:
    return "".join(text.split())


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def ocr_accuracy(predicted: Optional[str], expected: str) -> dict:
    """Exactitud del OCR frente a un LaTeX esperado (comparacion sin espacios)."""
    exp = _normalize(expected)
    if predicted is None:
        return {"exact_match": False, "edit_distance": None, "char_error_rate": None, "note": "El OCR no produjo salida."}
    pred = _normalize(predicted)
    distance = levenshtein(pred, exp)
    return {
        "exact_match": pred == exp,
        "edit_distance": distance,
        "char_error_rate": distance / max(1, len(exp)),
        "note": "Comparacion sin espacios; no reconoce equivalencias de LaTeX.",
    }


SEMANTIC_COHERENCE_RULES = ("SEM-002", "SEM-003", "SEM-004", "SEM-005", "SEM-006")


def _status(obj: object) -> str:
    return obj.status.value if obj is not None else "SKIPPED"


def summarize_stages(report: PipelineResult, expected_latex: Optional[str] = None) -> dict:
    """Resumen independiente por aspecto (sin agregar). Los aspectos no evaluables se marcan como tal."""
    summary: dict = {}

    ocr = report.ocr
    summary["ocr"] = {
        "status": _status(ocr),
        "engine": ocr.engine if ocr else None,
        "attempt": ocr.attempt if ocr else None,
        "reviewed_by_human": ocr.reviewed_by_human if ocr else None,
        "accuracy": (ocr_accuracy(ocr.latex_raw, expected_latex)
                     if expected_latex is not None and ocr and ocr.engine != "manual" else None),
    }

    conv = report.conversion
    if conv is None:
        latex_state = "not_evaluated"
    elif conv.status is StageStatus.OK:
        latex_state = "accepted_by_converter"     # el conversor lo acepto; no es una validacion completa de LaTeX
    elif conv.error_kind.value == "LATEX_SYNTAX":
        latex_state = "invalid"
    else:
        latex_state = "unknown"                   # fallo de conversion: no se puede concluir que el LaTeX sea invalido
    summary["latex"] = {"state": latex_state, "error_kind": conv.error_kind.value if conv else None}

    mathml_errors = validate_mathml(conv.mathml)[0] if conv and conv.mathml else []
    summary["mathml"] = {
        "status": _status(conv),
        "kind": conv.mathml_kind if conv else None,
        "structural_errors": [e.code for e in mathml_errors],
        "note": "Presentacion: no garantiza la semantica.",
    }

    sem = report.semantic
    summary["semantic_response"] = {"status": _status(sem), "attempts": sem.attempts if sem else None}

    if sem is not None and sem.output is not None:
        problems = validate_semantic_output(
            sem.output, latex=report.ocr.latex if report.ocr else None,
            mathml=conv.mathml if conv else None)
        summary["intent_structure"] = {
            "metrics": tree_metrics(sem.output),
            "errors": [p.code for p in errors_only(problems)],
            "warnings": [p.code for p in problems if p.severity == "warning"],
        }
    else:
        summary["intent_structure"] = {"status": "NOT_EVALUATED"}

    by_rule = {r.rule_id: r.outcome.value for r in report.rules}
    summary["semantic_coherence"] = {
        "rules": {rid: by_rule.get(rid, "NOT_EVALUATED") for rid in SEMANTIC_COHERENCE_RULES},
        "review_required": sem.output.review_required if sem and sem.output else None,
        "note": "No hay metrica automatica de correccion matematica; la valoracion final es humana.",
    }

    by_origin: dict[str, Counter] = {}
    for r in report.rules:
        by_origin.setdefault(r.origin.value, Counter())[r.outcome.value] += 1
    summary["catalog"] = {
        "status": report.rules_status.value,
        "catalog_version": report.rules_catalog_version,
        "by_outcome": dict(Counter(r.outcome.value for r in report.rules)),
        "by_origin": {k: dict(v) for k, v in by_origin.items()},
    }

    xml = report.xml
    summary["xml"] = {
        "status": _status(xml),
        "well_formed": xml.well_formed if xml else None,
        "schema_valid": xml.schema_valid if xml else None,
        "overall": overall_status(xml.xml) if xml and xml.xml else None,
    }

    b = report.braille
    summary["braille"] = {
        "status": _status(b),
        "code": b.braille_code if b else None,
        "language": b.language if b else None,
        "generated": bool(b and b.status is StageStatus.OK and b.text),
        "review_required": b.review_required if b else None,
        "reviewed_by_human": b.reviewed_by_human if b else None,
        "non_braille_chars": non_braille_chars(b.text) if b and b.text else None,
        "note": "Un XML valido no demuestra que la transcripcion braille sea correcta.",
    }
    s = report.speech
    summary["speech"] = {
        "status": _status(s),
        "language": s.language if s else None,
        "generated": bool(s and s.status is StageStatus.OK and s.text),
        "review_required": s.review_required if s else None,
    }
    a = report.audio
    summary["audio"] = {
        "status": _status(a),
        "provider": a.provider if a else None,
        "file_generated": bool(a and a.status is StageStatus.OK and a.file_name),
        "format": a.audio_format if a else None,
        "size_bytes": a.size_bytes if a else None,
    }
    return summary
