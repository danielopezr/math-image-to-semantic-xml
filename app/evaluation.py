"""Evaluacion POR ETAPA. No existe ni se calcula una puntuacion agregada.

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

from app.models import PipelineReport, StageStatus
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


def summarize_stages(report: PipelineReport, expected_latex: Optional[str] = None) -> dict:
    """Resumen independiente por etapa (sin agregar)."""
    summary: dict = {}

    ocr = report.ocr
    summary["ocr"] = {
        "status": ocr.status.value if ocr else "SKIPPED",
        "engine": ocr.engine if ocr else None,
        "attempt": ocr.attempt if ocr else None,
        "reviewed_by_human": ocr.reviewed_by_human if ocr else None,
        "accuracy": (ocr_accuracy(ocr.latex_raw if ocr else None, expected_latex)
                     if expected_latex is not None and ocr and ocr.engine != "manual" else None),
    }

    conv = report.conversion
    summary["conversion"] = {
        "status": conv.status.value if conv else "SKIPPED",
        "error_kind": conv.error_kind.value if conv else None,
        "mathml_kind": conv.mathml_kind if conv else None,
    }

    sem = report.semantic
    summary["semantic_structure"] = {
        "status": sem.status.value if sem else "SKIPPED",
        "attempts": sem.attempts if sem else None,
    }
    summary["semantic_quality"] = {
        "measured": False,
        "note": "No hay metrica automatica fiable; consultar las reglas y la revision humana.",
        "requires_review": sem.output.requires_review if sem and sem.output else None,
    }

    by_outcome = Counter(r.outcome.value for r in report.rules)
    by_origin: dict[str, Counter] = {}
    for r in report.rules:
        by_origin.setdefault(r.origin.value, Counter())[r.outcome.value] += 1
    summary["rules"] = {
        "status": report.rules_status.value,
        "by_outcome": dict(by_outcome),
        "by_origin": {k: dict(v) for k, v in by_origin.items()},
    }

    xml = report.xml
    summary["xml"] = {
        "status": xml.status.value if xml else "SKIPPED",
        "well_formed": xml.well_formed if xml else None,
        "schema_valid": xml.schema_valid if xml else None,
        "overall": overall_status(xml.xml) if xml and xml.xml else None,
    }
    return summary
