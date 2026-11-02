"""Etapa de braille matematico: MathML validado -> texto Unicode braille (via MathCAT).

* No se reimplementa ninguna regla de transcripcion: lo hace MathCAT.
* La salida es TEXTO braille Unicode (U+2800..U+28FF). No es un archivo BRF, ni formato de
  embosado, ni salida para una linea braille.
* Que el XML sea valido NO demuestra que la transcripcion sea matematicamente correcta: toda
  salida lleva un aviso y debe revisarla una persona con conocimiento del codigo braille.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app.converter import validate_mathml
from app.mathcat import BRAILLE_CODES, MathCAT, MathCATError, resolve_braille_code
from app.models import BrailleOutput, StageIssue, StageStatus

logger = logging.getLogger(__name__)
STAGE = "braille"
BRAILLE_BLOCK = range(0x2800, 0x2900)
_ALLOWED_WHITESPACE = {" ", "\n", "\t", "\u00a0"}


def _issue(code: str, message: str) -> StageIssue:
    return StageIssue(stage=STAGE, code=code, message=message)


def non_braille_chars(text: str) -> list[str]:
    return sorted({c for c in text if ord(c) not in BRAILLE_BLOCK and c not in _ALLOWED_WHITESPACE})


def mathml_to_braille(
    mathml: str,
    *,
    mathcat: MathCAT,
    braille_code: Optional[str] = None,
    language: Optional[str] = None,
    semantic_review_required: bool = False,
) -> BrailleOutput:
    """Convierte MathML a braille Unicode. Nunca lanza excepciones."""
    code = braille_code or mathcat.config.braille_code
    out = BrailleOutput(status=StageStatus.FAILED, braille_code=code, language=language or mathcat.config.speech_language)

    try:
        out.mathcat_code = resolve_braille_code(code)
    except MathCATError as exc:
        out.errors.append(_issue(exc.code, exc.message))
        return out

    errors, _ = validate_mathml(mathml or "")
    if errors:
        out.errors.append(_issue("INVALID_MATHML", "El MathML no es valido; no se envia a MathCAT: " + "; ".join(e.code for e in errors)))
        return out

    try:
        text = mathcat.to_braille(mathml, braille_code=code, language=out.language)
        out.mathcat_version = mathcat.version
    except MathCATError as exc:
        out.errors.append(_issue(exc.code, exc.message))
        return out
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fallo inesperado de MathCAT (braille)")
        out.errors.append(_issue("MATHCAT_UNEXPECTED", f"{type(exc).__name__}: {exc}"))
        return out

    if not text or not text.strip():
        out.errors.append(_issue("EMPTY_BRAILLE", "MathCAT no devolvio braille (expresion posiblemente no compatible)."))
        return out

    out.text_unicode = text
    out.status = StageStatus.OK
    foreign = non_braille_chars(text)
    out.review_required = bool(semantic_review_required or foreign)
    if foreign:
        out.warnings.append(_issue(
            "NON_BRAILLE_CHARS",
            "La salida contiene caracteres que no son celdas braille (" + " ".join(foreign) +
            "): puede haber simbolos no compatibles con este codigo."))
    if semantic_review_required:
        out.warnings.append(_issue("SEMANTIC_REVIEW_PENDING", "La interpretacion semantica requiere revision; revisa tambien el braille."))
    out.warnings.append(_issue(
        "UNVERIFIED_TRANSCRIPTION",
        "La transcripcion no esta verificada: un XML valido no demuestra que sea correcta. Requiere revision humana experta."))
    return out


def apply_braille_review(output: BrailleOutput, corrected: str) -> BrailleOutput:
    """Registra una correccion humana sin tocar el texto original de MathCAT."""
    if not corrected or not corrected.strip():
        raise ValueError("El braille corregido no puede estar vacio.")
    updated = output.model_copy(deep=True)
    updated.text_unicode_reviewed = corrected
    updated.reviewed_by_human = True
    foreign = non_braille_chars(corrected)
    if foreign:
        updated.warnings.append(_issue("NON_BRAILLE_CHARS", "El texto corregido contiene caracteres no braille: " + " ".join(foreign)))
    if updated.status is not StageStatus.OK:
        updated.status = StageStatus.OK
        updated.warnings.append(_issue("MANUAL_OVERRIDE", "El braille proviene de entrada manual, no de MathCAT."))
    return updated


def export_braille_text(output: BrailleOutput, path: str | Path) -> Path:
    """Guarda el braille como TEXTO Unicode UTF-8. Rechaza .brf: ese formato NO esta implementado."""
    target = Path(path)
    if target.suffix.lower() in {".brf", ".brl"}:
        raise ValueError(
            "No se genera BRF: el resultado es texto braille Unicode, no un archivo de embosado. "
            "Usa una extension .txt (o .brl.txt).")
    text = output.text
    if not text:
        raise ValueError("No hay braille que exportar.")
    target.write_text(text + "\n", encoding="utf-8")
    return target


SUPPORTED_BRAILLE_CODES = tuple(BRAILLE_CODES)
