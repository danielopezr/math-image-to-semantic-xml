"""Etapa 2: conversion LaTeX -> MathML con latex2mathml.

Alcance (importante):
* latex2mathml genera MathML de PRESENTACION. Describe como se dibuja la
  formula, no que significa. Este modulo nunca afirma que se conserve la semantica.
* Distingue errores de sintaxis LaTeX de errores de conversion. La clasificacion
  es heuristica: una excepcion definida en ``latex2mathml.exceptions`` se trata
  como error de sintaxis; cualquier otra, como error de conversion.
* El MathML resultante se comprueba: buena formacion, raiz ``<math>`` en el
  espacio de nombres MathML y solo elementos de presentacion.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Callable, Optional

from app.models import (
    ConversionErrorKind,
    ConversionResult,
    StageIssue,
    StageStatus,
)

logger = logging.getLogger(__name__)
STAGE = "converter"
MATHML_NS = "http://www.w3.org/1998/Math/MathML"

# Elementos de presentacion (MathML 3 / Core) aceptados por el proyecto.
PRESENTATION_ELEMENTS = frozenset({
    "math", "mi", "mn", "mo", "mtext", "mspace", "ms", "mglyph",
    "mrow", "mfrac", "msqrt", "mroot", "mstyle", "mpadded", "mphantom",
    "mfenced", "menclose", "msub", "msup", "msubsup", "munder", "mover",
    "munderover", "mmultiscripts", "mprescripts", "none",
    "mtable", "mtr", "mtd", "mlabeledtr", "maligngroup", "malignmark",
    "maction", "semantics", "annotation", "annotation-xml",
})
# Elementos tipicos de MathML de contenido: fuera del alcance de esta etapa.
CONTENT_ELEMENTS = frozenset({
    "apply", "ci", "cn", "csymbol", "bvar", "lambda", "interval", "set", "list",
})

Converter = Callable[[str], str]


def _issue(code: str, message: str) -> StageIssue:
    return StageIssue(stage=STAGE, code=code, message=message)


def _default_converter() -> Converter:
    from latex2mathml.converter import convert  # import perezoso

    return convert


def _library_version() -> Optional[str]:
    try:
        from importlib.metadata import version

        return version("latex2mathml")
    except Exception:  # noqa: BLE001
        return None


def _classify(exc: Exception) -> ConversionErrorKind:
    module = type(exc).__module__ or ""
    if module.startswith("latex2mathml.exceptions"):
        return ConversionErrorKind.LATEX_SYNTAX
    return ConversionErrorKind.CONVERSION


def validate_mathml(mathml: str) -> tuple[list[StageIssue], list[StageIssue]]:
    """Comprueba un MathML de presentacion. Devuelve (errores, advertencias).

    Usa ElementTree de la libreria estandar solo para esta comprobacion rapida;
    la validacion definitiva del documento final usa lxml con un parser endurecido.
    """
    errors: list[StageIssue] = []
    warnings: list[StageIssue] = []
    try:
        root = ET.fromstring(mathml)
    except ET.ParseError as exc:
        errors.append(_issue("MATHML_NOT_WELL_FORMED", f"MathML mal formado: {exc}"))
        return errors, warnings

    if root.tag != f"{{{MATHML_NS}}}math":
        errors.append(
            _issue("BAD_ROOT", f"La raiz debe ser <math> en el espacio de nombres MathML; es {root.tag!r}.")
        )

    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        if not el.tag.startswith(f"{{{MATHML_NS}}}"):
            errors.append(_issue("FOREIGN_ELEMENT", f"Elemento fuera del espacio de nombres MathML: {el.tag!r}."))
            continue
        local = el.tag.split("}", 1)[1]
        if local in CONTENT_ELEMENTS:
            errors.append(_issue("CONTENT_MATHML", f"<{local}> es MathML de contenido; el proyecto exige presentacion."))
        elif local == "merror":
            errors.append(_issue("MERROR_PRESENT", "El MathML contiene <merror>: el conversor marco un error."))
        elif local not in PRESENTATION_ELEMENTS:
            warnings.append(_issue("UNKNOWN_ELEMENT", f"Elemento no reconocido como presentacion: <{local}>."))
    return errors, warnings


def latex_to_mathml(latex: str, *, converter: Optional[Converter] = None) -> ConversionResult:
    """Convierte ``latex`` a MathML de presentacion. Nunca lanza excepciones."""
    result = ConversionResult(
        latex=latex,
        status=StageStatus.FAILED,
        converter="custom" if converter else "latex2mathml",
        converter_version=None if converter else _library_version(),
    )

    if not isinstance(latex, str) or not latex.strip():
        result.error_kind = ConversionErrorKind.LATEX_SYNTAX
        result.errors.append(_issue("EMPTY_INPUT", "No hay LaTeX que convertir."))
        return result

    try:
        run = converter or _default_converter()
    except ImportError as exc:
        result.error_kind = ConversionErrorKind.CONVERSION
        result.errors.append(_issue("ENGINE_NOT_INSTALLED", f"latex2mathml no esta instalado: {exc}"))
        return result

    try:
        mathml = run(latex)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Fallo de conversion", exc_info=True)
        result.error_kind = _classify(exc)
        result.errors.append(_issue(type(exc).__name__, str(exc) or "Error de conversion sin mensaje."))
        return result

    if not isinstance(mathml, str) or not mathml.strip():
        result.error_kind = ConversionErrorKind.CONVERSION
        result.errors.append(_issue("EMPTY_OUTPUT", "El conversor no devolvio MathML."))
        return result

    errors, warnings = validate_mathml(mathml)
    result.mathml = mathml
    result.warnings.extend(warnings)
    result.warnings.append(
        _issue(
            "PRESENTATION_ONLY",
            "MathML de presentacion: representa el aspecto de la formula, no garantiza su semantica.",
        )
    )
    if errors:
        result.error_kind = ConversionErrorKind.CONVERSION
        result.errors.extend(errors)
        return result

    result.status = StageStatus.OK
    return result
