"""Etapa 1: OCR matematico con pix2tex.

Principios:
* El LaTeX original (``latex_raw``) se conserva siempre sin modificar.
* El OCR puede fallar o ser incorrecto: se devuelven errores, no excepciones.
* pix2tex se importa de forma perezosa para poder probar sin instalarlo.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from app.models import OcrResult, StageIssue, StageStatus

logger = logging.getLogger(__name__)
STAGE = "ocr"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

# Un "motor" recibe una imagen PIL y devuelve LaTeX (str).
Engine = Callable[[object], str]
_cached_engine: Optional[Engine] = None


def _issue(code: str, message: str) -> StageIssue:
    return StageIssue(stage=STAGE, code=code, message=message)


def _load_default_engine() -> Engine:
    """Carga pix2tex una sola vez (la primera vez descarga los pesos)."""
    global _cached_engine
    if _cached_engine is None:
        from pix2tex.cli import LatexOCR  # import perezoso

        model = LatexOCR()
        _cached_engine = lambda img: model(img)  # noqa: E731
    return _cached_engine


def _pix2tex_version() -> Optional[str]:
    try:
        from importlib.metadata import version

        return version("pix2tex")
    except Exception:  # noqa: BLE001
        return None


def recognize_formula(
    image_path: str | Path,
    *,
    engine: Optional[Engine] = None,
    attempt: int = 1,
) -> OcrResult:
    """Reconoce una formula en ``image_path`` y devuelve un ``OcrResult``.

    ``engine`` permite inyectar un motor alternativo (pruebas, otro OCR).
    ``attempt`` identifica el numero de intento al repetir el reconocimiento.
    """
    path = Path(image_path)
    result = OcrResult(
        image_ref=str(path),
        status=StageStatus.FAILED,
        attempt=attempt,
        engine="custom" if engine else "pix2tex",
        engine_version=None if engine else _pix2tex_version(),
    )

    if not path.is_file():
        result.errors.append(_issue("IMAGE_NOT_FOUND", f"No existe el archivo: {path}"))
        return result
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        result.errors.append(
            _issue("UNSUPPORTED_FORMAT", f"Extension no soportada: {path.suffix or '(ninguna)'}")
        )
        return result

    try:
        from PIL import Image

        with Image.open(path) as im:
            image = im.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        result.errors.append(_issue("IMAGE_UNREADABLE", f"No se pudo abrir la imagen: {exc}"))
        return result

    try:
        run = engine or _load_default_engine()
    except ImportError as exc:
        result.errors.append(
            _issue("ENGINE_NOT_INSTALLED", f"pix2tex no esta instalado o falla al importarse: {exc}")
        )
        return result
    except Exception as exc:  # noqa: BLE001
        result.errors.append(_issue("ENGINE_INIT_FAILED", f"No se pudo iniciar el OCR: {exc}"))
        return result

    try:
        latex = run(image)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fallo del motor OCR")
        result.errors.append(_issue("OCR_RUNTIME_ERROR", f"Error durante el reconocimiento: {exc}"))
        return result

    if not isinstance(latex, str) or not latex.strip():
        result.errors.append(_issue("EMPTY_OUTPUT", "El OCR no devolvio ningun LaTeX."))
        return result

    result.latex_raw = latex
    result.status = StageStatus.OK
    result.warnings.append(
        _issue("UNVERIFIED", "El OCR no garantiza exactitud: se recomienda revisar el LaTeX.")
    )
    return result


def apply_manual_review(result: OcrResult, corrected_latex: str) -> OcrResult:
    """Registra una correccion manual sin tocar ``latex_raw``."""
    if not corrected_latex.strip():
        raise ValueError("El LaTeX corregido no puede estar vacio.")
    updated = result.model_copy(deep=True)
    updated.latex_reviewed = corrected_latex
    updated.reviewed_by_human = True
    if updated.status is not StageStatus.OK:
        # Una correccion manual permite continuar aunque el OCR haya fallado.
        updated.status = StageStatus.OK
        updated.warnings.append(
            _issue("MANUAL_OVERRIDE", "El LaTeX proviene de entrada manual, no del OCR.")
        )
    return updated


def retry_recognition(previous: OcrResult, *, engine: Optional[Engine] = None) -> OcrResult:
    """Repite el reconocimiento sobre la misma imagen incrementando el intento."""
    return recognize_formula(previous.image_ref, engine=engine, attempt=previous.attempt + 1)
