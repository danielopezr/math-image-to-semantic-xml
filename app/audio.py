"""Lectura matematica y audio.

Tres cosas distintas, con estados independientes:
  1. Texto de lectura matematica  -> lo genera MathCAT (``mathml_to_speech``).
  2. Sintesis de voz              -> la hace un proveedor TTS externo (``synthesize_audio``).
  3. Reproduccion de audio        -> opcional (``play_audio_file``).

MathCAT NO genera archivos de audio: solo texto. Un ``AudioOutput`` solo es OK si el archivo
existe y su cabecera se ha verificado; si no hay proveedor TTS el estado es SKIPPED.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable, Optional, Protocol

from app.converter import validate_mathml
from app.mathcat import MathCAT, MathCATError
from app.models import AudioOutput, SpeechOutput, StageIssue, StageStatus

logger = logging.getLogger(__name__)
STAGE_SPEECH = "speech"
STAGE_AUDIO = "audio"


class TTSError(Exception):
    """Fallo de un proveedor TTS, con codigo estable."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class TTSProvider(Protocol):
    name: str
    output_format: str  # extension del archivo que produce (p. ej. "wav")

    def synthesize(self, text: str, *, language: str, output_path: Path,
                   voice_id: Optional[str] = None, rate: Optional[int] = None) -> Optional[str]:
        """Escribe el audio en ``output_path`` y devuelve el identificador de voz usado (si lo sabe)."""
        ...


# ------------------------------------------------------------------ texto de lectura
def mathml_to_speech(
    mathml: str,
    *,
    mathcat: MathCAT,
    language: Optional[str] = None,
    style: Optional[str] = None,
    verbosity: Optional[str] = None,
    semantic_review_required: bool = False,
) -> SpeechOutput:
    """Genera el texto de lectura matematica con MathCAT. Nunca lanza excepciones."""
    out = SpeechOutput(
        status=StageStatus.FAILED,
        language=language or mathcat.config.speech_language,
        style=style or mathcat.config.speech_style,
        verbosity=verbosity or mathcat.config.verbosity,
    )
    errors, _ = validate_mathml(mathml or "")
    if errors:
        out.errors.append(StageIssue(stage=STAGE_SPEECH, code="INVALID_MATHML",
                                     message="El MathML no es valido; no se envia a MathCAT: " + "; ".join(e.code for e in errors)))
        return out
    try:
        text = mathcat.to_speech(mathml, language=out.language, style=out.style, verbosity=out.verbosity)
        out.mathcat_version = mathcat.version
    except MathCATError as exc:
        out.errors.append(StageIssue(stage=STAGE_SPEECH, code=exc.code, message=exc.message))
        return out
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fallo inesperado de MathCAT (lectura)")
        out.errors.append(StageIssue(stage=STAGE_SPEECH, code="MATHCAT_UNEXPECTED", message=f"{type(exc).__name__}: {exc}"))
        return out
    if not text or not text.strip():
        out.errors.append(StageIssue(stage=STAGE_SPEECH, code="EMPTY_SPEECH", message="MathCAT no devolvio texto de lectura."))
        return out
    out.text = text.strip()
    out.status = StageStatus.OK
    out.review_required = semantic_review_required
    if semantic_review_required:
        out.warnings.append(StageIssue(stage=STAGE_SPEECH, code="SEMANTIC_REVIEW_PENDING",
                                       message="La interpretacion semantica requiere revision; revisa tambien la lectura."))
    return out


# ------------------------------------------------------------------------- TTS
_LANGUAGE_NAMES = {
    "es": ("spanish", "español", "espanol"), "en": ("english",), "fr": ("french", "français", "francais"),
    "de": ("german", "deutsch"), "pt": ("portuguese", "português", "portugues"), "it": ("italian", "italiano"),
}


def voice_matches_language(voice: object, language: str) -> bool:
    """Heuristica: busca el idioma en voice.languages, en el nombre y en el id de la voz."""
    lang = language.split("-")[0].lower()
    for item in getattr(voice, "languages", None) or []:
        text = item.decode("utf-8", "ignore") if isinstance(item, bytes) else str(item)
        text = text.lstrip("\x00\x01\x02\x03\x04\x05 ").lower().replace("_", "-")
        if text.startswith(lang):
            return True
    haystack = f"{getattr(voice, 'id', '')} {getattr(voice, 'name', '')}".lower().replace("_", "-")
    if any(n in haystack for n in _LANGUAGE_NAMES.get(lang, ())):
        return True
    return f"{lang}-" in haystack


class Pyttsx3Provider:
    """TTS local con pyttsx3 (en Windows usa SAPI5, que debe tener una voz del idioma instalada)."""

    name = "pyttsx3"
    output_format = "wav"

    def __init__(self, engine_factory: Optional[Callable[[], object]] = None):
        self._factory = engine_factory

    def _make_engine(self) -> object:
        if self._factory is not None:
            return self._factory()
        try:
            import pyttsx3  # import perezoso
        except ImportError as exc:
            raise TTSError("TTS_NOT_INSTALLED", f"pyttsx3 no esta instalado: {exc}") from exc
        try:
            return pyttsx3.init()
        except Exception as exc:  # noqa: BLE001
            raise TTSError("TTS_INIT_FAILED", f"No se pudo iniciar pyttsx3: {exc}") from exc

    def synthesize(self, text: str, *, language: str, output_path: Path,
                   voice_id: Optional[str] = None, rate: Optional[int] = None) -> Optional[str]:
        engine = self._make_engine()
        try:
            voices = list(engine.getProperty("voices") or [])
            chosen = None
            if voice_id:
                chosen = next((v for v in voices if getattr(v, "id", None) == voice_id), None)
                if chosen is None:
                    raise TTSError("VOICE_NOT_FOUND", "La voz indicada no esta instalada.")
            else:
                chosen = next((v for v in voices if voice_matches_language(v, language)), None)
                if chosen is None:
                    raise TTSError("NO_VOICE_FOR_LANGUAGE",
                                   f"No hay ninguna voz instalada para '{language}'. Instala una o indica voice_id.")
            engine.setProperty("voice", chosen.id)
            if rate is not None:
                engine.setProperty("rate", int(rate))
            engine.save_to_file(text, str(output_path))
            engine.runAndWait()
            return getattr(chosen, "id", None)
        except TTSError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise TTSError("TTS_RUNTIME_ERROR", f"{type(exc).__name__}: {exc}") from exc


def _verify_audio_file(path: Path, audio_format: str) -> Optional[str]:
    if not path.is_file():
        return "El proveedor no genero ningun archivo."
    if path.stat().st_size == 0:
        return "El archivo generado esta vacio."
    if audio_format == "wav":
        with path.open("rb") as fh:
            header = fh.read(12)
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            return "El archivo no tiene cabecera WAV valida."
    return None


def synthesize_audio(
    speech: SpeechOutput,
    provider: Optional[TTSProvider],
    output_path: str | Path,
    *,
    voice_id: Optional[str] = None,
    rate: Optional[int] = None,
) -> AudioOutput:
    """Sintetiza el texto de lectura. Solo devuelve OK si el archivo existe y es valido."""
    out = AudioOutput(status=StageStatus.FAILED, language=speech.language,
                      provider=getattr(provider, "name", None), audio_format=getattr(provider, "output_format", None))
    if provider is None:
        out.status = StageStatus.SKIPPED
        out.provider = None
        out.warnings.append(StageIssue(stage=STAGE_AUDIO, code="NO_TTS_PROVIDER",
                                       message="No hay proveedor TTS configurado: solo se dispone del texto de lectura, no de audio."))
        return out
    if speech.status is not StageStatus.OK or not speech.text:
        out.errors.append(StageIssue(stage=STAGE_AUDIO, code="NO_SPEECH_TEXT", message="No hay texto de lectura que sintetizar."))
        return out

    path = Path(output_path)
    if path.suffix.lower() != f".{provider.output_format}":
        out.errors.append(StageIssue(stage=STAGE_AUDIO, code="BAD_EXTENSION",
                                     message=f"{provider.name} produce .{provider.output_format}; el archivo debe tener esa extension."))
        return out
    try:
        if path.is_dir():
            raise OSError("la ruta es una carpeta")
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        out.errors.append(StageIssue(stage=STAGE_AUDIO, code="BAD_OUTPUT_PATH", message=f"Ruta de salida no utilizable: {exc}"))
        return out

    try:
        # El texto viaja como dato al proveedor; nunca se construye una linea de comandos con el.
        out.voice = provider.synthesize(speech.text, language=speech.language or "es", output_path=path,
                                        voice_id=voice_id, rate=rate)
    except TTSError as exc:
        out.errors.append(StageIssue(stage=STAGE_AUDIO, code=exc.code, message=exc.message))
        return out
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fallo inesperado del proveedor TTS")
        out.errors.append(StageIssue(stage=STAGE_AUDIO, code="TTS_UNEXPECTED", message=f"{type(exc).__name__}: {exc}"))
        return out

    problem = _verify_audio_file(path, provider.output_format)
    if problem:
        out.errors.append(StageIssue(stage=STAGE_AUDIO, code="INVALID_AUDIO_FILE", message=problem))
        return out

    out.status = StageStatus.OK
    out.file_name = path.name
    out.file_path = str(path.resolve())
    out.size_bytes = path.stat().st_size
    out.review_required = speech.review_required
    return out


def play_audio_file(path: str | Path, *, player: Optional[Callable[[str], None]] = None) -> tuple[bool, Optional[str]]:
    """Reproduce un WAV (solo Windows, con winsound de la libreria estandar). Devuelve (ok, error)."""
    target = Path(path)
    if not target.is_file():
        return False, "El archivo de audio no existe."
    if player is None:
        if not sys.platform.startswith("win"):
            return False, "La reproduccion integrada solo esta disponible en Windows."
        try:
            import winsound
        except ImportError:
            return False, "winsound no esta disponible."
        player = lambda p: winsound.PlaySound(p, winsound.SND_FILENAME)  # noqa: E731
    try:
        player(str(target))
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
