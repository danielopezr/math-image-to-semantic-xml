"""Interfaz independiente con MathCAT (DAISY): MathML -> braille y texto de lectura matematica.

Referencias consultadas (documentacion oficial de MathCAT, 2026-10):
  * Orden de llamadas: SetRulesDir (la primera), SetPreference*, SetMathML, GetSpokenText / GetBraille.
  * GetBraille se llama con "" (sin nodo de navegacion).
  * Preferencias usadas: Language, BrailleCode, SpeechStyle, Verbosity, TTS (SSML | SAPI5 | None),
    IntentErrorRecovery (Error | IgnoreIntent). Nombres y valores distinguen mayusculas.
  * Idioma de voz "es" y codigos braille Nemeth, UEB (tecnico) y CMU figuran como soportados.

NO VERIFICADO: como instalar el binding de Python. MathCAT se distribuye como libreria Rust; su
interfaz Python (MathCATForPython) se construye con pyo3 y se entrega, por ejemplo, dentro del
complemento de NVDA. Este modulo carga el modulo nativo de forma perezosa por su nombre (varia
entre versiones: ``libmathcat_py`` o ``libmathcat``) y NO inventa ninguna otra llamada: el resto de
funcionalidad queda detras de la interfaz ``MathCATEngine``.

MathCAT guarda su estado (reglas y preferencias) a nivel de proceso: todas las llamadas se
serializan con un cerrojo y las preferencias se fijan antes de cada conversion.
"""
from __future__ import annotations

import importlib
import logging
import os
import re
import threading
from pathlib import Path
from typing import Mapping, Optional, Protocol

from app.models import MathCATConfiguration

logger = logging.getLogger(__name__)

# Codigo del proyecto -> valor de la preferencia BrailleCode de MathCAT.
BRAILLE_CODES: dict[str, str] = {"nemeth": "Nemeth", "ueb_technical": "UEB", "cmu": "CMU"}
SPEECH_STYLES = ("ClearSpeak", "SimpleSpeak")
VERBOSITIES = ("Terse", "Medium", "Verbose")
# Nombres del modulo nativo observados en distintas versiones de MathCATForPython.
KNOWN_PY_MODULES = ("libmathcat_py", "libmathcat")
_LANGUAGE_RE = re.compile(r"^[a-z]{2,3}(-[A-Za-z]{2,4})?$")


class MathCATError(Exception):
    """Fallo de MathCAT o de su configuracion, con codigo estable."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class MathCATEngine(Protocol):
    """Lo minimo que el resto de la aplicacion necesita de MathCAT."""

    def initialize(self) -> None: ...
    def set_preference(self, name: str, value: str) -> None: ...
    def get_preference(self, name: str) -> Optional[str]: ...
    def set_mathml(self, mathml: str) -> str: ...
    def get_spoken_text(self) -> str: ...
    def get_braille(self) -> str: ...
    def get_version(self) -> Optional[str]: ...
    def close(self) -> None: ...


class PythonMathCATEngine:
    """Motor real: llama al modulo nativo pyo3 con la API CamelCase documentada."""

    def __init__(self, rules_dir: str, module_name: Optional[str] = None, module: object = None):
        self._rules_dir = rules_dir
        self._module_name = module_name
        self._mod = module

    def initialize(self) -> None:
        if self._mod is None:
            names = (self._module_name,) if self._module_name else KNOWN_PY_MODULES
            if self._module_name and self._module_name not in KNOWN_PY_MODULES:
                raise MathCATError("ENGINE_NOT_AVAILABLE", f"Modulo no permitido: {self._module_name!r}.")
            last: Optional[Exception] = None
            for name in names:
                try:
                    self._mod = importlib.import_module(name)
                    break
                except ImportError as exc:
                    last = exc
            if self._mod is None:
                raise MathCATError(
                    "ENGINE_NOT_AVAILABLE",
                    f"No se pudo importar el modulo nativo de MathCAT ({', '.join(names)}): {last}")
        try:
            self._mod.SetRulesDir(self._rules_dir)  # debe ser la primera llamada a MathCAT
        except Exception as exc:  # noqa: BLE001
            raise MathCATError("ENGINE_INIT_FAILED", f"SetRulesDir fallo: {exc}") from exc

    def _call(self, fn: str, *args: str) -> object:
        try:
            return getattr(self._mod, fn)(*args)
        except Exception as exc:  # noqa: BLE001
            raise MathCATError("MATHCAT_CALL_FAILED", f"{fn} fallo: {exc}") from exc

    def set_preference(self, name: str, value: str) -> None:
        self._call("SetPreference", name, value)

    def get_preference(self, name: str) -> Optional[str]:
        value = self._call("GetPreference", name)
        return None if value is None else str(value)

    def set_mathml(self, mathml: str) -> str:
        return str(self._call("SetMathML", mathml))

    def get_spoken_text(self) -> str:
        return str(self._call("GetSpokenText"))

    def get_braille(self) -> str:
        return str(self._call("GetBraille", ""))

    def get_version(self) -> Optional[str]:
        fn = getattr(self._mod, "GetVersion", None)  # nombre derivado de la convencion CamelCase; opcional
        try:
            return str(fn()) if fn else None
        except Exception:  # noqa: BLE001
            return None

    def close(self) -> None:
        # No se ha verificado ninguna API de liberacion en el binding de Python: solo soltamos la referencia.
        self._mod = None


# ---------------------------------------------------------------- configuracion
def load_configuration(env: Optional[Mapping[str, str]] = None) -> MathCATConfiguration:
    """Lee la configuracion de variables de entorno. Los valores por defecto estan en MathCATConfiguration."""
    env = os.environ if env is None else env
    defaults = MathCATConfiguration()
    return MathCATConfiguration(
        rules_dir=env.get("MATHCAT_RULES_DIR") or env.get("MathCATRulesDir") or None,
        braille_code=env.get("MATHCAT_BRAILLE_CODE") or defaults.braille_code,
        speech_language=env.get("MATHCAT_SPEECH_LANGUAGE") or defaults.speech_language,
        speech_style=env.get("MATHCAT_SPEECH_STYLE") or defaults.speech_style,
        verbosity=env.get("MATHCAT_VERBOSITY") or defaults.verbosity,
        python_module=env.get("MATHCAT_PY_MODULE") or None,
    )


def resolve_braille_code(code: str) -> str:
    """Devuelve el valor de BrailleCode para MathCAT o lanza MathCATError."""
    key = (code or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key not in BRAILLE_CODES:
        raise MathCATError("UNSUPPORTED_BRAILLE_CODE",
                           f"Codigo braille no soportado: {code!r}. Opciones: {', '.join(BRAILLE_CODES)}.")
    return BRAILLE_CODES[key]


def validate_configuration(cfg: MathCATConfiguration) -> list[str]:
    problems: list[str] = []
    try:
        resolve_braille_code(cfg.braille_code)
    except MathCATError as exc:
        problems.append(exc.message)
    if not _LANGUAGE_RE.match(cfg.speech_language or ""):
        problems.append(f"Idioma invalido: {cfg.speech_language!r}.")
    if cfg.speech_style not in SPEECH_STYLES:
        problems.append(f"Estilo de lectura invalido: {cfg.speech_style!r}. Opciones: {', '.join(SPEECH_STYLES)}.")
    if cfg.verbosity not in VERBOSITIES:
        problems.append(f"Verbosidad invalida: {cfg.verbosity!r}. Opciones: {', '.join(VERBOSITIES)}.")
    return problems


# --------------------------------------------------------------------- fachada
class MathCAT:
    """Fachada: inicializa el motor, fija preferencias y convierte MathML."""

    def __init__(self, config: Optional[MathCATConfiguration] = None, engine: Optional[MathCATEngine] = None):
        self.config = config or load_configuration()
        self._engine = engine
        self._lock = threading.RLock()
        self._initialized = False
        self._version: Optional[str] = None

    @property
    def version(self) -> Optional[str]:
        return self._version

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            problems = validate_configuration(self.config)
            if problems:
                raise MathCATError("CONFIG_INVALID", " ".join(problems))
            if self._engine is None:
                rules = self.config.rules_dir
                if not rules or not Path(rules).is_dir():
                    raise MathCATError(
                        "RULES_DIR_MISSING",
                        "Indica la carpeta Rules de MathCAT (MATHCAT_RULES_DIR); no existe o no es una carpeta.")
                self._engine = PythonMathCATEngine(rules, self.config.python_module)
            self._engine.initialize()
            self._engine.set_preference("TTS", "None")  # texto plano (sin SSML ni SAPI5)
            self._version = self._engine.get_version()
            self._initialized = True

    def _set_checked(self, name: str, value: str) -> None:
        self._engine.set_preference(name, value)
        current = self._engine.get_preference(name)
        if current is not None and current != value:
            raise MathCATError("PREFERENCE_REJECTED", f"MathCAT no acepto {name}={value!r} (valor actual {current!r}).")

    def to_braille(self, mathml: str, *, braille_code: Optional[str] = None, language: Optional[str] = None) -> str:
        """Devuelve el texto Unicode braille generado por MathCAT."""
        value = resolve_braille_code(braille_code or self.config.braille_code)
        with self._lock:
            self.initialize()
            self._set_checked("Language", language or self.config.speech_language)
            self._set_checked("BrailleCode", value)
            self._engine.set_mathml(mathml)
            return self._engine.get_braille()

    def to_speech(self, mathml: str, *, language: Optional[str] = None, style: Optional[str] = None,
                  verbosity: Optional[str] = None) -> str:
        """Devuelve el texto de lectura matematica (sin sintetizar audio)."""
        style = style or self.config.speech_style
        verbosity = verbosity or self.config.verbosity
        problems = validate_configuration(self.config.model_copy(update={
            "speech_language": language or self.config.speech_language, "speech_style": style, "verbosity": verbosity}))
        if problems:
            raise MathCATError("CONFIG_INVALID", " ".join(problems))
        with self._lock:
            self.initialize()
            self._set_checked("Language", language or self.config.speech_language)
            self._set_checked("SpeechStyle", style)
            self._set_checked("Verbosity", verbosity)
            self._engine.set_mathml(mathml)
            return self._engine.get_spoken_text()

    def check_intent_attributes(self, mathml: str) -> None:
        """Comprueba que MathCAT acepta los atributos intent del MathML (modo estricto durante la prueba)."""
        with self._lock:
            self.initialize()
            self._engine.set_preference("IntentErrorRecovery", "Error")
            try:
                self._engine.set_mathml(mathml)
                self._engine.get_spoken_text()
            finally:
                self._engine.set_preference("IntentErrorRecovery", "IgnoreIntent")

    def close(self) -> None:
        with self._lock:
            if self._engine is not None:
                self._engine.close()
            self._initialized = False
            self._version = None
