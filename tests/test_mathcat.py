"""Pruebas de la interfaz con MathCAT usando un motor FALSO.

El motor falso solo registra llamadas y devuelve texto fijo: NO reproduce resultados reales de
MathCAT. Las pruebas de integracion reales (al final) se omiten si MathCAT no esta disponible.
"""
import os
import tempfile

from app.mathcat import (
    MathCAT, MathCATError, PythonMathCATEngine, load_configuration, resolve_braille_code, validate_configuration,
)
from app.models import MathCATConfiguration

NS = "http://www.w3.org/1998/Math/MathML"
MATHML = f'<math xmlns="{NS}"><mi>x</mi><mo>=</mo><mn>0</mn></math>'


class FakeEngine:
    def __init__(self, braille="⠼⠁⠃", speech="x igual a cero", version="fake-0", wrong_pref=None, fail_on=None):
        self.calls, self.prefs = [], {}
        self.braille, self.speech, self.version = braille, speech, version
        self.wrong_pref, self.fail_on = wrong_pref or {}, fail_on
        self.closed = False

    def initialize(self): self.calls.append(("initialize",))

    def set_preference(self, name, value):
        self.calls.append(("set_preference", name, value)); self.prefs[name] = value

    def get_preference(self, name): return self.wrong_pref.get(name, self.prefs.get(name))

    def set_mathml(self, mathml):
        self.calls.append(("set_mathml", mathml))
        if self.fail_on == "set_mathml":
            raise MathCATError("MATHCAT_CALL_FAILED", "SetMathML fallo")
        return mathml

    def get_spoken_text(self): self.calls.append(("get_spoken_text",)); return self.speech
    def get_braille(self): self.calls.append(("get_braille",)); return self.braille
    def get_version(self): return self.version
    def close(self): self.closed = True


def make(engine=None, **cfg):
    return MathCAT(MathCATConfiguration(**cfg), engine or FakeEngine())


# ------------------------------------------------------------------ configuracion
def test_valores_por_defecto_en_un_unico_lugar():
    cfg = load_configuration({})
    assert (cfg.braille_code, cfg.speech_language, cfg.speech_style, cfg.verbosity) == ("CMU", "es", "ClearSpeak", "Medium")
    assert cfg == MathCATConfiguration() and cfg.rules_dir is None


def test_configuracion_desde_entorno():
    env = {"MATHCAT_BRAILLE_CODE": "nemeth", "MATHCAT_SPEECH_LANGUAGE": "en", "MathCATRulesDir": "/x/Rules"}
    cfg = load_configuration(env)
    assert cfg.braille_code == "nemeth" and cfg.speech_language == "en" and cfg.rules_dir == "/x/Rules"
    assert load_configuration({"MATHCAT_RULES_DIR": "/y", "MathCATRulesDir": "/x"}).rules_dir == "/y"


def test_validacion_de_configuracion():
    assert validate_configuration(MathCATConfiguration()) == []
    bad = MathCATConfiguration(braille_code="braille-x", speech_language="Español!", speech_style="Otro", verbosity="Mucha")
    assert len(validate_configuration(bad)) == 4


def test_codigos_braille_soportados():
    assert resolve_braille_code("Nemeth") == "Nemeth"
    assert resolve_braille_code("UEB Technical") == "UEB" and resolve_braille_code("ueb-technical") == "UEB"
    assert resolve_braille_code("CMU") == "CMU"
    try:
        resolve_braille_code("brf")
    except MathCATError as exc:
        assert exc.code == "UNSUPPORTED_BRAILLE_CODE"
    else:
        raise AssertionError


# ------------------------------------------------------------------- fachada
def test_to_braille_fija_preferencias_antes_de_convertir():
    engine = FakeEngine()
    mc = make(engine, braille_code="cmu", speech_language="es")
    assert mc.to_braille(MATHML) == "⠼⠁⠃"
    names = [c[0] if c[0] != "set_preference" else c[1] for c in engine.calls]
    assert names == ["initialize", "TTS", "Language", "BrailleCode", "set_mathml", "get_braille"]
    assert engine.prefs["TTS"] == "None" and engine.prefs["BrailleCode"] == "CMU" and engine.prefs["Language"] == "es"
    assert ("set_mathml", MATHML) in engine.calls            # el MathML llega sin modificar
    assert mc.version == "fake-0"


def test_codigo_braille_configurable_por_llamada():
    engine = FakeEngine()
    make(engine).to_braille(MATHML, braille_code="ueb_technical")
    assert engine.prefs["BrailleCode"] == "UEB"


def test_codigo_no_soportado_no_llama_al_motor_para_convertir():
    engine = FakeEngine()
    try:
        make(engine).to_braille(MATHML, braille_code="otro")
    except MathCATError as exc:
        assert exc.code == "UNSUPPORTED_BRAILLE_CODE"
    assert not [c for c in engine.calls if c[0] == "set_mathml"]


def test_preferencia_rechazada_se_detecta():
    engine = FakeEngine(wrong_pref={"BrailleCode": "Nemeth"})
    try:
        make(engine).to_braille(MATHML)
    except MathCATError as exc:
        assert exc.code == "PREFERENCE_REJECTED"
    else:
        raise AssertionError


def test_to_speech_fija_idioma_estilo_y_verbosidad():
    engine = FakeEngine()
    assert make(engine).to_speech(MATHML, verbosity="Verbose") == "x igual a cero"
    assert (engine.prefs["Language"], engine.prefs["SpeechStyle"], engine.prefs["Verbosity"]) == ("es", "ClearSpeak", "Verbose")
    try:
        make(engine).to_speech(MATHML, style="Inventado")
    except MathCATError as exc:
        assert exc.code == "CONFIG_INVALID"


def test_comprobacion_estricta_de_intent_restaura_la_preferencia():
    engine = FakeEngine()
    make(engine).check_intent_attributes(MATHML)
    values = [c[2] for c in engine.calls if c[:2] == ("set_preference", "IntentErrorRecovery")]
    assert values == ["Error", "IgnoreIntent"]
    failing = FakeEngine(fail_on="set_mathml")
    try:
        make(failing).check_intent_attributes(MATHML)
    except MathCATError:
        pass
    assert [c[2] for c in failing.calls if c[:2] == ("set_preference", "IntentErrorRecovery")] == ["Error", "IgnoreIntent"]


def test_close_libera_el_motor_y_permite_reinicializar():
    engine = FakeEngine()
    mc = make(engine); mc.to_speech(MATHML); mc.close()
    assert engine.closed and mc.version is None
    mc.to_speech(MATHML)
    assert [c for c in engine.calls if c == ("initialize",)] == [("initialize",), ("initialize",)]


def test_sin_carpeta_rules_no_se_inicializa():
    try:
        MathCAT(MathCATConfiguration(rules_dir=None)).initialize()
    except MathCATError as exc:
        assert exc.code == "RULES_DIR_MISSING"
    else:
        raise AssertionError
    try:
        MathCAT(MathCATConfiguration(rules_dir="/ruta/que/no/existe")).initialize()
    except MathCATError as exc:
        assert exc.code == "RULES_DIR_MISSING"


def test_solo_se_importan_modulos_conocidos():
    with tempfile.TemporaryDirectory() as rules:
        try:
            MathCAT(MathCATConfiguration(rules_dir=rules, python_module="os")).initialize()
        except MathCATError as exc:
            assert exc.code == "ENGINE_NOT_AVAILABLE" and "no permitido" in exc.message.lower()
        else:
            raise AssertionError("no debe importar modulos arbitrarios")


# --------------------------------------- mapeo a la API CamelCase documentada
class FakeNativeModule:
    def __init__(self): self.calls, self.prefs = [], {}
    def SetRulesDir(self, d): self.calls.append(("SetRulesDir", d))
    def SetPreference(self, n, v): self.calls.append(("SetPreference", n, v)); self.prefs[n] = v
    def GetPreference(self, n): return self.prefs.get(n)
    def SetMathML(self, m): self.calls.append(("SetMathML", m)); return m
    def GetSpokenText(self): return "texto"
    def GetBraille(self, node_id): self.calls.append(("GetBraille", node_id)); return "⠁"
    def GetVersion(self): return "9.9-fake"


def test_el_motor_python_usa_los_nombres_de_la_api_documentada():
    mod = FakeNativeModule()
    engine = PythonMathCATEngine("/Rules", module=mod)
    engine.initialize()
    engine.set_preference("Language", "es")
    assert engine.set_mathml(MATHML) == MATHML and engine.get_braille() == "⠁" and engine.get_spoken_text() == "texto"
    assert mod.calls[0] == ("SetRulesDir", "/Rules")          # la primera llamada
    assert ("GetBraille", "") in mod.calls                     # sin nodo de navegacion
    assert engine.get_preference("Language") == "es" and engine.get_version() == "9.9-fake"


def test_errores_nativos_se_traducen():
    class Broken(FakeNativeModule):
        def SetMathML(self, m): raise RuntimeError("mathml invalido")
    engine = PythonMathCATEngine("/Rules", module=Broken())
    engine.initialize()
    try:
        engine.set_mathml("<x>")
    except MathCATError as exc:
        assert exc.code == "MATHCAT_CALL_FAILED" and "SetMathML" in exc.message


# ------------------------------------------------ integracion REAL (si esta disponible)
def _real():
    import pytest
    if not os.environ.get("MATHCAT_RULES_DIR") and not os.environ.get("MathCATRulesDir"):
        pytest.skip("Define MATHCAT_RULES_DIR para ejecutar las pruebas reales de MathCAT.")
    mc = MathCAT(load_configuration())
    try:
        mc.initialize()
    except MathCATError as exc:
        pytest.skip(f"MathCAT no disponible: {exc.message}")
    return mc


def test_integracion_real_braille_y_lectura_en_espanol():
    mc = _real()
    braille = mc.to_braille(MATHML)
    speech = mc.to_speech(MATHML, language="es")
    assert braille.strip() and speech.strip()
    assert all(0x2800 <= ord(c) < 0x2900 or c.isspace() for c in braille), "se esperaban solo celdas braille"
    mc.close()


def test_integracion_real_los_tres_codigos():
    mc = _real()
    for code in ("nemeth", "ueb_technical", "cmu"):
        out = mc.to_braille(MATHML, braille_code=code)
        assert out.strip(), code
    mc.close()
