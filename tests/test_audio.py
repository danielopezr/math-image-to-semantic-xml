"""Pruebas de lectura matematica, sintesis y reproduccion con componentes FALSOS."""
import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace

from app.audio import (
    Pyttsx3Provider, TTSError, mathml_to_speech, play_audio_file, synthesize_audio, voice_matches_language,
)
from app.mathcat import MathCAT
from app.models import MathCATConfiguration, SpeechOutput, StageStatus
from tests.test_mathcat import FakeEngine

NS = "http://www.w3.org/1998/Math/MathML"
MATHML = f'<math xmlns="{NS}"><mi>x</mi><mo>=</mo><mn>0</mn></math>'


def mc(engine=None, **cfg):
    return MathCAT(MathCATConfiguration(**cfg), engine or FakeEngine())


def write_wav(path):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000); w.writeframes(b"\x00\x00" * 80)


class FakeProvider:
    name, output_format = "fake-tts", "wav"

    def __init__(self, behaviour="ok"):
        self.behaviour, self.calls = behaviour, []

    def synthesize(self, text, *, language, output_path, voice_id=None, rate=None):
        self.calls.append((text, language, voice_id, rate))
        if self.behaviour == "ok":
            write_wav(output_path)
        elif self.behaviour == "garbage":
            Path(output_path).write_bytes(b"no es audio")
        elif self.behaviour == "empty":
            Path(output_path).write_bytes(b"")
        elif self.behaviour == "error":
            raise TTSError("NO_VOICE_FOR_LANGUAGE", "sin voz")
        elif self.behaviour == "crash":
            raise RuntimeError("boom")
        return "voz-1"


def ok_speech(**kw):
    return SpeechOutput(status=StageStatus.OK, language="es", text="x igual a cero", **kw)


# ------------------------------------------------------------ texto de lectura
def test_lectura_en_espanol_por_defecto():
    engine = FakeEngine()
    out = mathml_to_speech(MATHML, mathcat=mc(engine))
    assert out.status is StageStatus.OK and out.text == "x igual a cero" and out.language == "es"
    assert (out.style, out.verbosity) == ("ClearSpeak", "Medium")
    assert engine.prefs["Language"] == "es" and engine.prefs["TTS"] == "None"   # texto plano


def test_lectura_con_idioma_y_preferencias_de_pronunciacion():
    engine = FakeEngine()
    out = mathml_to_speech(MATHML, mathcat=mc(engine), language="en", style="SimpleSpeak", verbosity="Terse")
    assert out.status is StageStatus.OK and (out.language, out.style, out.verbosity) == ("en", "SimpleSpeak", "Terse")
    assert (engine.prefs["Language"], engine.prefs["SpeechStyle"], engine.prefs["Verbosity"]) == ("en", "SimpleSpeak", "Terse")


def test_lectura_rechaza_mathml_invalido_y_configuracion_invalida():
    engine = FakeEngine()
    assert mathml_to_speech("<math><mi>", mathcat=mc(engine)).errors[0].code == "INVALID_MATHML"
    assert engine.calls == []
    bad = mathml_to_speech(MATHML, mathcat=mc(), style="Otro")
    assert bad.status is StageStatus.FAILED and bad.errors[0].code == "CONFIG_INVALID"


def test_lectura_vacia_o_fallo_de_mathcat():
    assert mathml_to_speech(MATHML, mathcat=mc(FakeEngine(speech=" "))).errors[0].code == "EMPTY_SPEECH"
    assert mathml_to_speech(MATHML, mathcat=mc(FakeEngine(fail_on="set_mathml"))).status is StageStatus.FAILED


def test_la_lectura_no_es_audio():
    out = mathml_to_speech(MATHML, mathcat=mc())
    assert not hasattr(out, "file_name")          # SpeechOutput solo tiene texto


# ------------------------------------------------------------- sintesis de audio
def test_sin_proveedor_el_audio_se_omite_y_no_se_afirma_ningun_archivo():
    out = synthesize_audio(ok_speech(), None, "salida.wav")
    assert out.status is StageStatus.SKIPPED and out.file_name is None and out.file_path is None
    assert [w.code for w in out.warnings] == ["NO_TTS_PROVIDER"]


def test_audio_ok_solo_si_el_archivo_existe_y_es_valido():
    with tempfile.TemporaryDirectory() as tmp:
        provider = FakeProvider()
        out = synthesize_audio(ok_speech(), provider, Path(tmp) / "sub" / "lectura.wav", voice_id="v", rate=150)
        assert out.status is StageStatus.OK and out.file_name == "lectura.wav" and out.size_bytes > 44
        assert Path(out.file_path).is_file() and out.voice == "voz-1" and out.provider == "fake-tts"
        assert provider.calls == [("x igual a cero", "es", "v", 150)]


def test_fallos_del_proveedor_y_archivos_invalidos():
    with tempfile.TemporaryDirectory() as tmp:
        cases = {"error": "NO_VOICE_FOR_LANGUAGE", "crash": "TTS_UNEXPECTED", "garbage": "INVALID_AUDIO_FILE",
                 "empty": "INVALID_AUDIO_FILE", "none": "INVALID_AUDIO_FILE"}
        for behaviour, code in cases.items():
            out = synthesize_audio(ok_speech(), FakeProvider(behaviour), Path(tmp) / f"{behaviour}.wav")
            assert out.status is StageStatus.FAILED and out.errors[0].code == code, behaviour
            assert out.file_name is None and out.file_path is None


def test_validaciones_previas_de_ruta_y_texto():
    with tempfile.TemporaryDirectory() as tmp:
        assert synthesize_audio(ok_speech(), FakeProvider(), Path(tmp) / "a.mp3").errors[0].code == "BAD_EXTENSION"
        assert synthesize_audio(ok_speech(), FakeProvider(), Path(tmp)).errors[0].code == "BAD_EXTENSION"
        empty = SpeechOutput(status=StageStatus.FAILED)
        provider = FakeProvider()
        assert synthesize_audio(empty, provider, Path(tmp) / "a.wav").errors[0].code == "NO_SPEECH_TEXT"
        assert provider.calls == []


def test_la_revision_pendiente_de_la_lectura_se_propaga_al_audio():
    with tempfile.TemporaryDirectory() as tmp:
        out = synthesize_audio(ok_speech(review_required=True), FakeProvider(), Path(tmp) / "a.wav")
        assert out.review_required is True


# --------------------------------------------------------------------- pyttsx3
class FakeVoice(SimpleNamespace):
    pass


class FakePyttsxEngine:
    def __init__(self, voices):
        self.voices, self.props, self.saved = voices, {}, []

    def getProperty(self, name): return self.voices if name == "voices" else None
    def setProperty(self, name, value): self.props[name] = value
    def save_to_file(self, text, path): self.saved.append((text, path))
    def runAndWait(self): write_wav(self.saved[-1][1])


VOICES = [FakeVoice(id="HKEY\\EN-US", name="Microsoft David Desktop - English (United States)", languages=[]),
          FakeVoice(id="HKEY\\ES-MX", name="Microsoft Sabina Desktop - Spanish (Mexico)", languages=[])]


def test_seleccion_de_voz_por_idioma():
    assert voice_matches_language(VOICES[1], "es") and not voice_matches_language(VOICES[0], "es")
    assert voice_matches_language(FakeVoice(id="x", name="y", languages=[b"\x05es_ES"]), "es-CO")
    assert voice_matches_language(FakeVoice(id="x", name="y", languages=["en_US"]), "en")


def test_pyttsx3_elige_la_voz_espanola_y_genera_wav():
    with tempfile.TemporaryDirectory() as tmp:
        engine = FakePyttsxEngine(VOICES)
        out = synthesize_audio(ok_speech(), Pyttsx3Provider(lambda: engine), Path(tmp) / "a.wav", rate=140)
        assert out.status is StageStatus.OK and out.voice == "HKEY\\ES-MX"
        assert engine.props == {"voice": "HKEY\\ES-MX", "rate": 140}


def test_pyttsx3_sin_voz_del_idioma_falla_en_lugar_de_leer_con_otra():
    with tempfile.TemporaryDirectory() as tmp:
        engine = FakePyttsxEngine(VOICES[:1])
        out = synthesize_audio(ok_speech(), Pyttsx3Provider(lambda: engine), Path(tmp) / "a.wav")
        assert out.status is StageStatus.FAILED and out.errors[0].code == "NO_VOICE_FOR_LANGUAGE"
        assert engine.saved == []


def test_pyttsx3_voz_explicita_y_voz_inexistente():
    with tempfile.TemporaryDirectory() as tmp:
        engine = FakePyttsxEngine(VOICES)
        ok = synthesize_audio(ok_speech(), Pyttsx3Provider(lambda: engine), Path(tmp) / "a.wav", voice_id="HKEY\\EN-US")
        assert ok.status is StageStatus.OK and ok.voice == "HKEY\\EN-US"
        bad = synthesize_audio(ok_speech(), Pyttsx3Provider(lambda: engine), Path(tmp) / "b.wav", voice_id="otra")
        assert bad.errors[0].code == "VOICE_NOT_FOUND"


def test_pyttsx3_no_instalado_da_error_claro():
    class Provider(Pyttsx3Provider):
        def _make_engine(self):
            raise TTSError("TTS_NOT_INSTALLED", "pyttsx3 no esta instalado")
    with tempfile.TemporaryDirectory() as tmp:
        out = synthesize_audio(ok_speech(), Provider(), Path(tmp) / "a.wav")
        assert out.errors[0].code == "TTS_NOT_INSTALLED"


# ----------------------------------------------------------------- reproduccion
def test_reproduccion_con_reproductor_inyectado_y_errores():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "a.wav"; write_wav(path)
        played = []
        assert play_audio_file(path, player=played.append) == (True, None) and played == [str(path)]
        ok, err = play_audio_file(path, player=lambda p: (_ for _ in ()).throw(RuntimeError("sin dispositivo")))
        assert ok is False and "sin dispositivo" in err
        assert play_audio_file(Path(tmp) / "no.wav", player=played.append)[0] is False
