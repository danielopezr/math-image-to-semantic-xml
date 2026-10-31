"""Pruebas del agente semantico con proveedores falsos (no necesitan red ni LLM)."""
import json

from app.models import StageStatus
from app.semantic_agent import (
    OllamaProvider,
    ProviderError,
    analyze_semantics,
    extract_json,
)

LATEX = "x = 0"
MATHML = '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>x</mi><mo>=</mo><mn>0</mn></math>'


def valid_payload(**overrides):
    data = {
        "expression_type": "equation",
        "root": {
            "id": "n0", "kind": "equation", "symbol": "=",
            "children": [
                {"id": "n1", "kind": "variable", "symbol": "x"},
                {"id": "n2", "kind": "number", "symbol": "0"},
            ],
        },
        "relations": [{"type": "equality", "left": "n1", "right": "n2"}],
        "assumptions": [], "ambiguities": [], "requires_review": False,
    }
    data.update(overrides)
    return data


class ScriptedProvider:
    """Devuelve respuestas predefinidas en orden y registra las llamadas."""
    name, model = "scripted", "fake-1"

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def complete(self, system_prompt, user_prompt, json_schema):
        self.calls.append(user_prompt)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, str) else json.dumps(item)


def test_respuesta_valida():
    r = analyze_semantics(LATEX, MATHML, ScriptedProvider(valid_payload()))
    assert r.status is StageStatus.OK
    assert r.output.root.kind == "equation"
    assert r.attempts == 1 and len(r.raw_responses) == 1


def test_json_dentro_de_bloque_de_codigo():
    text = "```json\n" + json.dumps(valid_payload()) + "\n```"
    assert analyze_semantics(LATEX, MATHML, ScriptedProvider(text)).status is StageStatus.OK
    assert extract_json("Aqui va: " + json.dumps({"a": 1}) + " fin") == '{"a": 1}'


def test_json_invalido_agota_reintentos():
    p = ScriptedProvider("esto no es json", "tampoco")
    r = analyze_semantics(LATEX, MATHML, p, max_retries=1)
    assert r.status is StageStatus.FAILED and r.output is None
    assert r.attempts == 2
    assert r.errors[0].code == "SCHEMA_VALIDATION_FAILED"


def test_esquema_incumplido_y_reintento_exitoso():
    bad = valid_payload()
    del bad["requires_review"]
    p = ScriptedProvider(bad, valid_payload())
    r = analyze_semantics(LATEX, MATHML, p)
    assert r.status is StageStatus.OK and r.attempts == 2
    assert "requires_review" in p.calls[1]  # el error se devuelve como retroalimentacion


def test_campos_extra_rechazados():
    r = analyze_semantics(LATEX, MATHML, ScriptedProvider(valid_payload(extra_field=1)), max_retries=0)
    assert r.status is StageStatus.FAILED


def test_ids_duplicados_rechazados():
    bad = valid_payload()
    bad["root"]["children"][1]["id"] = "n1"
    r = analyze_semantics(LATEX, MATHML, ScriptedProvider(bad), max_retries=0)
    assert r.status is StageStatus.FAILED
    assert "duplicados" in r.errors[0].message


def test_relacion_con_id_inexistente():
    bad = valid_payload(relations=[{"type": "equality", "left": "n1", "right": "zzz"}])
    r = analyze_semantics(LATEX, MATHML, ScriptedProvider(bad), max_retries=0)
    assert r.status is StageStatus.FAILED
    assert "inexistente" in r.errors[0].message


def test_confianza_declarada_se_descarta():
    r = analyze_semantics(LATEX, MATHML, ScriptedProvider(valid_payload(confidence=0.99)))
    assert r.status is StageStatus.OK
    assert r.output.confidence is None
    assert "CONFIDENCE_DISCARDED" in [w.code for w in r.warnings]


def test_simbolo_inventado_marca_revision():
    bad = valid_payload()
    bad["root"]["children"][0]["symbol"] = "q"
    r = analyze_semantics(LATEX, MATHML, ScriptedProvider(bad))
    assert r.status is StageStatus.OK
    assert r.output.requires_review is True
    assert "SYMBOL_NOT_IN_FORMULA" in [w.code for w in r.warnings]


def test_error_del_proveedor():
    r = analyze_semantics(LATEX, MATHML, ScriptedProvider(ProviderError("sin conexion")))
    assert r.status is StageStatus.FAILED
    assert r.errors[0].code == "PROVIDER_ERROR"


def test_ollama_requiere_modelo():
    try:
        OllamaProvider(model="")
    except ProviderError:
        return
    raise AssertionError("debio lanzar ProviderError")
