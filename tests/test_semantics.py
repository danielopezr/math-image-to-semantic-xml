"""Pruebas del agente semantico con proveedores falsos (sin red ni LLM)."""
import json

from app.models import StageStatus
from app.semantic_agent import (
    OllamaProvider, ProviderError, analyze_semantics, build_system_prompt, extract_json,
)
from app.semantic_tree import validate_semantic_output

LATEX = "x = 0"
MATHML = '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>x</mi><mo>=</mo><mn>0</mn></math>'


def payload(**overrides):
    data = {
        "vocabulary_version": "1.0.0", "root_id": "n0",
        "nodes": [
            {"id": "n0", "node_type": "structure", "intent": "equation",
             "operands": [{"role": "statement", "node_id": "n1"}]},
            {"id": "n1", "node_type": "relation", "intent": "equality", "symbol": "=",
             "operands": [{"role": "left", "node_id": "n2"}, {"role": "right", "node_id": "n3"}]},
            {"id": "n2", "node_type": "element", "intent": "variable", "symbol": "x"},
            {"id": "n3", "node_type": "element", "intent": "number", "symbol": "0"},
        ],
        "review_required": False,
    }
    data.update(overrides)
    return data


class ScriptedProvider:
    name, model = "scripted", "fake-1"

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def complete(self, system_prompt, user_prompt, json_schema):
        self.calls.append(user_prompt)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, str) else json.dumps(item)


def test_respuesta_valida_con_orden_de_evaluacion_y_huellas():
    r = analyze_semantics(LATEX, MATHML, ScriptedProvider(payload()))
    assert r.status is StageStatus.OK and r.attempts == 1
    assert [n.evaluation_order for n in r.output.nodes] == [3, 2, 0, 1]
    assert len(r.input_latex_sha256) == 64 and len(r.input_mathml_sha256) == 64
    assert validate_semantic_output(r.output, latex=LATEX, mathml=MATHML) == []


def test_json_dentro_de_bloque_de_codigo():
    text = "```json\n" + json.dumps(payload()) + "\n```"
    assert analyze_semantics(LATEX, MATHML, ScriptedProvider(text)).status is StageStatus.OK
    assert extract_json("Aqui va: " + json.dumps({"a": 1}) + " fin") == '{"a": 1}'


def test_json_invalido_agota_reintentos():
    r = analyze_semantics(LATEX, MATHML, ScriptedProvider("no es json", "tampoco"), max_retries=1)
    assert r.status is StageStatus.FAILED and r.output is None and r.attempts == 2
    assert r.errors[0].code == "SCHEMA_VALIDATION_FAILED"


def test_esquema_incumplido_y_reintento_exitoso():
    bad = payload(); del bad["review_required"]
    p = ScriptedProvider(bad, payload())
    r = analyze_semantics(LATEX, MATHML, p)
    assert r.status is StageStatus.OK and r.attempts == 2
    assert "review_required" in p.calls[1]


def test_el_llm_no_puede_devolver_latex_ni_mathml():
    r = analyze_semantics(LATEX, MATHML, ScriptedProvider(payload(latex="y = 1")), max_retries=0)
    assert r.status is StageStatus.FAILED


def test_intent_fuera_del_vocabulario_se_rechaza_con_retroalimentacion():
    bad = payload(); bad["nodes"][1]["intent"] = "igualdad"
    p = ScriptedProvider(bad, payload())
    r = analyze_semantics(LATEX, MATHML, p)
    assert r.status is StageStatus.OK and "UNKNOWN_INTENT" in p.calls[1]


def test_ciclo_y_ids_duplicados_se_rechazan():
    cyc = payload(); cyc["nodes"][2]["operands"] = [{"role": "operand", "node_id": "n0"}]
    cyc["nodes"][2].update(node_type="operation", intent="negation")
    assert analyze_semantics(LATEX, MATHML, ScriptedProvider(cyc), max_retries=0).status is StageStatus.FAILED
    dup = payload(); dup["nodes"][3]["id"] = "n2"
    assert analyze_semantics(LATEX, MATHML, ScriptedProvider(dup), max_retries=0).status is StageStatus.FAILED


def test_simbolo_inventado_es_error_y_no_solo_advertencia():
    bad = payload(); bad["nodes"][2]["symbol"] = "q"
    r = analyze_semantics(LATEX, MATHML, ScriptedProvider(bad), max_retries=0)
    assert r.status is StageStatus.FAILED and "GROUNDING_MISMATCH" in r.errors[0].message


def test_confianza_declarada_se_descarta_y_el_orden_del_llm_tambien():
    data = payload(confidence=0.99)
    data["nodes"][2]["evaluation_order"] = 77
    r = analyze_semantics(LATEX, MATHML, ScriptedProvider(data))
    assert r.status is StageStatus.OK and r.output.confidence is None
    assert "CONFIDENCE_DISCARDED" in [w.code for w in r.warnings]
    assert r.output.nodes[2].evaluation_order == 0


def test_nodo_ambiguo_valido_se_acepta_con_revision():
    amb = payload(review_required=True, ambiguities=["xy"])
    amb["nodes"] = [
        {"id": "n0", "node_type": "unresolved", "ambiguous": True, "review_required": True,
         "candidate_intents": ["multiplication", "variable"], "justification": "sin contexto",
         "operands": [{"role": "component", "node_id": "n1"}, {"role": "component", "node_id": "n2"}]},
        {"id": "n1", "node_type": "element", "intent": "variable", "symbol": "x"},
        {"id": "n2", "node_type": "element", "intent": "variable", "symbol": "y"},
    ]
    ml = '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>x</mi><mi>y</mi></math>'
    r = analyze_semantics("xy", ml, ScriptedProvider(amb))
    assert r.status is StageStatus.OK and r.output.nodes[0].ambiguous and r.output.nodes[0].intent is None


def test_error_del_proveedor():
    r = analyze_semantics(LATEX, MATHML, ScriptedProvider(ProviderError("sin conexion")))
    assert r.errors[0].code == "PROVIDER_ERROR"


def test_el_prompt_incluye_el_vocabulario_y_las_reglas_de_ambiguedad():
    prompt = build_system_prompt({"x": 1})
    assert "power [node_type=operation] roles: base, exponent" in prompt
    assert "ambiguous" in prompt and "No modifiques" in prompt and "1.0.0" in prompt


def test_ollama_requiere_modelo():
    try:
        OllamaProvider(model="")
    except ProviderError:
        return
    raise AssertionError("debio lanzar ProviderError")
