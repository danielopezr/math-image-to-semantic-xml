"""Pruebas del vocabulario de intents y del arbol semantico. Sin dependencia de pytest."""
from app.intents import load_vocabulary
from app.models import SemanticOutput
from app.semantic_tree import (
    assign_evaluation_order, errors_only, format_tree, tree_metrics, validate_semantic_output,
)

NS = "http://www.w3.org/1998/Math/MathML"
# x² + 3x = 10
LATEX = "x^2 + 3x = 10"
MATHML = (f'<math xmlns="{NS}"><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo>'
          f'<mn>3</mn><mi>x</mi><mo>=</mo><mn>10</mn></math>')


def n(id, node_type, intent=None, symbol=None, ops=(), **kw):
    d = {"id": id, "node_type": node_type, "intent": intent, "symbol": symbol,
         "operands": [{"role": r, "node_id": c} for r, c in ops]}
    d.update(kw)
    return d


def make(nodes, root="n0", review=False, **kw):
    data = {"vocabulary_version": "1.0.0", "root_id": root, "nodes": nodes, "review_required": review}
    data.update(kw)
    return SemanticOutput.model_validate(data)


def equation_nodes():
    return [
        n("n0", "structure", "equation", ops=[("statement", "n1")]),
        n("n1", "relation", "equality", "=", ops=[("left", "n2"), ("right", "n9")]),
        n("n2", "operation", "addition", "+", ops=[("term", "n3"), ("term", "n6")]),
        n("n3", "operation", "power", ops=[("base", "n4"), ("exponent", "n5")]),
        n("n4", "element", "variable", "x"),
        n("n5", "element", "number", "2"),
        n("n6", "operation", "multiplication", ops=[("factor", "n7"), ("factor", "n8")], implicit=True),
        n("n7", "element", "number", "3"),
        n("n8", "element", "variable", "x"),
        n("n9", "element", "number", "10"),
    ]


def codes(problems, severity=None):
    return [p.code for p in problems if severity is None or p.severity == severity]


def check(nodes, expected_code, **kw):
    got = codes(validate_semantic_output(make(nodes, **kw), latex=LATEX, mathml=MATHML), "error")
    assert expected_code in got, f"se esperaba {expected_code}; obtenido {got}"


# ------------------------------------------------------------------ vocabulario
def test_vocabulario_cubre_el_minimo_pedido():
    v = load_vocabulary()
    need = ("variable constant number symbol parameter addition subtraction multiplication division power root "
            "factorial equality inequality less_than greater_than approximation equation expression fraction "
            "polynomial matrix vector interval function function_application composition logarithm exponential "
            "trigonometric_function derivative integral limit summation product").split()
    assert [i for i in need if i not in v.intents] == []
    assert v.version == "1.0.0"
    assert "NO son valores oficiales" in open(load_vocabulary.__globals__["VOCABULARY_PATH"], encoding="utf-8").read()


# ------------------------------------------- el ejemplo pedido: x² + 3x = 10
def test_ecuacion_distingue_simbolo_intent_y_expresion_completa():
    out = make(equation_nodes())
    problems = validate_semantic_output(out, latex=LATEX, mathml=MATHML)
    assert errors_only(problems) == [], [p.message for p in problems]
    by = {x.id: x for x in out.nodes}
    assert by["n0"].intent == "equation"                       # expresion completa: ecuacion
    assert by["n1"].intent == "equality" and by["n1"].symbol == "="   # =: igualdad
    assert by["n2"].intent == "addition" and by["n2"].symbol == "+"   # +: suma
    assert by["n3"].intent == "power"                          # x²: potenciacion
    assert by["n5"].intent == "number" and [o.role for o in by["n3"].operands] == ["base", "exponent"]  # 2: exponente
    assert by["n6"].intent == "multiplication" and by["n6"].implicit and by["n6"].symbol is None  # 3x
    assert by["n4"].intent == "variable"                       # x: variable
    assert by["n0"].intent != by["n1"].intent                  # no se confunde el intent del conjunto con el del simbolo


def test_orden_de_evaluacion_es_postorden_y_se_sobrescribe():
    nodes = equation_nodes()
    nodes[4]["evaluation_order"] = 999            # valor enviado por el "LLM"
    out = make(nodes)
    assert assign_evaluation_order(out) == 10
    by = {x.id: x for x in out.nodes}
    assert by["n0"].evaluation_order == 9 and by["n4"].evaluation_order == 0
    for node in out.nodes:
        for op in node.operands:
            assert by[op.node_id].evaluation_order < node.evaluation_order
    assert sorted(x.evaluation_order for x in out.nodes) == list(range(10))


def test_metricas_y_arbol_de_texto():
    out = make(equation_nodes())
    assign_evaluation_order(out)
    m = tree_metrics(out)
    assert m["nodes"] == 10 and m["depth"] == 5 and m["evaluation_order_assigned"] is True
    lines = format_tree(out)
    assert lines[0].startswith("- (root) [n0] equation") and any("implicito" in l for l in lines)


# ------------------------------------------------------------ errores de grafo
def test_id_duplicado():
    nodes = equation_nodes(); nodes[8]["id"] = "n7"
    check(nodes, "DUPLICATE_ID")


def test_operando_inexistente():
    nodes = equation_nodes(); nodes[3]["operands"][1]["node_id"] = "zzz"
    check(nodes, "DANGLING_OPERAND")


def test_referencia_circular():
    nodes = equation_nodes(); nodes[3]["operands"][0]["node_id"] = "n2"   # potencia -> suma -> potencia
    check(nodes, "CYCLE")


def test_padres_multiples_y_huerfanos():
    nodes = equation_nodes(); nodes[5]["operands"] = []
    nodes[6]["operands"][1]["node_id"] = "n4"      # n4 con dos padres; n8 queda huerfano
    got = codes(validate_semantic_output(make(nodes), latex=LATEX, mathml=MATHML), "error")
    assert "MULTIPLE_PARENTS" in got and "ORPHAN_NODE" in got


def test_raiz_inexistente():
    check(equation_nodes(), "ROOT_MISSING", root="nope")


# ------------------------------------------------- vocabulario, roles, propiedades
def test_intent_fuera_del_vocabulario():
    nodes = equation_nodes(); nodes[3]["intent"] = "exponentiation"
    check(nodes, "UNKNOWN_INTENT")


def test_tipo_de_nodo_incoherente_con_el_intent():
    nodes = equation_nodes(); nodes[3]["node_type"] = "relation"
    check(nodes, "TYPE_INTENT_MISMATCH")


def test_roles_y_aridad():
    nodes = equation_nodes(); nodes[3]["operands"][1]["role"] = "index"
    check(nodes, "BAD_ROLE")
    nodes = equation_nodes(); nodes[3]["operands"] = nodes[3]["operands"][:1]
    check(nodes, "MISSING_ROLE")
    nodes = equation_nodes(); nodes[2]["operands"] = nodes[2]["operands"][:1]
    check(nodes, "BAD_ARITY")


def test_limites_ambos_o_ninguno():
    nodes = [n("n0", "calculus", "integral", "∫", ops=[("integrand", "n1"), ("variable", "n2"), ("lower_bound", "n3")]),
             n("n1", "element", "variable", "x"), n("n2", "element", "variable", "x"), n("n3", "element", "number", "0")]
    got = codes(validate_semantic_output(make(nodes)), "error")
    assert "BOTH_OR_NONE" in got


def test_propiedades_requeridas_y_valores():
    nodes = [n("n0", "function", "trigonometric_function", "sin", ops=[("argument", "n1")]), n("n1", "element", "variable", "x")]
    assert "MISSING_PROPERTY" in codes(validate_semantic_output(make(nodes)), "error")
    nodes[0]["properties"] = {"function_name": "senh"}
    assert "BAD_PROPERTY" in codes(validate_semantic_output(make(nodes)), "error")
    nodes[0]["properties"] = {"function_name": "sin"}
    assert errors_only(validate_semantic_output(make(nodes), latex="sin x")) == []      # sin x: funcion trigonometrica


def test_implicit_solo_en_multiplicacion_y_sin_simbolo():
    nodes = equation_nodes(); nodes[6]["symbol"] = "·"
    check(nodes, "IMPLICIT_WITH_SYMBOL")
    nodes = equation_nodes(); nodes[2]["implicit"] = True
    check(nodes, "BAD_IMPLICIT")


def test_restricciones_estructurales():
    nodes = equation_nodes(); nodes[1].update(node_type="operation", intent="addition", symbol="+")
    nodes[1]["operands"] = [{"role": "term", "node_id": "n2"}, {"role": "term", "node_id": "n9"}]
    check(nodes, "EQUATION_NEEDS_RELATION")
    nodes = [n("n0", "function", "function_application", ops=[("function", "n1"), ("argument", "n2")], implicit=True),
             n("n1", "element", "variable", "f"), n("n2", "element", "variable", "x")]
    assert "APPLICATION_NEEDS_FUNCTION" in codes(validate_semantic_output(make(nodes)), "error")


# ---------------------------------------------------------------- ambiguedades
def ambiguous_xy(**kw):
    node = n("n0", "unresolved", None, None, ops=[("component", "n1"), ("component", "n2")],
             ambiguous=True, candidate_intents=["multiplication", "variable"], review_required=True,
             justification="xy puede ser un producto implicito o un identificador")
    node.update(kw)
    return [node, n("n1", "element", "variable", "x"), n("n2", "element", "variable", "y")]


def test_xy_ambiguo_es_valido_si_exige_revision():
    out = make(ambiguous_xy(), review=True, ambiguities=["xy"])
    assert errors_only(validate_semantic_output(out, latex="xy", mathml=f'<math xmlns="{NS}"><mi>x</mi><mi>y</mi></math>')) == []


def test_ambiguedad_mal_formada():
    for kw, code in [({"intent": "multiplication"}, "AMBIGUOUS_HAS_INTENT"),
                     ({"candidate_intents": ["multiplication"]}, "BAD_CANDIDATES"),
                     ({"candidate_intents": ["multiplication", "inventado"]}, "UNKNOWN_CANDIDATE"),
                     ({"review_required": False}, "AMBIGUOUS_NO_REVIEW"),
                     ({"node_type": "operation"}, "AMBIGUOUS_TYPE")]:
        got = codes(validate_semantic_output(make(ambiguous_xy(**kw), review=True)), "error")
        assert code in got, (code, got)


def test_marca_global_de_revision_debe_ser_coherente():
    got = codes(validate_semantic_output(make(ambiguous_xy(), review=False)), "error")
    assert "REVIEW_FLAG_INCONSISTENT" in got


def test_f_de_x_y_valor_absoluto_como_ambiguos():
    fx = [n("n0", "unresolved", None, ops=[("component", "n1"), ("component", "n2")], ambiguous=True,
            candidate_intents=["function_application", "multiplication"], review_required=True, justification="sin contexto"),
          n("n1", "element", "variable", "f"), n("n2", "element", "variable", "x")]
    assert errors_only(validate_semantic_output(make(fx, review=True))) == []
    absx = [n("n0", "unresolved", None, ops=[("component", "n1")], ambiguous=True,
              candidate_intents=["absolute_value", "group"], review_required=True, justification="|x| depende del contexto"),
            n("n1", "element", "variable", "x")]
    assert errors_only(validate_semantic_output(make(absx, review=True))) == []


def test_derivada_como_operador_sin_funcion():
    d = [n("n0", "calculus", "derivative", "d/dx", ops=[("variable", "n1")]), n("n1", "element", "variable", "x")]
    assert errors_only(validate_semantic_output(make(d))) == []


# ------------------------------------------------------- anclaje en la formula
def test_simbolo_inventado_es_error_de_anclaje():
    nodes = equation_nodes(); nodes[4]["symbol"] = "q"
    check(nodes, "GROUNDING_MISMATCH")


def test_token_sin_cubrir_es_advertencia():
    nodes = equation_nodes()
    nodes[9]["symbol"] = "3"          # el nodo del '10' ahora dice '3': el token '10' queda sin cubrir
    probs = validate_semantic_output(make(nodes), latex=LATEX, mathml=MATHML)
    uncovered = [p for p in probs if p.code == "UNCOVERED_TOKEN"]
    assert uncovered and all(p.severity == "warning" for p in uncovered)
    assert "10" in uncovered[0].message
