"""Pruebas del motor de reglas. Los ejemplos del catalogo se ejecutan como pruebas.
Sin dependencia de pytest."""
import copy

from app import intent_rules
from app.intent_rules import CatalogError, evaluate_intents, load_catalog, validate_catalog
from app.models import (
    RuleOrigin, RuleOutcome, SemanticNode, SemanticOutput, SemanticRelation,
    SemanticResult, StageStatus,
)


def node_from_dict(d):
    return SemanticNode(
        id=d["id"], kind=d["kind"], symbol=d.get("symbol"),
        evidence=d.get("evidence", "directly_supported"),
        children=[node_from_dict(c) for c in d.get("children", [])],
    )


def result_from_dict(d):
    out = SemanticOutput(
        expression_type=d["expression_type"],
        root=node_from_dict(d["root"]),
        relations=[SemanticRelation(**r) for r in d["relations"]],
        assumptions=d["assumptions"], ambiguities=d["ambiguities"],
        requires_review=d["requires_review"],
    )
    return SemanticResult(status=StageStatus.OK, output=out)


def outcomes(results):
    return {r.rule_id: r.outcome for r in results}


def test_catalogo_valido_y_completo():
    catalog = load_catalog()
    assert catalog["catalog_version"]
    assert len(catalog["rules"]) == len(intent_rules.IMPLEMENTATIONS)
    for rule in catalog["rules"]:
        assert rule["examples"], rule["id"]


def test_ejemplos_del_catalogo():
    catalog = load_catalog()
    fx = catalog["fixtures"]
    failures = []
    for rule in catalog["rules"]:
        for ex in rule["examples"]:
            sem = None if ex["semantic"] is None else result_from_dict(fx["semantic"][ex["semantic"]])
            got = outcomes(evaluate_intents(fx["mathml"][ex["mathml"]], sem, catalog=catalog))[rule["id"]]
            if got.value != ex["expected"]:
                failures.append(f"{rule['id']} / {ex['name']}: esperado {ex['expected']}, obtenido {got.value}")
    assert not failures, "\n".join(failures)


def test_devuelve_un_resultado_por_regla_sin_puntuacion_agregada():
    results = evaluate_intents("<math/>", None)
    assert len(results) == len(load_catalog()["rules"])
    assert all(isinstance(r.outcome, RuleOutcome) for r in results)


def test_sin_datos_nunca_hay_pass_semantico():
    # Sin interpretacion, ninguna regla semantica puede dar PASS
    res = outcomes(evaluate_intents(load_catalog()["fixtures"]["mathml"]["eq"], None))
    for rid in ("SEM-002", "SEM-003", "SEM-004", "SEM-005", "SEM-006", "SEM-007"):
        assert res[rid] is RuleOutcome.NOT_APPLICABLE, rid
    assert res["SEM-001"] is RuleOutcome.REVIEW


def test_interpretacion_fallida_se_trata_como_ausente():
    failed = SemanticResult(status=StageStatus.FAILED)
    res = outcomes(evaluate_intents(load_catalog()["fixtures"]["mathml"]["eq"], failed))
    assert res["SEM-001"] is RuleOutcome.REVIEW


def test_origenes_separados():
    origins = {r["id"]: r["origin"] for r in load_catalog()["rules"]}
    assert {o for i, o in origins.items() if i.startswith("MML-001")} == {"w3c_normative"}
    assert origins["SEM-007"] == RuleOrigin.LLM_DERIVED.value
    assert origins["MML-003"] == RuleOrigin.PROJECT_HEURISTIC.value  # intent es borrador: no es normativo


def test_error_interno_de_una_regla_da_review():
    original = intent_rules.IMPLEMENTATIONS["SEM-001"]

    def boom(ctx):
        raise RuntimeError("fallo")

    intent_rules.IMPLEMENTATIONS["SEM-001"] = boom
    try:
        res = outcomes(evaluate_intents("<math/>", None))
    finally:
        intent_rules.IMPLEMENTATIONS["SEM-001"] = original
    assert res["SEM-001"] is RuleOutcome.REVIEW


def test_catalogo_rechaza_inconsistencias():
    base = load_catalog()

    dup = copy.deepcopy(base); dup["rules"].append(copy.deepcopy(dup["rules"][0]))
    bad_origin = copy.deepcopy(base); bad_origin["rules"][0]["origin"] = "inventado"
    no_examples = copy.deepcopy(base); no_examples["rules"][0]["examples"] = []
    unknown = copy.deepcopy(base); unknown["rules"][0]["id"] = "XXX-999"
    bad_expected = copy.deepcopy(base); bad_expected["rules"][0]["examples"][0]["expected"] = "MAYBE"
    for broken in (dup, bad_origin, no_examples, unknown, bad_expected):
        try:
            validate_catalog(broken)
        except CatalogError:
            continue
        raise AssertionError("el catalogo inconsistente no fue rechazado")
