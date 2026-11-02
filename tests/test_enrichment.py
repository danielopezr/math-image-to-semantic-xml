"""Pruebas del enriquecimiento opcional con atributos intent (desactivado por defecto en el pipeline)."""
import xml.etree.ElementTree as ET

from app.enrichment import canonical, enrich_mathml, load_mapping, verify_unchanged
from app.mathcat import MathCAT
from app.models import MathCATConfiguration, SemanticOutput, StageStatus
from tests.test_mathcat import FakeEngine

NS = "http://www.w3.org/1998/Math/MathML"
POW = f'<math xmlns="{NS}"><msup><mi>x</mi><mn>2</mn></msup></math>'


def tree(**node_kw):
    power = {"id": "n0", "node_type": "operation", "intent": "power", "operands": [
        {"role": "base", "node_id": "n1"}, {"role": "exponent", "node_id": "n2"}]}
    power.update(node_kw)
    return SemanticOutput.model_validate({
        "vocabulary_version": "1.0.0", "root_id": "n0", "review_required": bool(node_kw.get("review_required")),
        "nodes": [power, {"id": "n1", "node_type": "element", "intent": "variable", "symbol": "x"},
                  {"id": "n2", "node_type": "element", "intent": "number", "symbol": "2"}]})


def mc(engine=None):
    return MathCAT(MathCATConfiguration(), engine or FakeEngine())


def test_aplica_intent_y_arg_sin_cambiar_la_estructura():
    res = enrich_mathml(POW, tree(), mathcat=mc())
    assert res.status is StageStatus.OK and res.mathcat_checked and res.applied == ["n0:power->power"]
    root = ET.fromstring(res.mathml)
    msup = root[0]
    assert msup.get("intent") == "power($b,$e)" and [c.get("arg") for c in msup] == ["b", "e"]
    assert verify_unchanged(POW, res.mathml)
    assert root.tag == f"{{{NS}}}math"            # espacio de nombres correcto


def test_el_mathml_sin_atributos_nuevos_es_identico_al_original():
    res = enrich_mathml(POW, tree(), mathcat=mc())
    assert canonical(ET.fromstring(res.mathml)) == canonical(ET.fromstring(POW))


def test_sin_mathcat_se_marca_como_no_comprobado():
    res = enrich_mathml(POW, tree())
    assert res.status is StageStatus.OK and res.mathcat_checked is False
    assert "NOT_CHECKED_WITH_MATHCAT" in [w.code for w in res.warnings]


def test_mathcat_rechaza_el_resultado_y_no_se_aplica_nada():
    res = enrich_mathml(POW, tree(), mathcat=mc(FakeEngine(fail_on="set_mathml")))
    assert res.status is StageStatus.SKIPPED and res.mathml is None and res.applied == []
    assert "REJECTED_BY_MATHCAT" in [w.code for w in res.warnings]


def test_respeta_el_intent_existente_del_autor():
    ml = f'<math xmlns="{NS}"><msup intent="otro"><mi>x</mi><mn>2</mn></msup></math>'
    res = enrich_mathml(ml, tree(), mathcat=mc())
    assert res.status is StageStatus.SKIPPED and "INTENT_ALREADY_PRESENT" in [w.code for w in res.warnings]


def test_no_enriquece_nodos_dudosos_ni_desalineados():
    res = enrich_mathml(POW, tree(evidence="context_dependent"), mathcat=mc())
    assert res.applied == [] and "NODE_NOT_RELIABLE" in [w.code for w in res.warnings]
    two = f'<math xmlns="{NS}"><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><msup><mi>y</mi><mn>3</mn></msup></math>'
    assert "COUNT_MISMATCH" in [w.code for w in enrich_mathml(two, tree(), mathcat=mc()).warnings]
    other = f'<math xmlns="{NS}"><msup><mi>y</mi><mn>2</mn></msup></math>'
    assert "MISALIGNED" in [w.code for w in enrich_mathml(other, tree(), mathcat=mc()).warnings]


def test_verify_unchanged_detecta_cambios_reales():
    changed = POW.replace("<mn>2</mn>", "<mn>3</mn>")
    assert verify_unchanged(POW, POW) and not verify_unchanged(POW, changed)
    assert not verify_unchanged(POW, "<math")


def test_mapping_valido_y_marcado_como_no_verificado():
    mapping = load_mapping()
    assert mapping["status"] == "experimental" and all(r["verified"] is False for r in mapping["rules"])
    assert "NO estan verificados" in mapping["note"]
