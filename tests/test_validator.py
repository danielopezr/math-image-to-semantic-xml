from image_to_mathml.mathml.concepts import get_registry
from image_to_mathml.mathml.normalizer import parse_and_normalize, serialize_mathml
from image_to_mathml.mathml.validator import prune_orphan_args, validate_mathml

NS = 'xmlns="http://www.w3.org/1998/Math/MathML"'


def test_generic_function_is_valid():
    markup = (
        f'<math {NS}><mrow intent="$name($var)">'
        '<mi arg="name">f</mi><mo>(</mo><mi arg="var">x</mi><mo>)</mo>'
        "</mrow></math>"
    )
    report = validate_mathml(markup)
    assert report.ok
    assert report.acceptable
    assert report.intent_count == 1


def test_missing_arg_is_an_error():
    markup = (
        f'<math {NS}><mrow intent="fraction($num,$den)">'
        "<mi>a</mi><mi>b</mi></mrow></math>"
    )
    report = validate_mathml(markup)
    assert not report.ok
    assert any(issue.code == "intent.unresolved-reference" for issue in report.errors)


def test_unknown_concept_is_an_error():
    markup = (
        f'<math {NS}><mrow intent="invented-concept($x)">'
        '<mi arg="x">x</mi></mrow></math>'
    )
    report = validate_mathml(markup)
    assert not report.ok
    assert any(issue.code == "intent.unknown-concept" for issue in report.errors)


def test_gamma_function_is_recognized_open_concept():
    markup = (
        f'<math {NS}><mrow intent="gamma-function($arg)">'
        '<mi arg="arg">x</mi></mrow></math>'
    )
    report = validate_mathml(markup, allow_open_concepts=True)
    assert report.ok
    assert any(issue.code == "intent.open-concept" for issue in report.warnings)

    strict = validate_mathml(markup, allow_open_concepts=False)
    assert not strict.ok


def test_open_concept_policy_can_allow_only_documented_exceptions():
    gamma = (
        f'<math {NS}><mrow intent="gamma-function($arg)">'
        '<mi arg="arg">x</mi></mrow></math>'
    )
    airy = (
        f'<math {NS}><mrow intent="airy-function($arg)">'
        '<mi arg="arg">x</mi></mrow></math>'
    )
    policy = {"gamma-function", "beta-function", "error-function"}
    assert validate_mathml(gamma, allowed_open_concepts=policy).ok
    restricted = validate_mathml(airy, allowed_open_concepts=policy)
    assert not restricted.ok
    assert any(
        issue.code == "intent.open-concept-not-allowed" for issue in restricted.errors
    )
    assert validate_mathml(airy, allowed_open_concepts=None).ok


def test_display_on_root_is_an_error():
    markup = f'<math {NS} display="block"><mrow><mi>x</mi></mrow></math>'
    report = validate_mathml(markup)
    assert not report.ok
    assert any(issue.code == "structure.display" for issue in report.errors)


def test_root_must_have_a_single_mrow():
    markup = f"<math {NS}><mi>x</mi><mi>y</mi></math>"
    report = validate_mathml(markup)
    assert not report.ok
    assert any(issue.code == "structure.mrow" for issue in report.errors)


def test_arity_is_checked():
    markup = (
        f'<math {NS}><mrow><mfrac intent="fraction($a)">'
        '<mi arg="a">a</mi><mi>b</mi></mfrac></mrow></math>'
    )
    report = validate_mathml(markup)
    assert not report.ok
    assert any(issue.code == "intent.arity" for issue in report.errors)


def test_unused_arg_is_a_warning():
    markup = f'<math {NS}><mrow><mi arg="unused">x</mi></mrow></math>'
    report = validate_mathml(markup)
    assert report.ok
    assert any(issue.code == "arg.unused" for issue in report.warnings)


def test_orphan_args_are_pruned():
    markup = (
        f'<math {NS}><mrow arg="unused">'
        '<mfrac intent="fraction($num,$den)"><mi arg="num">a</mi>'
        '<mi arg="den">b</mi></mfrac></mrow></math>'
    )
    root = parse_and_normalize(markup)
    assert prune_orphan_args(root) == 1
    result = serialize_mathml(root)
    assert 'arg="unused"' not in result
    assert 'arg="num"' in result
    assert 'arg="den"' in result


def test_registry_has_core_and_open_concepts():
    registry = get_registry()
    assert registry.classification("fraction") == "core"
    assert registry.classification("gamma-function") == "open"
    assert registry.classification("_my_literal") == "literal"
    assert registry.classification("definitely-not-a-concept") == "unknown"
    assert registry.accepts_arity("fraction", 2)
    assert not registry.accepts_arity("fraction", 1)
    assert registry.accepts_arity("list", 0)
