import pytest

from image_to_mathml.mathml.intent_parser import IntentSyntaxError, parse_intent


def test_parses_application_and_references():
    node = parse_intent("fraction($num,$den)")
    assert node.kind == "application"
    assert node.children[0].kind == "name"
    assert node.children[0].value == "fraction"
    assert [child.value for child in node.children[1:]] == ["num", "den"]


def test_parses_generic_function_reference_head():
    node = parse_intent("$name($var)")
    assert node.kind == "application"
    assert node.children[0].kind == "reference"
    assert node.children[0].value == "name"
    assert node.children[1].kind == "reference"
    assert node.children[1].value == "var"


def test_parses_self_properties():
    node = parse_intent(":prefix:pause-medium")
    assert node.kind == "self"
    assert node.properties == ["prefix", "pause-medium"]


def test_parses_head_properties():
    node = parse_intent("power:infix($base,$exp)")
    assert node.children[0].properties == ["infix"]


def test_parses_numbers_and_negative_numbers():
    node = parse_intent("root($radicand, 2)")
    assert node.children[2].kind == "number"
    assert node.children[2].value == "2"
    node = parse_intent("f(-1.5)")
    assert node.children[1].value == "-1.5"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "fraction(",
        ":prefix fraction($x)",
        "fraction($x,)",
        "$",
        "foo($x))",
    ],
)
def test_rejects_invalid_grammar(value):
    with pytest.raises(IntentSyntaxError):
        parse_intent(value)
