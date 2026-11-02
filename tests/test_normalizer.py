from lxml import etree

from image_to_mathml.mathml.dom import localname
from image_to_mathml.mathml.normalizer import normalize_mathml


def _root(markup: str):
    return etree.fromstring(markup.encode("utf-8"))


def test_wraps_expression_in_one_top_level_mrow():
    result = normalize_mathml("<math><mi>x</mi><mo>+</mo><mn>1</mn></math>")
    root = _root(result)
    assert localname(root.tag) == "math"
    children = list(root)
    assert len(children) == 1
    assert localname(children[0].tag) == "mrow"


def test_removes_display_attribute():
    result = normalize_mathml('<math display="block"><mrow><mi>x</mi></mrow></math>')
    assert "display=" not in result


def test_adds_mathml_namespace():
    result = normalize_mathml("<math><mrow><mi>x</mi></mrow></math>")
    assert 'xmlns="http://www.w3.org/1998/Math/MathML"' in result


def test_unwraps_redundant_mrows():
    result = normalize_mathml("<math><mrow><mrow><mi>x</mi></mrow></mrow></math>")
    root = _root(result)
    mrow = root[0]
    assert localname(mrow.tag) == "mrow"
    assert [localname(child.tag) for child in mrow] == ["mi"]


def test_keeps_mrow_that_carries_intent():
    result = normalize_mathml(
        '<math><mrow intent="$name($var)">'
        '<mi arg="name">f</mi><mo>(</mo><mi arg="var">x</mi><mo>)</mo>'
        "</mrow></math>"
    )
    root = _root(result)
    assert root[0].get("intent") == "$name($var)"


def test_extracts_mathml_from_markdown_and_prose():
    raw = """Here is the result:
```xml
<math><mrow><mi>x</mi></mrow></math>
```
Done.
"""
    result = normalize_mathml(raw)
    assert "<mi>x</mi>" in result


def test_fraction_mrow_children_are_unwrapped():
    result = normalize_mathml(
        "<math><mrow><mfrac><mrow><mi>a</mi></mrow>"
        "<mrow><mi>b</mi></mrow></mfrac></mrow></math>"
    )
    root = _root(result)
    mfrac = root[0][0]
    assert [localname(child.tag) for child in mfrac] == ["mi", "mi"]
