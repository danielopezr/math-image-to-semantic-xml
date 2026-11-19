from lxml import etree

from image_to_mathml.mathml.dom import localname
from image_to_mathml.mathml.normalizer import normalize_mathml, serialize_mathml


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


def _find(root, name: str):
    for element in root.iter():
        if localname(element.tag) == name:
            return element
    return None


def test_splits_two_digit_matrix_subscripts_into_individual_mn():
    result = normalize_mathml(
        "<math><mrow><mtable><mtr><mtd>"
        "<msub><mi>a</mi><mn>11</mn></msub>"
        "</mtd></mtr></mtable></mrow></math>"
    )
    root = _root(result)
    msub = _find(root, "msub")
    assert msub is not None
    subscript = list(msub)[1]
    assert localname(subscript.tag) == "mrow"
    assert [localname(child.tag) for child in subscript] == ["mn", "mn"]
    assert [child.text for child in subscript] == ["1", "1"]
    assert "<mn>11</mn>" not in result


def test_splits_digits_in_matrix_subsup_subscript_only():
    result = normalize_mathml(
        "<math><mrow><mtable><mtr><mtd>"
        "<msubsup><mi>a</mi><mn>12</mn><mn>3</mn></msubsup>"
        "</mtd></mtr></mtable></mrow></math>"
    )
    root = _root(result)
    msubsup = _find(root, "msubsup")
    assert msubsup is not None
    subscript, superscript = list(msubsup)[1:3]
    assert [child.text for child in subscript] == ["1", "2"]
    assert localname(superscript.tag) == "mn"
    assert superscript.text == "3"


def test_does_not_split_ordinary_multi_digit_subscripts():
    result = normalize_mathml(
        "<math><mrow><msub><mi>x</mi><mn>11</mn></msub></mrow></math>"
    )
    root = _root(result)
    msub = _find(root, "msub")
    assert msub is not None
    assert localname(list(msub)[1].tag) == "mn"
    assert list(msub)[1].text == "11"


def test_does_not_split_single_digit_matrix_subscripts():
    result = normalize_mathml(
        "<math><mrow><mtable><mtr><mtd>"
        "<msub><mi>a</mi><mn>1</mn></msub>"
        "</mtd></mtr></mtable></mrow></math>"
    )
    root = _root(result)
    msub = _find(root, "msub")
    assert msub is not None
    assert localname(list(msub)[1].tag) == "mn"
    assert list(msub)[1].text == "1"


def test_matrix_subscript_attributes_move_to_the_new_mrow():
    result = normalize_mathml(
        "<math><mrow><mtable><mtr><mtd>"
        '<msub><mi>a</mi><mn arg="sub">11</mn></msub>'
        "</mtd></mtr></mtable></mrow></math>"
    )
    root = _root(result)
    msub = _find(root, "msub")
    assert msub is not None
    subscript = list(msub)[1]
    assert localname(subscript.tag) == "mrow"
    assert subscript.get("arg") == "sub"
    assert [child.get("arg") for child in subscript] == [None, None]


def test_splits_digits_inside_an_existing_attributed_mrow_subscript():
    result = normalize_mathml(
        "<math><mrow><mtable><mtr><mtd>"
        '<msub><mi>a</mi><mrow arg="sub"><mn>11</mn></mrow></msub>'
        "</mtd></mtr></mtable></mrow></math>"
    )
    root = _root(result)
    msub = _find(root, "msub")
    assert msub is not None
    subscript = list(msub)[1]
    assert localname(subscript.tag) == "mrow"
    assert subscript.get("arg") == "sub"
    assert [child.text for child in subscript] == ["1", "1"]


def test_serialization_reapplies_matrix_subscript_separation():
    root = _root(
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mtable><mtr><mtd>'
        "<msub><mi>a</mi><mn>11</mn></msub>"
        "</mtd></mtr></mtable></mrow></math>"
    )

    result = serialize_mathml(root)

    assert "<mn>11</mn>" not in result
    assert "<mn>1</mn>" in result
    msub = _find(root, "msub")
    assert msub is not None
    assert localname(list(msub)[1].tag) == "mrow"
    assert [child.text for child in list(msub)[1]] == ["1", "1"]


def _all(root, name: str):
    return [element for element in root.iter() if localname(element.tag) == name]


def test_splits_matrix_indexes_referenced_outside_the_table():
    result = normalize_mathml(
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>'
        "<mtable><mtr><mtd><msub><mi>a</mi><mn>11</mn></msub></mtd></mtr></mtable>"
        "<mo>+</mo>"
        "<msub><mi>a</mi><mn>11</mn></msub>"
        "</mrow></math>"
    )
    root = _root(result)
    subscripts = _all(root, "msub")
    assert len(subscripts) == 2
    for msub in subscripts:
        subscript = list(msub)[1]
        assert localname(subscript.tag) == "mrow"
        assert [child.text for child in subscript] == ["1", "1"]
    assert "<mn>11</mn>" not in result


def test_splits_matrix_reference_that_appears_before_the_table():
    result = normalize_mathml(
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>'
        "<msub><mi>a</mi><mn>11</mn></msub>"
        "<mo>+</mo>"
        "<mtable><mtr><mtd><msub><mi>a</mi><mn>11</mn></msub></mtd></mtr></mtable>"
        "</mrow></math>"
    )
    subscripts = _all(_root(result), "msub")
    for msub in subscripts:
        assert localname(list(msub)[1].tag) == "mrow"
        assert [child.text for child in list(msub)[1]] == ["1", "1"]


def test_does_not_split_nonmatching_matrix_references():
    result = normalize_mathml(
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>'
        "<mtable><mtr><mtd><msub><mi>a</mi><mn>11</mn></msub></mtd></mtr></mtable>"
        "<mo>+</mo>"
        "<msub><mi>a</mi><mn>12</mn></msub>"
        "<mo>+</mo>"
        "<msub><mi>b</mi><mn>11</mn></msub>"
        "<mo>+</mo>"
        "<msub><mi>a</mi><mn>13</mn></msub>"
        "</mrow></math>"
    )
    root = _root(result)
    subscripts = _all(root, "msub")
    assert localname(list(subscripts[0])[1].tag) == "mrow"
    for msub in subscripts[1:]:
        assert localname(list(msub)[1].tag) == "mn"
        assert list(msub)[1].text in {"12", "11", "13"}


def test_repeated_correction_rebuilds_indexes_from_already_split_matrix():
    root = _root(
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>'
        "<mtable><mtr><mtd><msub><mi>a</mi>"
        "<mrow><mn>1</mn><mn>1</mn></mrow>"
        "</msub></mtd></mtr></mtable>"
        "<mo>+</mo>"
        "<msub><mi>a</mi><mn>11</mn></msub>"
        "</mrow></math>"
    )

    result = serialize_mathml(root)

    assert "<mn>11</mn>" not in result
    subscripts = _all(root, "msub")
    for msub in subscripts:
        assert [child.text for child in list(msub)[1]] == ["1", "1"]


def test_keeps_empty_mrow_that_holds_a_positional_slot():
    from image_to_mathml.mathml.converter import LatexToMathML
    from image_to_mathml.mathml.validator import validate_mathml

    for latex in (r"\frac{}{b}", r"{}^{2}H"):
        result = normalize_mathml(LatexToMathML().convert(latex))
        report = validate_mathml(result)
        assert not any(issue.code == "structure.arity" for issue in report.errors), (
            latex,
            result,
        )


def test_removing_an_empty_mrow_keeps_the_following_text():
    result = normalize_mathml(
        "<math><mrow><mi>a</mi><mrow/>tail<mi>b</mi></mrow></math>"
    )
    assert "tail" in result


def test_strips_reasoning_blocks_before_extracting_mathml():
    answer = "<think>maybe <math><mi>wrong</mi></math></think><math><mi>x</mi></math>"
    assert "wrong" not in normalize_mathml(answer)


def test_serialization_reindents_inserted_elements_and_keeps_token_text():
    from image_to_mathml.mathml.dom import new_element
    from image_to_mathml.mathml.normalizer import parse_and_normalize

    root = parse_and_normalize(
        normalize_mathml(
            "<math><mrow><mi>f</mi><mo>=</mo><mtext> si y solo si </mtext></mrow></math>"
        )
    )
    mrow = next(iter(root))
    wrapper = new_element("mrow")
    wrapper.set("intent", "x")
    mrow.insert(0, wrapper)
    wrapper.append(list(mrow)[1])

    result = serialize_mathml(root)

    assert '<mrow intent="x">\n' in result
    assert "<mtext> si y solo si </mtext>" in result


def test_infinity_is_an_identifier_not_an_operator():
    result = normalize_mathml('<math><mo stretchy="false" arg="b">∞</mo></math>')
    assert '<mi arg="b">∞</mi>' in result
    assert "<mo" not in result


def test_an_ellipsis_is_an_operator_not_an_identifier():
    result = normalize_mathml('<math><mi arg="x">…</mi><mi>⋯</mi></math>')
    assert '<mo arg="x">…</mo>' in result
    assert "<mo>⋯</mo>" in result
    assert "<mi" not in result


def test_table_cells_with_several_children_get_one_mrow():
    root = _root(
        normalize_mathml(
            '<math xmlns="http://www.w3.org/1998/Math/MathML"><mtable><mtr>'
            "<mtd><mn>2</mn><mi>x</mi><mo>=</mo><mn>1</mn></mtd>"
            "<mtd><mi>y</mi></mtd></mtr></mtable></math>"
        )
    )
    cells = [element for element in root.iter() if localname(element.tag) == "mtd"]
    first, second = cells
    assert [localname(child.tag) for child in first] == ["mrow"]
    assert len(first[0]) == 4
    assert [localname(child.tag) for child in second] == ["mi"]


def test_a_denominator_left_with_only_the_attached_power_loses_its_mrow():
    # latex2mathml for \frac{1}{[u(t)]^{2}}: the 2 hangs on the closing "]".
    root = _root(
        normalize_mathml(
            '<math xmlns="http://www.w3.org/1998/Math/MathML"><mfrac><mn>1</mn>'
            "<mrow><mo>[</mo><mi>u</mi><mo>(</mo><mi>t</mi><mo>)</mo>"
            "<msup><mo>]</mo><mn>2</mn></msup></mrow></mfrac></math>"
        )
    )
    frac = next(element for element in root.iter() if localname(element.tag) == "mfrac")
    assert [localname(child.tag) for child in frac] == ["mn", "msup"]
