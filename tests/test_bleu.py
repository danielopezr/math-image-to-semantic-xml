import pytest

from image_to_mathml.metrics.bleu import (
    MathMLNormalizationError,
    canonical_mathml_tokens,
    corpus_bleu,
    mathml_bleu,
    mathml_corpus_bleu,
    sentence_bleu,
)

NS = 'xmlns="http://www.w3.org/1998/Math/MathML"'


def _math(body: str) -> str:
    return f"<math {NS}><mrow>{body}</mrow></math>"


def test_argument_names_are_canonicalised_for_the_intent_metric():
    reference = _math(
        '<msup intent="power($base,$exp)">'
        '<mi arg="base">x</mi><mi arg="exp">2</mi></msup>'
    )
    candidate = _math(
        '<msup intent="power($b, $e)">'
        '<mi mathvariant="italic" arg="b">x</mi>'
        '<mspace width="1em"/>'
        '<mi arg="e" stretchy="false">2</mi></msup>'
    )

    result = mathml_bleu(reference, candidate)

    assert result.structure.score == 1.0
    assert result.intent.score == 1.0


def test_visual_helpers_and_wrappers_are_ignored():
    reference = _math("<msqrt><mi>x</mi></msqrt>")
    candidate = _math(
        '<mpadded width="2em"><mstyle mathvariant="bold">'
        '<mspace width="1em"/><msqrt><mi>x</mi></msqrt>'
        "</mstyle></mpadded>"
    )

    result = mathml_bleu(reference, candidate)

    assert result.structure.score == 1.0
    assert result.intent.score == 1.0


def test_semantics_annotations_are_not_part_of_the_comparison():
    reference = _math("<msqrt><mi>x</mi></msqrt>")
    candidate = (
        f"<math {NS}><semantics><mrow><msqrt><mi>x</mi></msqrt></mrow>"
        '<annotation encoding="application/x-tex">\\sqrt{x}</annotation>'
        "</semantics></math>"
    )

    result = mathml_bleu(reference, candidate)

    assert result.structure.score == 1.0
    assert result.intent.score == 1.0


def test_missing_top_level_mrow_is_added_for_comparison():
    wrapped = _math("<mfrac><mi>a</mi><mi>b</mi></mfrac>")
    unwrapped = f"<math {NS}><mfrac><mi>a</mi><mi>b</mi></mfrac></math>"

    assert canonical_mathml_tokens(wrapped) == canonical_mathml_tokens(unwrapped)
    assert mathml_bleu(wrapped, unwrapped).structure.score == 1.0


def test_multiple_top_level_nodes_are_wrapped_once():
    reference = f"<math {NS}><mi>x</mi><mo>+</mo><mi>y</mi></math>"
    candidate = _math("<mi>x</mi><mo>+</mo><mi>y</mi>")

    assert canonical_mathml_tokens(reference) == canonical_mathml_tokens(candidate)
    assert mathml_bleu(reference, candidate).structure.score == 1.0


def test_existing_mrow_is_not_wrapped_again():
    tokens = canonical_mathml_tokens(_math("<mfrac><mi>a</mi><mi>b</mi></mfrac>"))

    assert tokens[:3] == ["<math>", "<mrow>", "<mfrac>"]
    assert tokens.count("<mrow>") == 1


def test_structure_metric_removes_intent_and_argument_attributes():
    reference = _math("<msup><mi>x</mi><mi>2</mi></msup>")
    candidate = _math(
        '<msup intent="power($base,$exp)">'
        '<mi arg="base">x</mi><mi arg="exp">2</mi></msup>'
    )

    result = mathml_bleu(reference, candidate)

    assert result.structure.score == 1.0
    assert result.intent.score < 1.0


def test_different_structure_reduces_the_structure_score():
    reference = _math("<mfrac><mi>a</mi><mi>b</mi></mfrac>")
    candidate = _math("<msqrt><mi>a</mi></msqrt>")

    result = mathml_bleu(reference, candidate)

    assert result.structure.score < 1.0
    assert result.intent.score < 1.0


def test_intent_argument_enumeration_uses_first_appearance_order():
    tokens = canonical_mathml_tokens(
        _math(
            '<mfrac intent="fraction($num,$den)">'
            '<mi arg="num">a</mi><mi arg="den">b</mi></mfrac>'
        )
    )

    assert '<mfrac intent="fraction($arg1,$arg2)">' in tokens
    assert '<mi arg="arg1">' in tokens
    assert '<mi arg="arg2">' in tokens


def test_corpus_bleu_aggregates_several_equations():
    references = [
        _math("<msup><mi>x</mi><mi>2</mi></msup>"),
        _math("<mfrac><mi>a</mi><mi>b</mi></mfrac>"),
    ]
    candidates = [
        _math("<msup><mi>x</mi><mi>2</mi></msup>"),
        _math("<mfrac><mi>a</mi><mi>c</mi></mfrac>"),
    ]

    result = mathml_corpus_bleu(references, candidates)

    assert 0.0 < result.structure.score < 1.0
    assert 0.0 < result.intent.score < 1.0


def test_sentence_bleu_is_one_for_identical_sequences():
    score = sentence_bleu(["a", "b", "c", "d"], ["a", "b", "c", "d"])

    assert score.score == 1.0
    assert score.brevity_penalty == 1.0


def test_corpus_bleu_requires_equal_list_lengths():
    with pytest.raises(ValueError):
        corpus_bleu([["a"]], [["a"], ["b"]])


def test_invalid_mathml_raises_a_normalisation_error():
    with pytest.raises(MathMLNormalizationError):
        canonical_mathml_tokens("<math><mrow></math>")


def test_numeric_character_references_match_the_symbol_itself():
    reference = _math("<mo>&#x2212;</mo><mi>x</mi>")
    candidate = _math("<mo>\u2212</mo><mi>x</mi>")

    assert canonical_mathml_tokens(reference) == canonical_mathml_tokens(candidate)
    assert mathml_bleu(reference, candidate).structure.score == 1.0


def test_keyboard_hyphen_and_mathematical_minus_are_equivalent():
    reference = _math("<mn>-1</mn>")
    candidate = _math("<mn>\u22121</mn>")

    assert canonical_mathml_tokens(reference) == canonical_mathml_tokens(candidate)
    assert mathml_bleu(reference, candidate).structure.score == 1.0


def test_double_escaped_character_references_are_decoded():
    reference = _math("<mo>&amp;#x222B;</mo>")
    candidate = _math("<mo>\u222b</mo>")

    assert canonical_mathml_tokens(reference) == canonical_mathml_tokens(candidate)
    assert mathml_bleu(reference, candidate).structure.score == 1.0


def test_named_entities_are_accepted():
    reference = _math("<mo>&minus;</mo><mn>1</mn>")
    candidate = _math("<mo>\u2212</mo><mn>1</mn>")

    assert canonical_mathml_tokens(reference) == canonical_mathml_tokens(candidate)
    assert mathml_bleu(reference, candidate).structure.score == 1.0


def test_math_italic_letters_match_plain_greek():
    reference = _math("<mi>&#x1D706;</mi>")
    candidate = _math("<mi>\u03bb</mi>")

    assert canonical_mathml_tokens(reference) == canonical_mathml_tokens(candidate)
    assert mathml_bleu(reference, candidate).structure.score == 1.0


def test_equivalent_symbol_variants_are_unified():
    reference = _math("<mo>\u00b7</mo><mo>\u2a7d</mo>")
    candidate = _math("<mo>\u22c5</mo><mo>\u2264</mo>")

    assert canonical_mathml_tokens(reference) == canonical_mathml_tokens(candidate)
    assert mathml_bleu(reference, candidate).structure.score == 1.0


def test_distinct_symbols_are_not_unified():
    reference = _math("<mo>+</mo>")
    candidate = _math("<mo>\u2212</mo>")

    assert mathml_bleu(reference, candidate).structure.score < 1.0
