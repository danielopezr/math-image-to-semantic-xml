from image_to_mathml.agents.prompts import (
    direct_messages,
    hybrid_messages,
    vision_messages,
)
from image_to_mathml.models import MathContext

BASE = '<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math>'


def test_open_prompt_allows_documented_open_concepts():
    system = direct_messages(r"\Gamma(x)", MathContext())[0]["content"]
    assert "gamma-function" in system
    assert "Do NOT use open concepts" not in system


def test_allow_all_open_prompt_is_more_permissive():
    system = direct_messages(r"\Gamma(x)", MathContext(), allow_all_open_concepts=True)[
        0
    ]["content"]
    assert "Any documented W3C open concept is allowed" in system


def test_strict_core_prompt_forbids_and_omits_open_concept_example():
    system = direct_messages(r"\Gamma(x)", MathContext(), allow_open_concepts=False)[0][
        "content"
    ]
    assert "Do NOT use open concepts" in system
    assert "gamma-function($arg)" not in system


def test_hybrid_prompt_respects_strict_core():
    system = hybrid_messages(
        r"\Gamma(x)", BASE, MathContext(), allow_open_concepts=False
    )[0]["content"]
    assert "Do NOT use open concepts" in system

    normal = hybrid_messages(r"\Gamma(x)", BASE, MathContext())[0]["content"]
    assert "Do NOT use open concepts" not in normal


def test_prompts_forbid_content_mathml_and_invented_elements():
    system = direct_messages("f(x)", MathContext())[0]["content"]

    assert "Never use Content MathML" in system
    assert "<apply>" in system
    assert "<csymbol>" in system
    assert "custom elements or invented tag names" in system
    assert "Presentation MathML elements" in system


def test_prompts_forbid_deprecated_mfenced():
    system = direct_messages("f(x)", MathContext())[0]["content"]

    assert "Never use <mfenced>" in system
    assert "<mo>(</mo>" in system


def test_prompts_support_indexed_generic_function_names():
    system = direct_messages("f_{i+1}(x)", MathContext())[0]["content"]

    assert "f_{i} or f_{i+1}" in system
    assert '<msub arg="name"><mi>f</mi><mi>i</mi></msub>' in system
    assert (
        '<msub arg="name"><mi>f</mi><mrow><mi>i</mi><mo>+</mo><mn>1</mn></mrow></msub>'
        in system
    )
    assert "Never invent a\n    concept name for an indexed function" in system


def test_hybrid_prompt_examples_are_well_formed():
    system = hybrid_messages("f_{i}(x)", BASE, MathContext())[0]["content"]

    assert '<mfrac">' not in system
    assert '<mfrac intent="fraction($num,$den)">' in system
    assert '<msub arg="name"><mi>f</mi><mi>i</mi></msub>' in system


def test_vision_prompt_requests_direct_mathml_without_latex():
    system = vision_messages(MathContext())[0]["content"]

    assert "image of a mathematical equation" in system
    assert "Do not transcribe the image to LaTeX" in system
    assert "do not include any LaTeX" in system
    assert "Never use Content MathML" in system
    assert "Never use <mfenced>" in system
