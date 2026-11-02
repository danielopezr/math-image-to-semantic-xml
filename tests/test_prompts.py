from image_to_mathml.agents.prompts import direct_messages, hybrid_messages
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
