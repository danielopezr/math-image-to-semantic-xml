"""Prompts shared by the language-model agents."""

from __future__ import annotations

from ..mathml.concepts import get_registry
from ..models import MathContext

_CORE_NAMES = ", ".join(get_registry().names())


def _structure_rules(
    allow_open_concepts: bool, allow_all_open_concepts: bool = False
) -> str:
    function_rule = """9. A recognised named function must use its core concept
   name instead of a generic application (for example sine for sin)."""
    if not allow_open_concepts:
        concept_rule = f"""8. Use only the following W3C intent core concept names.
   Do NOT use open concepts. Core names are:
   {_CORE_NAMES}"""
    elif allow_all_open_concepts:
        concept_rule = f"""8. Use only names from the W3C intent registries. Do not
   invent names. Core names are:
   {_CORE_NAMES}
   Any documented W3C open concept is allowed when it is the correct meaning,
   but prefer a core concept whenever one exists."""
    else:
        concept_rule = f"""8. Use only names from the W3C intent registries. Do not
   invent names. Core names are:
   {_CORE_NAMES}
   The only open concepts allowed by the current policy are gamma-function,
   beta-function and error-function. Do not use any other open concept."""

    return f"""Hard output contract:
1. Return ONLY one MathML document. No Markdown fences, no comments, no explanation.
2. The root must be <math xmlns="http://www.w3.org/1998/Math/MathML">.
3. The <math> root MUST NOT have a display attribute.
4. The <math> root MUST contain exactly one <mrow> and that <mrow> must wrap the
   entire expression.
5. Preserve the mathematical structure and every symbol faithfully.
6. Do not wrap simple tokens in an unnecessary <mrow>. Use <mrow> only for real
   grouping or when the <mrow> carries intent/arg attributes.
7. Every $reference in an intent must have a matching arg="reference" on a
   descendant element.
{concept_rule}
{function_rule}
10. A generic function application such as f(x) must use
    intent="$name($var)" on the enclosing <mrow>, with arg="name" on the <mi>
    that represents f and arg="var" on the element that represents x.
11. Put intent and arg on the smallest element that owns the meaning. Prefer
    <mfrac intent="fraction($num,$den)"> rather than an extra wrapper.
12. Only add intent attributes for operators tag <mo> when the context needs it for
    avoid any ambiguity, prefer the default fallback instead, e.g. do not add intent for
    equal, plus, minus and other common basic operations when the identified semantic meaning 
    is the default. Example:
    Avoid: <mo intent="equals">=</mo>, prefers <mo>=</mo>, add intent when the context requires
    a diferent meaning from the defualt equals. Use only the intent fraction($num, den) when the
    representation doesn't use the <mfrac> tag, for example for presentation in the form a/b, in
    order to guarantee the correct reading from the reading tool as 'a over b'.
13. It is okey if for one simple equation the final output MathML document doesn't have any intent.
"""


def _direct_system(
    allow_open_concepts: bool, allow_all_open_concepts: bool = False
) -> str:
    gamma_example = ""
    if allow_open_concepts:
        gamma_example = """
LaTeX: \\Gamma(x)
<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow intent="gamma-function($arg)"><mi>\u0393</mi><mo>(</mo><mi arg="arg">x</mi><mo>)</mo></mrow></math>
"""
    return (
        "You are a meticulous MathML 4 accessibility engineer. You convert LaTeX "
        "equations into Presentation MathML annotated with W3C intent attributes "
        "for screen readers.\n"
        + _structure_rules(allow_open_concepts, allow_all_open_concepts)
        + f"""
Examples (output the full XML, not an ellipsis):
LaTeX: \\frac{{a}}{{b}}
<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mi>a</mi><mi>b</mi></mfrac></mrow></math>

LaTeX: f(x)
<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow intent="$name($var)"><mi arg="name">f</mi><mo>(</mo><mi arg="var">x</mi><mo>)</mo></mrow></math>
{gamma_example}
LaTeX: \\sin(x)
<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow intent="sine($arg)"><mi>sin</mi><mo>(</mo><mi arg="arg">x</mi><mo>)</mo></mrow></math>
"""
    )


def _hybrid_system(
    allow_open_concepts: bool, allow_all_open_concepts: bool = False
) -> str:
    return (
        "You are a MathML accessibility annotator. You receive correct Presentation "
        "MathML and the LaTeX it came from. Your ONLY task is to add or correct "
        "intent and arg attributes.\n"
        "You MUST keep every element, its order, its text and its grouping exactly "
        "as given. Never add, remove, move or rename elements. If you cannot "
        "determine an intent safely, leave the element unchanged.\n"
        + _structure_rules(allow_open_concepts, allow_all_open_concepts)
        + """
Example:
Base MathML:
<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mi>a</mi><mi>b</mi></mfrac></mrow></math>
Annotated MathML:
<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac"><mi>a</mi><mi>b</mi></mfrac></mrow></math>

Base MathML:
<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo>(</mo><mi>x</mi><mo>)</mo></mrow></math>
Annotated MathML:
<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow intent="$name($var)"><mi arg="name">f</mi><mo>(</mo><mi arg="var">x</mi><mo>)</mo></mrow></math>
"""
    )


def _context_block(context: MathContext) -> str:
    return (
        "Textbook context (use it only to disambiguate notation):\n"
        + context.prompt_lines()
    )


def _feedback_block(feedback: list[str] | None) -> str:
    if not feedback:
        return ""
    lines = "\n".join(f"- {item}" for item in feedback[:12])
    return (
        "\nYour previous output failed validation with these problems:\n"
        f"{lines}\n"
        "Return a complete corrected MathML document. Do not explain.\n"
    )


def direct_messages(
    latex: str,
    context: MathContext,
    feedback: list[str] | None = None,
    *,
    allow_open_concepts: bool = True,
    allow_all_open_concepts: bool = False,
) -> list[dict[str, str]]:
    user = (
        "Convert this LaTeX equation to accessible MathML with intent attributes.\n\n"
        f"LaTeX:\n{latex}\n\n{_context_block(context)}\n"
        f"{_feedback_block(feedback)}"
    )
    return [
        {
            "role": "system",
            "content": _direct_system(allow_open_concepts, allow_all_open_concepts),
        },
        {"role": "user", "content": user},
    ]


def hybrid_messages(
    latex: str,
    base_mathml: str,
    context: MathContext,
    feedback: list[str] | None = None,
    *,
    allow_open_concepts: bool = True,
    allow_all_open_concepts: bool = False,
) -> list[dict[str, str]]:
    user = (
        "Add W3C intent/arg attributes to the following MathML. Keep the "
        "structure exactly the same.\n\n"
        f"LaTeX:\n{latex}\n\n"
        f"Base MathML:\n{base_mathml}\n\n"
        f"{_context_block(context)}\n"
        f"{_feedback_block(feedback)}"
    )
    return [
        {
            "role": "system",
            "content": _hybrid_system(allow_open_concepts, allow_all_open_concepts),
        },
        {"role": "user", "content": user},
    ]
