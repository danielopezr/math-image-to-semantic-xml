"""Prompts shared by the language-model agents."""

from __future__ import annotations

from ..mathml.concepts import get_registry
from ..models import MathContext

_CORE_NAMES = ", ".join(get_registry().names())


def _structure_rules(
    allow_open_concepts: bool, allow_all_open_concepts: bool = False
) -> str:
    function_rule = """11. A recognised named function must use its core concept
   name instead of a generic application (for example sine for sin)."""
    if not allow_open_concepts:
        concept_rule = f"""10. Use only the following W3C intent core concept names.
   Do NOT use open concepts. Core names are:
   {_CORE_NAMES}"""
    elif allow_all_open_concepts:
        concept_rule = f"""10. Use only names from the W3C intent registries. Do not
   invent names. Core names are:
   {_CORE_NAMES}
   Any documented W3C open concept is allowed when it is the correct meaning,
   but prefer a core concept whenever one exists."""
    else:
        concept_rule = f"""10. Use only names from the W3C intent registries. Do not
   invent names. Core names are:
   {_CORE_NAMES}
   The only open concepts allowed by the current policy are gamma-function,
   beta-function and error-function. Do not use any other open concept."""

    return f"""Hard output contract:
1. Return ONLY one MathML Presentation document. Never use Content MathML
   elements such as <apply>, <ci>, <cn>, <csymbol>, <lambda>, <bvar>,
   <condition>, <declare>, <reln>, <fn> or <domainofapplication>.
   No Markdown fences, no comments, no explanation.
2. The root must be <math xmlns="http://www.w3.org/1998/Math/MathML">.
   All elements must be in that MathML namespace. Do not use prefixes, foreign
   namespaces, custom elements or invented tag names.
3. Use only these Presentation MathML elements: <math>, <mrow>, <mi>, <mn>,
   <mo>, <mtext>, <ms>, <mspace>, <mglyph>, <mfrac>, <msqrt>, <mroot>,
   <mstyle>, <merror>, <mpadded>, <mphantom>, <menclose>, <msub>, <msup>,
   <msubsup>, <munder>, <mover>, <munderover>, <mmultiscripts>, <mprescripts>,
   <none>, <mtable>, <mtr>, <mtd>, <mlabeledtr>, <maction>. Never emit
   <semantics>, <annotation>, <annotation-xml> or any Content MathML element.
4. Never use <mfenced>; it is deprecated. Write parentheses, brackets and
   braces explicitly with <mo> tokens, for example <mo>(</mo> and <mo>)</mo>.
5. The <math> root MUST NOT have a display attribute.
6. The <math> root MUST contain exactly one <mrow> and that <mrow> must wrap the
   entire expression.
7. Preserve the mathematical structure and every symbol faithfully.
8. Do not wrap simple tokens in an unnecessary <mrow>. Use <mrow> only for real
   grouping or when the <mrow> carries intent/arg attributes.
9. Every $reference in an intent must have a matching arg="reference" on a
   descendant element.
{concept_rule}
{function_rule}
12. A generic function application such as f(x) must use
    intent="$name($var)" on the enclosing <mrow>, with arg="name" on the element
    that represents the complete function name and arg="var" on the element
    that represents the argument. Indexed function names written in LaTeX as
    f_{{i}} or f_{{i+1}} are still generic names: keep the complete indexed
    symbol as the name and use the same $name($var) pattern. Never invent a
    concept name for an indexed function. Examples:
    f_{{i}}(x): <mrow intent="$name($var)"><msub arg="name"><mi>f</mi><mi>i</mi></msub><mo>(</mo><mi arg="var">x</mi><mo>)</mo></mrow>
    f_{{i+1}}(x): <mrow intent="$name($var)"><msub arg="name"><mi>f</mi><mrow><mi>i</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>(</mo><mi arg="var">x</mi><mo>)</mo></mrow>
13. Put intent and arg on the smallest element that owns the meaning. Prefer
    <mfrac intent="fraction($num,$den)"> rather than an extra wrapper.
14. Add intent to an <mo> only when the context requires a meaning different
    from its default reading. Do not annotate common operators such as equals,
    plus or minus in their default meaning. Example: prefer <mo>=</mo> over
    <mo intent="equals">=</mo>. Use the intent fraction($num,$den) only when
    the source uses a linear form such as a/b instead of <mfrac>, so it reads
    as "a over b".
15. It is acceptable for a simple equation to have no intent attribute.
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

LaTeX: f_{{i}}(x)
<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow intent="$name($var)"><msub arg="name"><mi>f</mi><mi>i</mi></msub><mo>(</mo><mi arg="var">x</mi><mo>)</mo></mrow></math>

LaTeX: f_{{i+1}}(x)
<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow intent="$name($var)"><msub arg="name"><mi>f</mi><mrow><mi>i</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>(</mo><mi arg="var">x</mi><mo>)</mo></mrow></math>
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
<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac intent="fraction($num,$den)"><mi arg="num">a</mi><mi arg="den">b</mi></mfrac></mrow></math>

Base MathML:
<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo>(</mo><mi>x</mi><mo>)</mo></mrow></math>
Annotated MathML:
<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow intent="$name($var)"><mi arg="name">f</mi><mo>(</mo><mi arg="var">x</mi><mo>)</mo></mrow></math>

Base MathML:
<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>f</mi><mi>i</mi></msub><mo>(</mo><mi>x</mi><mo>)</mo></mrow></math>
Annotated MathML:
<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow intent="$name($var)"><msub arg="name"><mi>f</mi><mi>i</mi></msub><mo>(</mo><mi arg="var">x</mi><mo>)</mo></mrow></math>
"""
    )


def _semantic_concept_policy(
    allow_open_concepts: bool, allow_all_open_concepts: bool
) -> str:
    if not allow_open_concepts:
        return f"""Use only the following W3C intent core concept names. Do NOT
use open concepts. Core names are:
{_CORE_NAMES}"""
    if allow_all_open_concepts:
        return f"""Use only names from the W3C intent registries. Do not invent
names. Core names are:
{_CORE_NAMES}
Any documented W3C open concept is allowed when it is the correct meaning, but
prefer a core concept whenever one exists."""
    return f"""Use only names from the W3C intent registries. Do not invent
names. Core names are:
{_CORE_NAMES}
The only open concepts allowed by the current policy are gamma-function,
beta-function and error-function."""


def _semantic_system(
    allow_open_concepts: bool, allow_all_open_concepts: bool = False
) -> str:
    return f"""You are a W3C MathML 4 semantic annotator. You receive a clean
Presentation MathML tree with temporary id attributes and the LaTeX it came
from. Decide which concepts and argument positions carry a non-default
meaning. Do not rewrite MathML.

Return ONLY one JSON object. No Markdown fences, no comments, no explanation.

JSON contract:
{{
  "version": 1,
  "annotations": [
    {{
      "target": "n3",
      "concept": "fraction",
      "args": {{"a1": "n4", "a2": "n5"}}
    }}
  ]
}}

Optional keys per annotation: "wrap" instead of "target"; "new_id" only when
a later annotation must reference the new wrapper; "groups" for an argument
that spans several sibling ids; "literals" for literal intent arguments that
are not elements.

Rules:
1. Use only ids that exist in the supplied tree. Ids are temporary and must
   never appear in the final MathML.
2. Each annotation must have either "target" (an existing id) or "wrap" (a
   list of contiguous siblings to wrap in a new mrow). Never use both. A wrap
   does not need a new id unless another annotation references that wrapper.
   Never emit two annotations for the same target; choose one concept or omit
   both.
3. Argument slots are positional: "a1" is the first argument, "a2" the second,
   and so on. Do not use concept role names in the JSON.
4. For common concepts the positions mean:
   fraction: a1=num, a2=den
   power: a1=base, a2=exp
   root: a1=radicand, a2=index
   indexed-by: a1=base, a2=sub
   exponential: a1=exp
   sine, cosine, tangent, secant, cosecant, cotangent, arcsine, arccosine,
   arctangent, hyperbolic-sine, hyperbolic-cosine, hyperbolic-tangent,
   natural-logarithm, absolute-value, floor, ceiling, factorial: a1=arg
   logarithm: a1=arg, a2=base
   generic-application: a1=name, a2=var
5. Use "groups" when one argument spans several sibling elements. Grouped ids
   must be contiguous siblings under the same parent.
6. Use "literals" only for literal intent arguments that are not elements in
   the tree, such as the index 2 of a square root.
7. A generic function application such as f(x) or f_i(x+1) uses
   concept="generic-application": a1 is the complete function name and a2 is
   the complete argument.
8. Do not annotate the default meaning of common operators such as +, -, =,
   <, >, times or a standard fraction. Prefer no annotation over a misleading
   one.
9. Do not annotate a plain symbol that has no special mathematical role.
10. {_semantic_concept_policy(allow_open_concepts, allow_all_open_concepts)}
11. Keep every element, its text and its order unchanged. The code will add
    the intent and arg attributes and any wrapper mrows.
12. A node may have its own intent and also be an argument of an outer intent;
    annotate the inner and outer meanings in separate annotations.
13. If you are not sure, omit that annotation.

Examples:
Fraction:
{{"version": 1, "annotations": [{{"target": "n3", "concept": "fraction", "args": {{"a1": "n4", "a2": "n5"}}}}]}}

Generic function f_i(x+1):
{{"version": 1, "annotations": [{{"target": "n2", "concept": "generic-application", "args": {{"a1": "n3"}}, "groups": {{"a2": ["n6", "n7", "n8"]}}}}]}}

Generic function inside a larger expression (use wrap when no container exists):
{{"version": 1, "annotations": [{{"wrap": ["n2", "n3", "n4", "n5"], "concept": "generic-application", "args": {{"a1": "n2", "a2": "n4"}}}}]}}

Square root:
{{"version": 1, "annotations": [{{"target": "n2", "concept": "root", "args": {{"a1": "n3"}}, "literals": {{"a2": "2"}}}}]}}
"""


def _semantic_feedback_block(feedback: list[str] | None) -> str:
    if not feedback:
        return ""
    lines = "\n".join(f"- {item}" for item in feedback[:12])
    return (
        "\nYour previous JSON could not be applied. Problems:\n"
        f"{lines}\n"
        "Return a complete corrected JSON object. Do not explain.\n"
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


def _vision_system(
    allow_open_concepts: bool, allow_all_open_concepts: bool = False
) -> str:
    return (
        "You are a meticulous MathML 4 accessibility engineer. You receive an "
        "image of a mathematical equation. Read the image and return directly "
        "its Presentation MathML annotated with W3C intent attributes. Do not "
        "transcribe the image to LaTeX first and do not include any LaTeX in "
        "your answer.\n"
        + _structure_rules(allow_open_concepts, allow_all_open_concepts)
        + """
Examples of valid final outputs:
<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac intent="fraction($num,$den)"><mi arg="num">a</mi><mi arg="den">b</mi></mfrac></mrow></math>

<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow intent="$name($var)"><mi arg="name">f</mi><mo>(</mo><mi arg="var">x</mi><mo>)</mo></mrow></math>

<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow intent="$name($var)"><msub arg="name"><mi>f</mi><mi>i</mi></msub><mo>(</mo><mi arg="var">x</mi><mo>)</mo></mrow></math>
"""
    )


def vision_messages(
    context: MathContext,
    feedback: list[str] | None = None,
    *,
    allow_open_concepts: bool = True,
    allow_all_open_concepts: bool = False,
) -> list[dict[str, str]]:
    user = (
        "Read the mathematical equation in the attached image and return its "
        "complete Presentation MathML with W3C intent attributes.\n\n"
        f"{_context_block(context)}\n"
        f"{_feedback_block(feedback)}"
    )
    return [
        {
            "role": "system",
            "content": _vision_system(allow_open_concepts, allow_all_open_concepts),
        },
        {"role": "user", "content": user},
    ]


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


def semantic_messages(
    latex: str,
    semantic_tree: str,
    context: MathContext,
    feedback: list[str] | None = None,
    *,
    allow_open_concepts: bool = True,
    allow_all_open_concepts: bool = False,
) -> list[dict[str, str]]:
    user = (
        "Choose the W3C intent semantics for this expression and return the "
        "JSON contract described in the system message.\n\n"
        f"LaTeX:\n{latex}\n\n"
        f"Clean MathML tree with temporary ids:\n{semantic_tree}\n\n"
        f"{_context_block(context)}\n"
        f"{_semantic_feedback_block(feedback)}"
    )
    return [
        {
            "role": "system",
            "content": _semantic_system(allow_open_concepts, allow_all_open_concepts),
        },
        {"role": "user", "content": user},
    ]
