"""Normalised MathML comparison and BLEU metrics.

The evaluator compares a reference MathML document with the MathML produced by
the pipeline.  Both documents are first reduced to a canonical token stream so
that:

* visual-only helpers such as ``mspace``, ``mphantom`` and ``mglyph`` do not
  affect the score;
* transparent wrappers such as ``mstyle`` and ``mpadded`` are unwrapped;
* presentation attributes such as ``stretchy`` or ``mathvariant`` are ignored;
* character references (``&#x2212;``, ``&#8722;``) are decoded and visually
  equivalent symbol variants (keyboard hyphen vs. mathematical minus, math
  italic letters vs. plain Greek, ``\\cdot`` dots, ...) are unified, so the
  reference may use either form without being penalised;
* intent argument names are positional (``$arg1``, ``$arg2``, ...), so
  ``power($base,$exp)`` and ``power($b,$e)`` are equivalent when they reference
  the same argument nodes.

Two scores are returned for each comparison:

``structure``
    Canonical tokens with ``intent`` and ``arg`` removed.  It measures how
    similar the recognised structure and symbol sequence are.

``intent``
    Canonical tokens including normalised ``intent`` and ``arg`` attributes.  It
    measures structure and semantic annotation together.
"""

from __future__ import annotations

import html.entities
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from lxml import etree

from ..mathml.concepts import ASSOCIATIVE_CONCEPTS
from ..mathml.dom import as_text, drop_attribute, is_element, localname
from ..mathml.intent_parser import (
    IntentNode,
    IntentSyntaxError,
    format_intent,
    parse_intent,
)
from ..mathml.validator import _reference_candidates

_INTENT_VAR_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_.-]*)")
_XML_WHITESPACE_RE = re.compile(r"\s+")

_DROP_ELEMENTS = frozenset(
    {
        "annotation",
        "annotation-xml",
        "mspace",
        "mphantom",
        "mglyph",
        "maligngroup",
        "malignmark",
    }
)
_UNWRAP_ELEMENTS = frozenset({"merror", "mpadded", "mstyle", "semantics"})

_IGNORED_ATTRIBUTES = frozenset(
    {
        "accent",
        "accentunder",
        "align",
        "background",
        "bevelled",
        "charalign",
        "charspacing",
        "class",
        "color",
        "columnalign",
        "columnlines",
        "columnspacing",
        "columnwidth",
        "depth",
        "denomalign",
        "dir",
        "display",
        "displaystyle",
        "edge",
        "equalcolumns",
        "equalrows",
        "fence",
        "form",
        "frame",
        "framespacing",
        "groupalign",
        "height",
        "href",
        "id",
        "idref",
        "indentalign",
        "indentalignfirst",
        "indentalignlast",
        "indentshift",
        "indentshiftfirst",
        "indentshiftlast",
        "infixlinebreakstyle",
        "largeop",
        "length",
        "linebreak",
        "linebreakmultchar",
        "linebreakstyle",
        "lineleading",
        "linethickness",
        "lang",
        "lspace",
        "mathbackground",
        "mathcolor",
        "mathsize",
        "mathvariant",
        "maxsize",
        "minlabelspacing",
        "minsize",
        "movablelimits",
        "notation",
        "numalign",
        "position",
        "rspace",
        "rowalign",
        "rowlines",
        "rowspacing",
        "scriptlevel",
        "scriptminsize",
        "scriptsizemultiplier",
        "separator",
        "shift",
        "side",
        "stackalign",
        "stretchy",
        "style",
        "symmetric",
        "voffset",
        "width",
        "xref",
    }
)
_IGNORED_ATTRIBUTE_PREFIXES = ("aria-", "data-", "xml:")


_NAMED_ENTITY_RE = re.compile(r"&([A-Za-z][A-Za-z0-9]+);")
_CHARACTER_REFERENCE_RE = re.compile(
    r"&(?:#([0-9]+)|#[xX]([0-9A-Fa-f]+)|([A-Za-z][A-Za-z0-9]+));"
)
_KEPT_ENTITIES = frozenset({"amp", "lt", "gt", "quot", "apos"})

# Visually equivalent symbols that must not change the score.  The mapping is
# applied after Unicode NFKC, which already unifies compatibility variants such
# as the mathematical italic letters used by some references (``&#x1D706;``)
# with their plain Greek counterparts (``&#x03BB;``).
_SYMBOL_EQUIVALENTS = str.maketrans(
    {
        # Hyphens and dashes used as the minus operator.
        "\u002d": "\u2212",  # hyphen-minus (keyboard)
        "\u2010": "\u2212",  # hyphen
        "\u2011": "\u2212",  # non-breaking hyphen
        "\u2012": "\u2212",  # figure dash
        "\u2013": "\u2212",  # en dash
        "\u2014": "\u2212",  # em dash
        "\ufe63": "\u2212",  # small hyphen-minus
        "\uff0d": "\u2212",  # fullwidth hyphen-minus
        # Multiplication / inner product dots.
        "\u00b7": "\u22c5",  # middle dot
        "\u2219": "\u22c5",  # bullet operator
        # Comparison signs.
        "\u2a7d": "\u2264",  # less-than or slanted equal to
        "\u2a7e": "\u2265",  # greater-than or slanted equal to
        # Division slash.
        "\u2215": "\u002f",  # division slash
        # Primes: NFKC splits double/triple primes into U+2032 sequences.
        "\u2032": "'",
        # Double vertical bars (norm).
        "\u2225": "\u2016",  # parallel to
        # Ellipses: NFKC already turns the baseline ``\u2026`` into ``...``.
        "\u22ef": "...",  # midline horizontal ellipsis
        # Arrows and angle brackets (NFKC maps U+2329/U+232A to U+3008/U+3009).
        "\u27f6": "\u2192",  # long rightward arrow
        "\u3008": "\u27e8",  # left angle bracket
        "\u3009": "\u27e9",  # right angle bracket
    }
)


class MathMLNormalizationError(ValueError):
    """Raised when a MathML document cannot be parsed for evaluation."""


@dataclass(frozen=True)
class BleuScore:
    """One BLEU score with its component statistics."""

    score: float
    precisions: tuple[float, ...]
    brevity_penalty: float
    candidate_length: int
    reference_length: int
    max_order: int = 4

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 6),
            "precisions": [round(value, 6) for value in self.precisions],
            "brevity_penalty": round(self.brevity_penalty, 6),
            "candidate_length": self.candidate_length,
            "reference_length": self.reference_length,
            "max_order": self.max_order,
        }


@dataclass(frozen=True)
class MathMLBleuScore:
    """Structure-only and structure-plus-intent BLEU scores."""

    structure: BleuScore
    intent: BleuScore

    def as_dict(self) -> dict[str, Any]:
        return {
            "structure_bleu": self.structure.as_dict(),
            "intent_bleu": self.intent.as_dict(),
        }


def _entity_codepoint(name: str) -> int | None:
    codepoint = html.entities.name2codepoint.get(name)
    if codepoint is not None:
        return codepoint
    value = html.entities.html5.get(f"{name};")
    return ord(value) if value is not None and len(value) == 1 else None


def _sanitize_entities(markup: str) -> str:
    """Turn named HTML/MathML entities into numeric references.

    XML only defines the five predefined entities; names such as ``&minus;``
    would make the parser fail.  Converting them to ``&#...;`` keeps the
    reference usable without depending on a DTD.
    """

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in _KEPT_ENTITIES:
            return match.group(0)
        codepoint = _entity_codepoint(name)
        return match.group(0) if codepoint is None else f"&#{codepoint};"

    return _NAMED_ENTITY_RE.sub(replace, markup)


def _parse_mathml(markup: str) -> etree._Element:
    if not isinstance(markup, str) or not markup.strip():
        raise MathMLNormalizationError("The MathML document is empty")
    markup = _sanitize_entities(markup)
    parser = etree.XMLParser(
        resolve_entities=False,
        recover=False,
        remove_comments=True,
        remove_pis=True,
        huge_tree=False,
    )
    try:
        return etree.fromstring(markup.encode("utf-8"), parser=parser)
    except etree.XMLSyntaxError as exc:
        raise MathMLNormalizationError(f"Invalid MathML/XML: {exc}") from exc


def _normalise_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(_XML_WHITESPACE_RE.sub(" ", text).split())


def _decode_character_references(text: str) -> str:
    """Decode references left as literal text (e.g. double-escaped markup)."""

    def replace(match: re.Match[str]) -> str:
        decimal, hexadecimal, name = match.groups()
        if name is not None:
            if name in _KEPT_ENTITIES:
                return match.group(0)
            codepoint = _entity_codepoint(name)
            if codepoint is None:
                return match.group(0)
        else:
            try:
                codepoint = (
                    int(decimal, 10) if decimal is not None else int(hexadecimal, 16)
                )
            except (TypeError, ValueError):
                return match.group(0)
        if not 0 <= codepoint <= 0x10FFFF:
            return match.group(0)
        return chr(codepoint)

    return _CHARACTER_REFERENCE_RE.sub(replace, text)


def _normalise_symbols(text: str | None) -> str:
    """Canonicalise symbol spellings so equivalent forms compare equal.

    ``&#x2212;`` and ``−`` are already the same character after XML parsing,
    but references may keep the reference literally (double escaping) and the
    keyboard hyphen (``-``) is not the mathematical minus used by
    ``latex2mathml``.  NFKC also folds the mathematical alphanumeric symbols
    used by some references into their plain letter equivalents.
    """

    if not text:
        return ""
    decoded = _decode_character_references(text)
    return unicodedata.normalize("NFKC", decoded).translate(_SYMBOL_EQUIVALENTS)


def _new_element_like(element: etree._Element, name: str) -> etree._Element:
    tag = element.tag
    if isinstance(tag, str) and tag.startswith("{"):
        namespace = tag[1:].split("}", 1)[0]
        return etree.Element(f"{{{namespace}}}{name}")
    return etree.Element(name)


def _clean_children(element: etree._Element) -> None:
    index = 0
    while index < len(element):
        child = element[index]
        if not is_element(child):
            index += 1
            continue
        tag = localname(child.tag)
        if tag in _DROP_ELEMENTS:
            element.remove(child)
            continue
        _clean_children(child)
        if tag in _UNWRAP_ELEMENTS:
            for grandchild in list(child):
                if not is_element(grandchild):
                    child.remove(grandchild)
                    continue
                child.remove(grandchild)
                element.insert(index, grandchild)
                index += 1
            element.remove(child)
            continue
        index += 1


def _wrap_math_root(element: etree._Element) -> None:
    if localname(element.tag) != "math":
        return
    children = [child for child in element if is_element(child)]
    if not children:
        return
    if len(children) == 1 and localname(children[0].tag) == "mrow":
        return

    wrapper = _new_element_like(element, "mrow")
    for child in children:
        element.remove(child)
        wrapper.append(child)
    element.insert(0, wrapper)


_PRIME_CHARS = frozenset("\u2032\u2033\u2034\u2057'")


def _is_prime_token(element: etree._Element) -> bool:
    if localname(element.tag) not in {"mo", "mi"} or len(element):
        return False
    text = (element.text or "").strip()
    return bool(text) and all(char in _PRIME_CHARS for char in text)


def _canonicalise_primes(root: etree._Element) -> None:
    """Write every prime as ``<msup>base <mo>prime</mo></msup>``.

    Labels often put the prime next to its base (``<mi>y</mi><mo>'</mo>``)
    while LaTeX converters attach it as a superscript; both render and read
    the same.  A row that only holds the base and its prime becomes the
    ``msup`` itself, so an intent on that row stays on the same element.
    """

    for prime in [node for node in root.iter() if is_element(node)]:
        if not _is_prime_token(prime):
            continue
        prime.tag = _new_element_like(prime, "mo").tag
        parent = prime.getparent()
        if parent is None:
            continue
        siblings = [child for child in parent if is_element(child)]
        position = siblings.index(prime)
        if localname(parent.tag) == "msup" and position == 1:
            continue  # already a superscript
        if position == 0 or localname(parent.tag) not in {"mrow", "math", "mtd"}:
            continue
        base = siblings[position - 1]
        if localname(base.tag) == "mo":
            continue
        if localname(parent.tag) == "mrow" and len(siblings) == 2:
            parent.tag = _new_element_like(parent, "msup").tag
            continue
        script = _new_element_like(parent, "msup")
        parent.insert(parent.index(base), script)
        script.tail = prime.tail
        prime.tail = None
        script.append(base)
        script.append(prime)


def _unwrap_redundant_mrows(root: etree._Element) -> None:
    """Replace attribute-less single-child ``<mrow>`` wrappers by their child.

    Such wrappers do not change the rendering or the reading, and the
    pipeline's normaliser removes them; without this step, a reference that
    keeps them (for example raw ``latex2mathml`` output) is penalised for a
    purely cosmetic difference.  The top-level ``<mrow>`` is kept.
    """

    top = [child for child in root if is_element(child)]
    keep = top[0] if len(top) == 1 and localname(top[0].tag) == "mrow" else None
    for element in list(root.iter()):
        if not is_element(element) or element is keep:
            continue
        if localname(element.tag) != "mrow" or element.attrib:
            continue
        children = [child for child in element if is_element(child)]
        parent = element.getparent()
        if parent is None or len(children) != 1 or (element.text or "").strip():
            continue
        child = children[0]
        if (child.tail or "").strip():
            continue
        child.tail = element.tail
        parent.replace(element, child)


def _is_associative_application(node: IntentNode, head: str | None = None) -> bool:
    if node.kind != "application" or node.properties:
        return False
    name = node.children[0]
    if name.kind != "name" or name.properties:
        return False
    if name.value not in ASSOCIATIVE_CONCEPTS:
        return False
    return head is None or name.value == head


def _flatten_node(node: IntentNode) -> IntentNode:
    """times(a, times(b, c)) -> times(a, b, c) inside one intent value."""

    if node.kind != "application":
        return node
    head, *arguments = node.children
    arguments = [_flatten_node(argument) for argument in arguments]
    if _is_associative_application(node):
        flat: list[IntentNode] = []
        for argument in arguments:
            if _is_associative_application(argument, head.value):
                flat.extend(argument.children[1:])
            else:
                flat.append(argument)
        arguments = flat
    return IntentNode(
        kind="application", children=[head, *arguments], properties=node.properties
    )


def _flatten_associative(root: etree._Element) -> None:
    """Make nested and flat associative products compare as equal.

    ``times($a,$b)`` whose ``$b`` element carries ``times($c,$d)`` becomes
    ``times($a,$c,$d)`` (the inner element loses its intent and arg), and
    nesting inside one intent value is flattened as well.  Grouping does not
    change the meaning of an associative operator, so different labelling
    habits should not change the score.
    """

    counter = 0
    changed = True
    while changed:
        changed = False
        for element in list(root.iter()):
            if not is_element(element):
                continue
            intent = element.get("intent")
            if intent is None:
                continue
            try:
                ast = _flatten_node(parse_intent(intent))
            except IntentSyntaxError:
                continue
            if not _is_associative_application(ast):
                element.set("intent", format_intent(ast))
                continue
            head = ast.children[0].value
            new_arguments: list[IntentNode] = []
            for argument in ast.children[1:]:
                candidates = (
                    _reference_candidates(element, argument.value)
                    if argument.kind == "reference"
                    else []
                )
                child = candidates[0] if len(candidates) == 1 else None
                child_ast = None
                child_intent = child.get("intent") if child is not None else None
                if child_intent:
                    try:
                        child_ast = _flatten_node(parse_intent(child_intent))
                    except IntentSyntaxError:
                        child_ast = None
                if (
                    child is None
                    or child_ast is None
                    or not _is_associative_application(child_ast, head)
                ):
                    new_arguments.append(argument)
                    continue
                # Splice the inner product's arguments into this one.
                for inner in child_ast.children[1:]:
                    if inner.kind != "reference":
                        new_arguments.append(inner)
                        continue
                    targets = _reference_candidates(child, inner.value)
                    counter += 1
                    fresh = f"flat{counter}"
                    for target in targets[:1]:
                        target.set("arg", fresh)
                    new_arguments.append(IntentNode(kind="reference", value=fresh))
                del child.attrib["intent"]
                drop_attribute(child, "arg")
                parent = child.getparent()
                if (
                    localname(child.tag) == "mrow"
                    and not child.attrib
                    and parent is not None
                ):
                    # The row only grouped the inner product; grouping is
                    # meaningless for an associative operator.
                    position = parent.index(child)
                    for offset, node in enumerate(list(child)):
                        parent.insert(position + offset, node)
                    parent.remove(child)
                changed = True
            flattened = IntentNode(
                kind="application",
                children=[ast.children[0], *new_arguments],
                properties=ast.properties,
            )
            element.set("intent", format_intent(flattened))


def _normalise_attribute(value: str) -> str:
    return _normalise_text(value)


def _canonicalise_intents(root: etree._Element) -> None:
    covered_args: set[etree._Element] = set()
    for element in root.iter():
        if not is_element(element):
            continue
        intent = element.get("intent")
        if intent is None:
            continue

        mapping: dict[str, str] = {}
        pieces: list[str] = []
        last = 0
        for match in _INTENT_VAR_RE.finditer(intent):
            name = match.group(1)
            if name not in mapping:
                mapping[name] = f"arg{len(mapping) + 1}"
            pieces.append(intent[last : match.start()])
            pieces.append(f"${mapping[name]}")
            last = match.end()
        pieces.append(intent[last:])
        normalised = _XML_WHITESPACE_RE.sub("", "".join(pieces))
        element.set("intent", normalised)

        if not mapping:
            continue
        for descendant in element.iterdescendants():
            if not is_element(descendant):
                continue
            arg = descendant.get("arg")
            if arg is None or arg not in mapping:
                continue
            descendant.set("arg", mapping[arg])
            covered_args.add(descendant)

    for element in root.iter():
        if not is_element(element):
            continue
        if element.get("arg") is not None and element not in covered_args:
            drop_attribute(element, "arg")


def normalize_mathml_tree(
    markup: str,
    *,
    include_intent: bool = True,
) -> etree._Element:
    """Parse and normalise one MathML document for evaluation."""

    root = _parse_mathml(markup)
    _clean_children(root)
    _canonicalise_primes(root)
    _wrap_math_root(root)
    _unwrap_redundant_mrows(root)
    _flatten_associative(root)
    _canonicalise_intents(root)
    if not include_intent:
        for element in root.iter():
            if not is_element(element):
                continue
            drop_attribute(element, "intent")
            drop_attribute(element, "arg")
    return root


def _ignored_attribute(name: str) -> bool:
    if name in _IGNORED_ATTRIBUTES:
        return True
    return name.startswith(_IGNORED_ATTRIBUTE_PREFIXES)


def _attribute_tokens(element: etree._Element, include_intent: bool) -> list[str]:
    attributes = []
    for raw_name, value in sorted(element.attrib.items()):
        name = localname(raw_name)
        if not include_intent and name in {"intent", "arg"}:
            continue
        if _ignored_attribute(name):
            continue
        attributes.append(f'{name}="{_normalise_attribute(as_text(value))}"')
    return attributes


def _text_tokens(text: str | None) -> list[str]:
    normalised = _normalise_text(_normalise_symbols(text))
    return normalised.split() if normalised else []


def _element_tokens(
    element: etree._Element,
    include_intent: bool,
) -> list[str]:
    tag = localname(element.tag)
    if tag in _DROP_ELEMENTS:
        return []
    if tag in _UNWRAP_ELEMENTS:
        tokens = _text_tokens(element.text)
        for child in element:
            if not is_element(child):
                continue
            tokens.extend(_element_tokens(child, include_intent))
            tokens.extend(_text_tokens(child.tail))
        return tokens

    attributes = _attribute_tokens(element, include_intent)
    start = f"<{tag}" + (" " + " ".join(attributes) if attributes else "") + ">"
    tokens = [start]
    tokens.extend(_text_tokens(element.text))
    for child in element:
        if not is_element(child):
            continue
        tokens.extend(_element_tokens(child, include_intent))
        tokens.extend(_text_tokens(child.tail))
    tokens.append(f"</{tag}>")
    return tokens


def canonical_mathml_tokens(
    markup: str,
    *,
    include_intent: bool = True,
) -> list[str]:
    """Return the canonical BLEU token stream for one MathML document."""

    root = normalize_mathml_tree(markup, include_intent=include_intent)
    return _element_tokens(root, include_intent)


def _ngrams(tokens: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(
        tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1)
    )


def _brevity_penalty(candidate_length: int, reference_length: int) -> float:
    if candidate_length == 0:
        return 0.0
    if candidate_length > reference_length:
        return 1.0
    return math.exp(1.0 - reference_length / candidate_length)


def sentence_bleu(
    reference_tokens: Sequence[str],
    candidate_tokens: Sequence[str],
    *,
    max_n: int = 4,
    smooth: bool = True,
) -> BleuScore:
    """Compute sentence-level BLEU for two token sequences."""

    reference = list(reference_tokens)
    candidate = list(candidate_tokens)
    candidate_length = len(candidate)
    reference_length = len(reference)
    if candidate_length == 0:
        return BleuScore(
            score=0.0,
            precisions=(),
            brevity_penalty=0.0,
            candidate_length=0,
            reference_length=reference_length,
            max_order=max(0, max_n),
        )

    max_order = max(1, min(max_n, candidate_length))
    precisions: list[float] = []
    for order in range(1, max_order + 1):
        candidate_counts = _ngrams(candidate, order)
        reference_counts = _ngrams(reference, order)
        matches = sum(
            min(count, reference_counts[ngram])
            for ngram, count in candidate_counts.items()
        )
        total = candidate_length - order + 1
        if total <= 0:
            precisions.append(0.0)
        elif matches == 0 and smooth and order > 1:
            precisions.append(1.0 / (total + 1.0))
        else:
            precisions.append(matches / total)

    brevity = _brevity_penalty(candidate_length, reference_length)
    if any(precision <= 0.0 for precision in precisions):
        score = 0.0
    else:
        log_sum = sum(math.log(precision) / max_order for precision in precisions)
        score = brevity * math.exp(log_sum)

    return BleuScore(
        score=score,
        precisions=tuple(precisions),
        brevity_penalty=brevity,
        candidate_length=candidate_length,
        reference_length=reference_length,
        max_order=max_order,
    )


def corpus_bleu(
    reference_token_lists: Sequence[Sequence[str]],
    candidate_token_lists: Sequence[Sequence[str]],
    *,
    max_n: int = 4,
    smooth: bool = True,
) -> BleuScore:
    """Compute corpus-level BLEU by aggregating clipped n-gram counts."""

    if len(reference_token_lists) != len(candidate_token_lists):
        raise ValueError("The reference and candidate lists must have equal length")

    references = [list(tokens) for tokens in reference_token_lists]
    candidates = [list(tokens) for tokens in candidate_token_lists]
    candidate_length = sum(len(tokens) for tokens in candidates)
    reference_length = sum(len(tokens) for tokens in references)
    if candidate_length == 0:
        return BleuScore(
            score=0.0,
            precisions=(),
            brevity_penalty=0.0,
            candidate_length=0,
            reference_length=reference_length,
            max_order=max(0, max_n),
        )

    orders = [
        order
        for order in range(1, max(1, max_n) + 1)
        if any(len(tokens) >= order for tokens in candidates)
    ]
    if not orders:
        orders = [1]

    precisions: list[float] = []
    for order in orders:
        matches = 0
        total = 0
        for reference, candidate in zip(references, candidates, strict=False):
            candidate_counts = _ngrams(candidate, order)
            reference_counts = _ngrams(reference, order)
            matches += sum(
                min(count, reference_counts[ngram])
                for ngram, count in candidate_counts.items()
            )
            total += max(0, len(candidate) - order + 1)
        if total <= 0:
            precisions.append(0.0)
        elif matches == 0 and smooth and order > 1:
            precisions.append(1.0 / (total + 1.0))
        else:
            precisions.append(matches / total)

    brevity = _brevity_penalty(candidate_length, reference_length)
    if any(precision <= 0.0 for precision in precisions):
        score = 0.0
    else:
        log_sum = sum(math.log(precision) / len(orders) for precision in precisions)
        score = brevity * math.exp(log_sum)

    return BleuScore(
        score=score,
        precisions=tuple(precisions),
        brevity_penalty=brevity,
        candidate_length=candidate_length,
        reference_length=reference_length,
        max_order=len(orders),
    )


def structure_bleu(
    reference_mathml: str,
    candidate_mathml: str,
    *,
    max_n: int = 4,
    smooth: bool = True,
) -> BleuScore:
    """BLEU over canonical structure tokens, without intents."""

    return sentence_bleu(
        canonical_mathml_tokens(reference_mathml, include_intent=False),
        canonical_mathml_tokens(candidate_mathml, include_intent=False),
        max_n=max_n,
        smooth=smooth,
    )


def intent_bleu(
    reference_mathml: str,
    candidate_mathml: str,
    *,
    max_n: int = 4,
    smooth: bool = True,
) -> BleuScore:
    """BLEU over canonical structure tokens including normalised intents."""

    return sentence_bleu(
        canonical_mathml_tokens(reference_mathml, include_intent=True),
        canonical_mathml_tokens(candidate_mathml, include_intent=True),
        max_n=max_n,
        smooth=smooth,
    )


def mathml_bleu(
    reference_mathml: str,
    candidate_mathml: str,
    *,
    max_n: int = 4,
    smooth: bool = True,
) -> MathMLBleuScore:
    """Compare one generated MathML document with its reference."""

    return MathMLBleuScore(
        structure=structure_bleu(
            reference_mathml,
            candidate_mathml,
            max_n=max_n,
            smooth=smooth,
        ),
        intent=intent_bleu(
            reference_mathml,
            candidate_mathml,
            max_n=max_n,
            smooth=smooth,
        ),
    )


def _candidate_tokens(document: str | None, *, include_intent: bool) -> list[str]:
    """Tokens of a candidate; a missing or empty candidate scores as empty.

    Samples whose pipeline run failed are passed as ``""`` so they count as a
    zero-length output instead of disappearing from the corpus.
    """

    if document is None or not document.strip():
        return []
    try:
        return canonical_mathml_tokens(document, include_intent=include_intent)
    except MathMLNormalizationError:
        return []


def mathml_corpus_bleu(
    reference_mathml_documents: Sequence[str],
    candidate_mathml_documents: Sequence[str | None],
    *,
    max_n: int = 4,
    smooth: bool = True,
) -> MathMLBleuScore:
    """Compare several MathML documents at corpus level.

    An empty (or unparsable) candidate counts as an empty output, so failed
    samples lower the corpus score instead of being silently excluded.
    """

    if len(reference_mathml_documents) != len(candidate_mathml_documents):
        raise ValueError("The reference and candidate lists must have equal length")

    reference_structure = [
        canonical_mathml_tokens(document, include_intent=False)
        for document in reference_mathml_documents
    ]
    candidate_structure = [
        _candidate_tokens(document, include_intent=False)
        for document in candidate_mathml_documents
    ]
    reference_intent = [
        canonical_mathml_tokens(document, include_intent=True)
        for document in reference_mathml_documents
    ]
    candidate_intent = [
        _candidate_tokens(document, include_intent=True)
        for document in candidate_mathml_documents
    ]
    return MathMLBleuScore(
        structure=corpus_bleu(
            reference_structure,
            candidate_structure,
            max_n=max_n,
            smooth=smooth,
        ),
        intent=corpus_bleu(
            reference_intent,
            candidate_intent,
            max_n=max_n,
            smooth=smooth,
        ),
    )


# ---------------------------------------------------------------------------
# complementary metrics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IntentF1:
    """Precision/recall of the intents, compared per annotated element.

    Each intent is identified by the tag that carries it and its canonical
    value (argument names already made positional), so ``power($b,$e)`` on an
    ``msup`` matches ``power($base,$exp)`` on an ``msup``.
    """

    precision: float
    recall: float
    f1: float
    reference_count: int
    candidate_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "reference_count": self.reference_count,
            "candidate_count": self.candidate_count,
        }


def canonical_intents(markup: str) -> Counter[str]:
    root = normalize_mathml_tree(markup, include_intent=True)
    return Counter(
        f"{localname(element.tag)}:{element.get('intent')}"
        for element in root.iter()
        if is_element(element) and element.get("intent") is not None
    )


def intent_f1(reference_mathml: str, candidate_mathml: str | None) -> IntentF1:
    reference = canonical_intents(reference_mathml)
    candidate = (
        Counter()
        if candidate_mathml is None or not candidate_mathml.strip()
        else canonical_intents(candidate_mathml)
    )
    reference_count = sum(reference.values())
    candidate_count = sum(candidate.values())
    if reference_count == 0 and candidate_count == 0:
        return IntentF1(1.0, 1.0, 1.0, 0, 0)
    matches = sum((reference & candidate).values())
    precision = matches / candidate_count if candidate_count else 0.0
    recall = matches / reference_count if reference_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return IntentF1(precision, recall, f1, reference_count, candidate_count)


def exact_match(
    reference_mathml: str,
    candidate_mathml: str | None,
    *,
    include_intent: bool,
) -> bool:
    """True when both documents have the same canonical token stream."""

    if candidate_mathml is None or not candidate_mathml.strip():
        return False
    return canonical_mathml_tokens(
        reference_mathml, include_intent=include_intent
    ) == canonical_mathml_tokens(candidate_mathml, include_intent=include_intent)


# Elements whose children are positional arguments: an ``<mrow>`` directly
# inside them delimits an argument, so it is never treated as mere grouping.
_FIXED_ARITY_ELEMENTS = frozenset(
    {
        "mfrac",
        "mmultiscripts",
        "mover",
        "mroot",
        "msub",
        "msubsup",
        "msup",
        "munder",
        "munderover",
    }
)
_TRAILING_PUNCTUATION = frozenset({".", ",", ";"})


def _unwrap_grouping_mrows(element: etree._Element, include_intent: bool) -> None:
    """Splice every bare ``<mrow>`` into its parent, bottom-up.

    A bare row is one without scored attributes (with ``include_intent`` the
    rows carrying ``intent``/``arg`` are kept).  Rows that are arguments of a
    fixed-arity element are kept too, so ``\\frac{ab}{c}`` and
    ``\\frac{a}{bc}`` stay different.
    """

    for child in [node for node in element if is_element(node)]:
        _unwrap_grouping_mrows(child, include_intent)
    if localname(element.tag) in _FIXED_ARITY_ELEMENTS:
        return
    index = 0
    while index < len(element):
        child = element[index]
        if (
            not is_element(child)
            or localname(child.tag) != "mrow"
            or _attribute_tokens(child, include_intent)
            or (child.text or "").strip()
        ):
            index += 1
            continue
        grandchildren = list(child)
        if grandchildren:
            last = grandchildren[-1]
            last.tail = (last.tail or "") + (child.tail or "")
        element.remove(child)
        for offset, grandchild in enumerate(grandchildren):
            element.insert(index + offset, grandchild)
        index += len(grandchildren)


def _drop_trailing_punctuation(root: etree._Element, include_intent: bool) -> None:
    while True:
        children = [node for node in root if is_element(node)]
        if not children:
            return
        last = children[-1]
        if (
            localname(last.tag) != "mo"
            or len(last)
            or _attribute_tokens(last, include_intent)
            or (last.text or "").strip() not in _TRAILING_PUNCTUATION
        ):
            return
        root.remove(last)


def lenient_mathml_tokens(markup: str, *, include_intent: bool = True) -> list[str]:
    """Canonical tokens without grouping rows or sentence punctuation.

    On top of :func:`canonical_mathml_tokens` it ignores how the expression is
    grouped with bare ``<mrow>`` elements and drops the ``.``, ``,`` or ``;``
    that ends the expression (the sentence punctuation copied from the page).
    """

    root = normalize_mathml_tree(markup, include_intent=include_intent)
    _unwrap_grouping_mrows(root, include_intent)
    _drop_trailing_punctuation(root, include_intent)
    return _element_tokens(root, include_intent)


def lenient_exact_match(
    reference_mathml: str,
    candidate_mathml: str | None,
    *,
    include_intent: bool,
) -> bool:
    """Exact match that ignores ``<mrow>`` grouping and trailing punctuation."""

    if candidate_mathml is None or not candidate_mathml.strip():
        return False
    return lenient_mathml_tokens(
        reference_mathml, include_intent=include_intent
    ) == lenient_mathml_tokens(candidate_mathml, include_intent=include_intent)
