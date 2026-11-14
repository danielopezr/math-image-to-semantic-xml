"""Normalised MathML comparison and BLEU metrics.

The evaluator compares a reference MathML document with the MathML produced by
the pipeline.  Both documents are first reduced to a canonical token stream so
that:

* visual-only helpers such as ``mspace``, ``mphantom`` and ``mglyph`` do not
  affect the score;
* transparent wrappers such as ``mstyle`` and ``mpadded`` are unwrapped;
* presentation attributes such as ``stretchy`` or ``mathvariant`` are ignored;
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

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from lxml import etree

from ..mathml.dom import is_element, localname

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


def _parse_mathml(markup: str) -> etree._Element:
    if not isinstance(markup, str) or not markup.strip():
        raise MathMLNormalizationError("The MathML document is empty")
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
            element.attrib.pop("arg", None)


def normalize_mathml_tree(
    markup: str,
    *,
    include_intent: bool = True,
) -> etree._Element:
    """Parse and normalise one MathML document for evaluation."""

    root = _parse_mathml(markup)
    _clean_children(root)
    _wrap_math_root(root)
    _canonicalise_intents(root)
    if not include_intent:
        for element in root.iter():
            if not is_element(element):
                continue
            element.attrib.pop("intent", None)
            element.attrib.pop("arg", None)
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
        attributes.append(f'{name}="{_normalise_attribute(value)}"')
    return attributes


def _text_tokens(text: str | None) -> list[str]:
    normalised = _normalise_text(text)
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


def mathml_corpus_bleu(
    reference_mathml_documents: Sequence[str],
    candidate_mathml_documents: Sequence[str],
    *,
    max_n: int = 4,
    smooth: bool = True,
) -> MathMLBleuScore:
    """Compare several MathML documents at corpus level."""

    if len(reference_mathml_documents) != len(candidate_mathml_documents):
        raise ValueError("The reference and candidate lists must have equal length")

    reference_structure = [
        canonical_mathml_tokens(document, include_intent=False)
        for document in reference_mathml_documents
    ]
    candidate_structure = [
        canonical_mathml_tokens(document, include_intent=False)
        for document in candidate_mathml_documents
    ]
    reference_intent = [
        canonical_mathml_tokens(document, include_intent=True)
        for document in reference_mathml_documents
    ]
    candidate_intent = [
        canonical_mathml_tokens(document, include_intent=True)
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
