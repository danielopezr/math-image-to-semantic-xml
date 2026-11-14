"""Retrieval of concept cards for the semantic intent annotator."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from lxml import etree

from ..models import MathContext
from .annotator import FUNCTION_CONCEPTS
from .concept_cards import ConceptCard, get_concept_cards
from .concepts import ConceptRegistry, get_registry
from .dom import element_children, element_text, is_element, localname

_LATEX_COMMAND_RE = re.compile(r"\\[A-Za-z]+")
_IDENT_RE = re.compile(r"[A-Za-z]+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_EXPONENTIAL_E_RE = re.compile(r"(?<![A-Za-z])e\s*\^")

_SYMBOL_CONCEPTS = {
    "\u2211": "sum",
    "\u220f": "product",
    "\u222b": "definite-integral",
    "\u2202": "partial-derivative",
    "!": "factorial",
    "|": "absolute-value",
    "\u2016": "norm",
    "\u2225": "norm",
    "\u221a": "root",
}

_DEFAULT_OPERATOR_CONCEPTS = frozenset(
    {
        "and",
        "applied-to",
        "approximately",
        "composed-with",
        "congruent",
        "defined-as",
        "divided-by",
        "equals",
        "equivalent-to",
        "evaluates-to",
        "given",
        "greater-than",
        "greater-than-or-equal-to",
        "identically-equals",
        "if-and-only-if",
        "implies",
        "invisible-separator",
        "invisible-times",
        "less-than",
        "less-than-or-equal-to",
        "list-separator",
        "maps-to",
        "minus",
        "minus-or-plus",
        "not",
        "not-equal-to",
        "not-parallel-to",
        "obtained-from",
        "or",
        "parallel-to",
        "perpendicular",
        "plus",
        "plus-or-minus",
        "precedes",
        "proportional",
        "range-separator",
        "similar",
        "such-that",
        "succeeds",
        "tends-to",
        "tends-to-from-above",
        "tends-to-from-below",
        "time-separator",
        "times",
        "xor",
    }
)

_CONTEXT_ONLY_CONCEPTS = frozenset(
    {
        "cartesian-coordinate",
        "polar-coordinate",
        "spherical-coordinate",
    }
)

_CONTEXT_HINTS: dict[str, tuple[str, ...]] = {
    "cartesiana": ("cartesian-coordinate",),
    "cartesian": ("cartesian-coordinate",),
    "coordenada": ("coordinate",),
    "coordinate": ("coordinate",),
    "esferica": ("spherical-coordinate",),
    "polar": ("polar-coordinate",),
    "spherical": ("spherical-coordinate",),
    "trigonometria": (
        "sine",
        "cosine",
        "tangent",
        "secant",
        "cosecant",
        "cotangent",
        "arcsine",
        "arccosine",
        "arctangent",
    ),
    "trigonometry": (
        "sine",
        "cosine",
        "tangent",
        "secant",
        "cosecant",
        "cotangent",
    ),
    "calculo": (
        "limit",
        "derivative",
        "partial-derivative",
        "definite-integral",
        "sum",
        "product",
    ),
    "analisis": (
        "limit",
        "derivative",
        "definite-integral",
        "sum",
        "product",
    ),
    "algebra": (
        "fraction",
        "power",
        "root",
        "indexed-by",
        "factorial",
        "absolute-value",
    ),
    "geometria": (
        "distance",
        "area",
        "volume",
        "line-segment",
        "angle",
        "vector",
        "norm",
    ),
    "estadistica": (
        "mean",
        "median",
        "mode",
        "variance",
        "standard-deviation",
        "probability",
    ),
    "probabilidad": (
        "probability",
        "conditional-probability",
        "mean",
        "variance",
    ),
    "matriz": (
        "matrix",
        "determinant",
        "transpose",
        "inverse",
        "norm",
    ),
    "linear algebra": (
        "matrix",
        "determinant",
        "transpose",
        "inverse",
        "norm",
        "vector",
        "inner-product",
    ),
    "algebra lineal": (
        "matrix",
        "determinant",
        "transpose",
        "inverse",
        "norm",
        "vector",
        "inner-product",
    ),
}


@dataclass(frozen=True)
class MathFeatures:
    tokens: frozenset[str]
    tags: frozenset[str]
    symbols: frozenset[str]
    latex: frozenset[str]
    functions: frozenset[str]
    anchors: frozenset[str]
    context_concepts: frozenset[str]

    @property
    def all(self) -> frozenset[str]:
        return frozenset(
            set(self.tokens)
            | set(self.tags)
            | set(self.symbols)
            | set(self.latex)
            | set(self.functions)
        )


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(
        char for char in decomposed if not unicodedata.combining(char)
    ).lower()


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_WORD_RE.findall(_fold(text)))


def _values(text: str) -> frozenset[str]:
    folded = _fold(text).strip()
    if not folded:
        return frozenset()
    return frozenset((*_tokens(folded), folded))


def _card_tokens(card: ConceptCard) -> frozenset[str]:
    values: set[str] = set()
    for alias in (card.name, *card.aliases):
        values.update(_values(alias))
    for cue in (*card.latex, *card.mathml_tags, *card.symbols):
        values.add(_fold(cue))
        values.add(_fold(cue).lstrip("\\"))
    return frozenset(value for value in values if value)


def _latex_features(latex: str) -> tuple[set[str], set[str], set[str]]:
    tokens: set[str] = set()
    commands: set[str] = set()
    symbols: set[str] = set()
    text = latex or ""

    tokens.update(_tokens(text))
    for match in _LATEX_COMMAND_RE.finditer(text):
        command = match.group(0)
        commands.add(command)
        commands.add(command.lower())
        name = command[1:]
        commands.add(f"\\{name.lower()}")
        tokens.add(name.lower())
    for name in _IDENT_RE.findall(text):
        tokens.add(name.lower())
    for symbol in "^_/[]()|!+*=-<>":
        if symbol in text:
            symbols.add(symbol)
    if _EXPONENTIAL_E_RE.search(text):
        commands.add("e^")
    return tokens, commands, symbols


def _context_concepts(context: MathContext) -> set[str]:
    text = _fold(
        " ".join(
            (
                context.area,
                context.topic,
                context.surrounding_text[:2000],
            )
        )
    )
    concepts: set[str] = set()
    for cue, names in _CONTEXT_HINTS.items():
        if _fold(cue) in text:
            concepts.update(names)
    return concepts


def _is_name_element(element: etree._Element) -> bool:
    tag = localname(element.tag)
    if tag == "mi":
        return bool(element_text(element))
    if tag == "msub":
        children = element_children(element)
        return len(children) == 2 and localname(children[0].tag) == "mi"
    return False


def _matching_paren(children: list[etree._Element], open_index: int) -> int | None:
    depth = 0
    for index in range(open_index, len(children)):
        child = children[index]
        if localname(child.tag) != "mo":
            continue
        text = element_text(child)
        if text == "(":
            depth += 1
        elif text == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _known_function_name(element: etree._Element) -> str | None:
    if localname(element.tag) == "mi":
        text = element_text(element).lower()
        return text if text in FUNCTION_CONCEPTS else None
    if localname(element.tag) == "msub":
        children = element_children(element)
        if children:
            text = element_text(children[0]).lower()
            return text if text in FUNCTION_CONCEPTS else None
    return None


def _has_generic_application(root: etree._Element) -> bool:
    for element in root.iter():
        if not is_element(element) or localname(element.tag) != "mrow":
            continue
        children = element_children(element)
        for index in range(len(children) - 2):
            name = children[index]
            if not _is_name_element(name):
                continue
            if localname(children[index + 1].tag) != "mo":
                continue
            if element_text(children[index + 1]) != "(":
                continue
            close_index = _matching_paren(children, index + 1)
            if close_index is None or close_index <= index + 2:
                continue
            if _known_function_name(name) is None:
                return True
    return False


def _structural_anchors(
    element: etree._Element,
    anchors: set[str],
    symbols: set[str],
    functions: set[str],
    tokens: set[str],
) -> None:
    tag = localname(element.tag)
    text = element_text(element)

    if tag == "mo":
        symbols.add(text)
        symbols.update(_tokens(text))
        concept = _SYMBOL_CONCEPTS.get(text)
        if concept is not None:
            anchors.add(concept)
    elif tag == "mi":
        lowered = text.lower()
        tokens.add(lowered)
        if lowered in FUNCTION_CONCEPTS:
            functions.add(lowered)
            anchors.add(FUNCTION_CONCEPTS[lowered])
    elif tag in {"mn", "mtext"}:
        tokens.update(_tokens(text))

    if tag in {"mfrac"}:
        anchors.add("fraction")
    elif tag in {"msqrt", "mroot"}:
        anchors.add("root")
    elif tag in {"msup", "msubsup"}:
        children = element_children(element)
        if children:
            base = element_text(children[0])
            if base == "e":
                anchors.add("exponential")
            elif base.lower() in FUNCTION_CONCEPTS:
                anchors.add(FUNCTION_CONCEPTS[base.lower()])
            elif tag == "msup":
                anchors.add("power")
    if tag in {"msub", "msubsup"}:
        children = element_children(element)
        if children:
            base = element_text(children[0])
            lowered = base.lower()
            if lowered in FUNCTION_CONCEPTS:
                anchors.add(FUNCTION_CONCEPTS[lowered])
            elif base == "lim":
                anchors.add("limit")
            elif tag == "msub":
                anchors.add("indexed-by")
    if tag in {"munder", "munderover"}:
        children = element_children(element)
        if children:
            base = element_text(children[0])
            concept = _SYMBOL_CONCEPTS.get(base)
            if concept is not None:
                anchors.add(concept)
            elif base == "lim":
                anchors.add("limit")


def extract_features(
    latex: str | None,
    root: etree._Element | None,
    context: MathContext | None = None,
) -> MathFeatures:
    tokens, commands, symbols = _latex_features(latex or "")
    tags: set[str] = set()
    functions: set[str] = set()
    anchors: set[str] = set()

    if root is not None:
        for element in root.iter():
            if not is_element(element):
                continue
            tags.add(localname(element.tag))
            _structural_anchors(element, anchors, symbols, functions, tokens)
        if _has_generic_application(root):
            anchors.add("generic-application")

    return MathFeatures(
        tokens=frozenset(tokens),
        tags=frozenset(tags),
        symbols=frozenset(symbols),
        latex=frozenset(commands),
        functions=frozenset(functions),
        anchors=frozenset(anchors),
        context_concepts=frozenset(
            _context_concepts(context) if context is not None else ()
        ),
    )


class ConceptRAG:
    def __init__(
        self,
        registry: ConceptRegistry | None = None,
        cards: tuple[ConceptCard, ...] | None = None,
    ) -> None:
        self.registry = registry or get_registry()
        self.cards = cards if cards is not None else get_concept_cards()
        self.by_name = {card.name: card for card in self.cards}
        self._search_tokens = {card.name: _card_tokens(card) for card in self.cards}

    def _allowed(
        self,
        card: ConceptCard,
        allow_open: bool,
        allowed_open_concepts: set[str] | None,
    ) -> bool:
        if card.kind != "open":
            return True
        if not allow_open:
            return False
        return not (
            allowed_open_concepts is not None and card.name not in allowed_open_concepts
        )

    def _score(self, card: ConceptCard, features: MathFeatures) -> float:
        score = 0.0
        if card.name in features.anchors:
            score += 100.0
        if card.name in features.context_concepts:
            score += 4.0

        for alias in card.aliases:
            alias_tokens = _tokens(alias)
            if alias_tokens and alias_tokens <= features.tokens:
                score += 7.0
                break

        for command in card.latex:
            if command in features.latex:
                score += 12.0
                break

        for symbol in card.symbols:
            if symbol in features.symbols or _fold(symbol) in features.tokens:
                score += 10.0
                break

        for tag in card.mathml_tags:
            if tag in features.tags:
                score += 5.0
                break

        overlap = {
            token
            for token in self._search_tokens[card.name] & features.tokens
            if len(token) > 1
        }
        if overlap:
            score += min(6.0, 1.5 * len(overlap))

        if score <= 0:
            return 0.0
        if card.kind == "core":
            score += 0.5
        return score + float(card.priority)

    def retrieve(
        self,
        latex: str | None,
        root: etree._Element | None,
        context: MathContext | None = None,
        *,
        allow_open: bool = True,
        allowed_open_concepts: set[str] | None = None,
        top_k: int = 8,
    ) -> list[ConceptCard]:
        if top_k <= 0:
            return []

        features = extract_features(latex, root, context)
        limit = max(top_k, len(features.anchors))
        scored: list[tuple[float, ConceptCard]] = []

        for card in self.cards:
            if card.name in _DEFAULT_OPERATOR_CONCEPTS:
                continue
            if (
                card.name in _CONTEXT_ONLY_CONCEPTS
                and card.name not in features.context_concepts
            ):
                continue
            if not self._allowed(card, allow_open, allowed_open_concepts):
                continue
            if card.requires and not (set(card.requires) & set(features.all)):
                continue
            if card.requires_all and not (set(card.requires_all) <= set(features.all)):
                continue
            score = self._score(card, features)
            if score >= 2.5:
                scored.append((score, card))

        scored.sort(key=lambda item: (-item[0], item[1].name))
        selected = [card for _, card in scored[:limit]]
        seen = {card.name for card in selected}

        for name in sorted(features.anchors):
            card = self.by_name.get(name)
            if card is None or name in seen:
                continue
            if not self._allowed(card, allow_open, allowed_open_concepts):
                continue
            selected.append(card)
            seen.add(name)

        return selected[:limit]
