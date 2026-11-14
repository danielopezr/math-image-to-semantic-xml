"""Retrieval of open-list concept cards for the semantic intent annotator.

The core concepts have a compact, hand-tuned retrieval layer in
:mod:`concept_rag`.  The W3C open list is much larger (nearly a thousand
names), so this module complements the core RAG with a context-first index:

* every open concept gets a card generated from the local registry, so no
  retrieval path can introduce an unknown name;
* ``open_concept_hints.json`` maps textbook areas and topics to open concepts
  and adds LaTeX, symbol and structural cues for common concepts;
* the open-concept policy (``allow_open``/``allowed_open_concepts``) is applied
  before any card is returned.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, replace
from functools import lru_cache
from importlib.resources import files

from lxml import etree

from ..models import MathContext
from .concept_cards import ConceptCard, build_concept_cards
from .concept_rag import MathFeatures, extract_features
from .concepts import ConceptRegistry, get_registry

_WORD_RE = re.compile(r"[a-z0-9]+")
_LATEX_COMMAND_RE = re.compile(r"\\[A-Za-z]+")

_TUPLE_FIELDS = (
    "aliases",
    "latex",
    "mathml_tags",
    "symbols",
    "examples",
    "requires",
    "requires_all",
)
_REPLACE_FIELDS = ("roles", "targets", "priority")

_GENERIC_TOKENS = frozenset(
    {
        "al",
        "and",
        "con",
        "constant",
        "constante",
        "de",
        "del",
        "distribution",
        "distribucion",
        "ecuacion",
        "el",
        "en",
        "equation",
        "espacio",
        "formula",
        "funcion",
        "function",
        "group",
        "grupo",
        "integral",
        "la",
        "las",
        "los",
        "matriz",
        "matrix",
        "number",
        "numero",
        "of",
        "operador",
        "operator",
        "para",
        "polinomio",
        "polynomial",
        "por",
        "simbolo",
        "sobre",
        "space",
        "symbol",
        "the",
        "transform",
        "transformada",
        "un",
        "una",
        "y",
    }
)


@dataclass(frozen=True)
class OpenConceptEntry:
    """One retrieved open concept with its registry card."""

    card: ConceptCard
    areas: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(
        char for char in decomposed if not unicodedata.combining(char)
    ).lower()


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_WORD_RE.findall(_fold(text)))


def _phrase_present(text: str, phrase: str) -> bool:
    if not phrase or not text:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _as_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return ()


def _read_hints() -> dict[str, object]:
    path = files("image_to_mathml").joinpath("data", "open_concept_hints.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_card(card: ConceptCard, hint: dict[str, object]) -> ConceptCard:
    changes: dict[str, object] = {}
    for field in _TUPLE_FIELDS:
        extra = _as_tuple(hint.get(field))
        if extra:
            current = tuple(getattr(card, field))
            changes[field] = tuple(dict.fromkeys((*current, *extra)))
    for field in _REPLACE_FIELDS:
        if field in hint:
            value = hint[field]
            if field == "priority":
                changes[field] = int(value)  # type: ignore[arg-type]
            else:
                changes[field] = _as_tuple(value)
    return replace(card, **changes) if changes else card


def _normalized_index(
    raw: dict[str, object],
    valid: set[str],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    index: list[tuple[str, tuple[str, ...]]] = []
    for phrase, names in raw.items():
        normalized = " ".join(_fold(str(phrase)).split())
        accepted = tuple(
            dict.fromkeys(str(name) for name in _as_tuple(names) if str(name) in valid)
        )
        if normalized and accepted:
            index.append((normalized, accepted))
    return tuple(index)


def _search_tokens(card: ConceptCard) -> frozenset[str]:
    values: set[str] = set()
    for alias in (card.name, *card.aliases):
        values.update(_tokens(alias))
    for cue in (*card.symbols, *card.mathml_tags):
        values.update(_tokens(cue))
        folded = _fold(cue).lstrip("\\")
        if len(folded) > 1:
            values.add(folded)
    return frozenset(value for value in values if value)


def _symbol_matches(symbol: str, features: MathFeatures) -> bool:
    folded = _fold(symbol).strip()
    if len(folded) <= 1:
        return False
    if symbol in features.symbols:
        return True
    return folded.isascii() and folded in features.tokens


def _has_direct_signal(
    card: ConceptCard,
    features: MathFeatures,
    raw_commands: frozenset[str],
) -> bool:
    if any(command in raw_commands for command in card.latex):
        return True
    return any(_symbol_matches(symbol, features) for symbol in card.symbols)


def _entry_areas_topics(
    name: str,
    area_index: tuple[tuple[str, tuple[str, ...]], ...],
    topic_index: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    areas = tuple(phrase for phrase, names in area_index if name in names)
    topics = tuple(phrase for phrase, names in topic_index if name in names)
    return areas, topics


class OpenConceptRAG:
    """Select candidate open concepts from area, topic and notation cues."""

    def __init__(
        self,
        registry: ConceptRegistry | None = None,
        entries: tuple[OpenConceptEntry, ...] | None = None,
    ) -> None:
        self.registry = registry or get_registry()
        hints = _read_hints()
        raw_cues = hints.get("cues", {})
        if not isinstance(raw_cues, dict):
            raw_cues = {}

        if entries is None:
            open_names = set(self.registry.open) - set(self.registry.core)
            cards = {card.name: card for card in build_concept_cards(self.registry)}
            entries = []
            for name in sorted(open_names):
                card = cards.get(name)
                if card is None:
                    continue
                hint = raw_cues.get(name, {})
                if not isinstance(hint, dict):
                    hint = {}
                entries.append(OpenConceptEntry(card=_merge_card(card, hint)))
            entries = tuple(entries)

        self.entries = entries
        self.by_name = {entry.card.name: entry for entry in entries}
        self._search_tokens = {
            entry.card.name: _search_tokens(entry.card) for entry in entries
        }

        valid = {name for name in self.registry.open if name not in self.registry.core}
        raw_areas = hints.get("areas", {})
        raw_topics = hints.get("topics", {})
        self._area_index = _normalized_index(
            raw_areas if isinstance(raw_areas, dict) else {}, valid
        )
        self._topic_index = _normalized_index(
            raw_topics if isinstance(raw_topics, dict) else {}, valid
        )

        with_areas = []
        for entry in entries:
            areas, topics = _entry_areas_topics(
                entry.card.name, self._area_index, self._topic_index
            )
            with_areas.append(replace(entry, areas=areas, topics=topics))
        self.entries = tuple(with_areas)
        self.by_name = {entry.card.name: entry for entry in self.entries}

    def _allowed(
        self,
        name: str,
        allow_open: bool,
        allowed_open_concepts: set[str] | None,
    ) -> bool:
        if not allow_open:
            return False
        if allowed_open_concepts is None:
            return True
        return name in allowed_open_concepts

    def _bonuses(
        self,
        text: str,
        index: tuple[tuple[str, tuple[str, ...]], ...],
        base: float,
    ) -> dict[str, float]:
        bonuses: dict[str, float] = {}
        for phrase, names in index:
            if not _phrase_present(text, phrase):
                continue
            specificity = min(12.0, len(phrase) / 4.0)
            weight = base + specificity
            for name in names:
                bonuses[name] = max(bonuses.get(name, 0.0), weight)
        return bonuses

    def _score(
        self,
        card: ConceptCard,
        features: MathFeatures,
        area_text: str,
        topic_text: str,
        context_text: str,
        area_tokens: frozenset[str],
        topic_tokens: frozenset[str],
        context_tokens: frozenset[str],
        area_bonus: dict[str, float],
        topic_bonus: dict[str, float],
        raw_commands: frozenset[str],
    ) -> float:
        score = area_bonus.get(card.name, 0.0)
        score += topic_bonus.get(card.name, 0.0)

        if card.name in features.anchors:
            score += 120.0
        if card.name in features.context_concepts:
            score += 8.0

        for alias in card.aliases:
            alias_tokens = _tokens(alias)
            if not alias_tokens:
                continue
            if alias_tokens <= topic_tokens:
                score += 22.0
                break
            if alias_tokens <= area_tokens:
                score += 16.0
                break
            if alias_tokens <= features.tokens:
                score += 10.0
                break

        if any(command in raw_commands for command in card.latex):
            score += 16.0

        if any(_symbol_matches(symbol, features) for symbol in card.symbols):
            score += 14.0

        for tag in card.mathml_tags:
            if tag in features.tags:
                score += 6.0
                break

        search_tokens = self._search_tokens[card.name]
        for source, weight, cap in (
            (topic_tokens, 6.0, 18.0),
            (area_tokens, 4.0, 12.0),
            (context_tokens, 1.5, 6.0),
            (features.tokens, 2.0, 10.0),
        ):
            overlap = {
                token
                for token in search_tokens & source
                if len(token) > 1 and token not in _GENERIC_TOKENS
            }
            if overlap:
                score += min(cap, weight * len(overlap))

        for cue in card.symbols:
            if _phrase_present(context_text, _fold(cue)):
                score += 3.0
                break

        if score <= 0:
            return 0.0
        return score + float(card.priority)

    def retrieve(
        self,
        latex: str | None,
        root: etree._Element | None,
        context: MathContext | None = None,
        *,
        allow_open: bool = True,
        allowed_open_concepts: set[str] | None = None,
        top_k: int = 5,
    ) -> list[ConceptCard]:
        if top_k <= 0 or not allow_open:
            return []

        features = extract_features(latex, root, context)
        area_text = _fold(context.area) if context is not None else ""
        topic_text = _fold(context.topic) if context is not None else ""
        surrounding = context.surrounding_text[:2000] if context is not None else ""
        context_text = _fold(
            f"{context.area} {context.topic} {surrounding}"
            if context is not None
            else ""
        )
        area_tokens = _tokens(area_text)
        topic_tokens = _tokens(topic_text)
        context_tokens = _tokens(context_text)
        area_bonus = self._bonuses(area_text, self._area_index, 18.0)
        topic_bonus = self._bonuses(topic_text, self._topic_index, 28.0)
        has_context = bool(context_text.strip())
        raw_commands = frozenset(_LATEX_COMMAND_RE.findall(latex or ""))

        scored: list[tuple[float, ConceptCard]] = []
        for entry in self.entries:
            card = entry.card
            if not self._allowed(card.name, allow_open, allowed_open_concepts):
                continue
            if (
                not has_context
                and card.name not in features.anchors
                and not _has_direct_signal(card, features, raw_commands)
            ):
                continue
            score = self._score(
                card,
                features,
                area_text,
                topic_text,
                context_text,
                area_tokens,
                topic_tokens,
                context_tokens,
                area_bonus,
                topic_bonus,
                raw_commands,
            )
            if score >= 3.0:
                scored.append((score, card))

        scored.sort(key=lambda item: (-item[0], item[1].name))
        return [card for _, card in scored[:top_k]]


@lru_cache(maxsize=1)
def get_open_concept_entries() -> tuple[OpenConceptEntry, ...]:
    return OpenConceptRAG().entries


def get_open_concept_cards() -> tuple[ConceptCard, ...]:
    return tuple(entry.card for entry in get_open_concept_entries())


def get_open_concept_card(
    name: str,
    cards: tuple[ConceptCard, ...] | None = None,
) -> ConceptCard | None:
    for card in cards or get_open_concept_cards():
        if card.name == name:
            return card
    return None
