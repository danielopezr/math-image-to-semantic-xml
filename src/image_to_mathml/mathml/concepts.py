"""Local W3C intent concept registries and validation helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

CORE_PROPERTIES = {
    # fixity
    "function",
    "prefix",
    "infix",
    "postfix",
    "silent",
    "nofix",
    # style / interpretation
    "literal",
    "common",
    "legacy",
    "verbose",
    "terse",
    "unit",
    "chemistry",
    "matrix",
    "array",
    "system-of-equations",
    # pauses
    "pause-short",
    "pause-medium",
    "pause-long",
}


@dataclass(frozen=True)
class Signature:
    fixity: str
    min_arity: int
    max_arity: int | None

    def accepts(self, arity: int) -> bool:
        if arity < self.min_arity:
            return False
        return self.max_arity is None or arity <= self.max_arity


@dataclass(frozen=True)
class Concept:
    name: str
    kind: str
    signatures: tuple[Signature, ...]

    def accepts_arity(self, arity: int) -> bool:
        return any(signature.accepts(arity) for signature in self.signatures)


def _read_registry(filename: str) -> dict[str, Concept]:
    path = files("image_to_mathml") / "data" / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    concepts: dict[str, Concept] = {}
    for name, entry in payload.get("concepts", {}).items():
        signatures = tuple(
            Signature(
                fixity=signature.get("fixity", "function"),
                min_arity=int(signature.get("min_arity", 0)),
                max_arity=(
                    None
                    if signature.get("max_arity") is None
                    else int(signature["max_arity"])
                ),
            )
            for signature in entry.get("signatures", [])
        )
        concepts[name] = Concept(
            name=name,
            kind=entry.get("kind", "core"),
            signatures=signatures,
        )
    return concepts


# Associative infix operators: grouping does not change the meaning, so
# times($a,$b,$c) is equivalent to times($a,times($b,$c)).  Content MathML
# accepts any number of operands for them.  minus, divided-by and power are
# deliberately absent: their order and grouping change the meaning.
ASSOCIATIVE_CONCEPTS = frozenset(
    {"times", "plus", "and", "or", "union", "intersection"}
)


def _env_nary_default() -> bool:
    value = os.getenv("IMAGE_TO_MATHML_NARY_ASSOCIATIVE")
    if value is None or not value.strip():
        return True
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class ConceptRegistry:
    """Known W3C concept names and arities.

    ``nary_associative`` lets the associative operators in
    :data:`ASSOCIATIVE_CONCEPTS` take two or more arguments.  The W3C intent
    text could not be verified on this point, so it is configurable
    (``IMAGE_TO_MATHML_NARY_ASSOCIATIVE=false`` restores exactly two).
    """

    def __init__(self, nary_associative: bool | None = None) -> None:
        self.core = _read_registry("core_concepts.json")
        self.open = _read_registry("open_concepts.json")
        self.nary_associative = (
            _env_nary_default() if nary_associative is None else nary_associative
        )

    def get(self, name: str) -> Concept | None:
        return self.core.get(name) or self.open.get(name)

    def classification(self, name: str) -> str:
        """Return ``core``, ``open``, ``literal`` or ``unknown``."""

        if name.startswith("_"):
            return "literal"
        if name in self.core:
            return "core"
        if name in self.open:
            return "open"
        return "unknown"

    def accepts_arity(self, name: str, arity: int) -> bool:
        if name.startswith("_"):
            return True
        concept = self.get(name)
        if concept is None:
            return False
        if self.nary_associative and name in ASSOCIATIVE_CONCEPTS and arity >= 2:
            return True
        return concept.accepts_arity(arity)

    def signatures(self, name: str) -> tuple[Signature, ...]:
        """Signatures of ``name`` as this registry accepts them."""

        concept = self.get(name)
        if concept is None:
            return ()
        if self.nary_associative and name in ASSOCIATIVE_CONCEPTS:
            return tuple(
                Signature(signature.fixity, signature.min_arity, None)
                if signature.fixity == "infix"
                else signature
                for signature in concept.signatures
            )
        return concept.signatures

    def is_allowed(self, name: str, *, allow_open: bool) -> bool:
        classification = self.classification(name)
        if classification in {"core", "literal"}:
            return True
        if classification == "open":
            return allow_open
        return False

    def names(self, *, include_open: bool = False) -> list[str]:
        names = set(self.core)
        if include_open:
            names |= set(self.open)
        return sorted(names)


def get_registry(nary_associative: bool | None = None) -> ConceptRegistry:
    """Shared registry; ``None`` reads ``IMAGE_TO_MATHML_NARY_ASSOCIATIVE``."""

    return _cached_registry(
        _env_nary_default() if nary_associative is None else nary_associative
    )


@lru_cache(maxsize=2)
def _cached_registry(nary_associative: bool) -> ConceptRegistry:
    return ConceptRegistry(nary_associative=nary_associative)
