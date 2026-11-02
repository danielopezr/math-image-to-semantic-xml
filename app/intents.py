"""Vocabulario controlado de intents (fuente unica: app/intent_vocabulary.json).

Es un vocabulario INTERNO del proyecto. Sus nombres no son valores oficiales de MathML
ni nombres de concepto del atributo ``intent`` del W3C.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

VOCABULARY_PATH = Path(__file__).with_name("intent_vocabulary.json")


class VocabularyError(ValueError):
    """El vocabulario es inconsistente."""


@dataclass(frozen=True)
class RoleDef:
    name: str
    required: bool


@dataclass(frozen=True)
class IntentDef:
    id: str
    category: str
    description: str
    roles: tuple[RoleDef, ...] = ()
    variadic_role: Optional[str] = None
    variadic_min: int = 0
    properties: dict = field(default_factory=dict)  # nombre -> lista de valores o "integer"
    required_properties: tuple[str, ...] = ()
    both_or_none: tuple[tuple[str, ...], ...] = ()
    mathml_hint: tuple[str, ...] = ()
    example: str = ""
    note: str = ""

    @property
    def role_names(self) -> tuple[str, ...]:
        names = [r.name for r in self.roles]
        if self.variadic_role:
            names.append(self.variadic_role)
        return tuple(names)


@dataclass(frozen=True)
class Vocabulary:
    version: str
    node_types: tuple[str, ...]
    unresolved_child_role: str
    intents: dict[str, IntentDef]

    def get(self, intent_id: Optional[str]) -> Optional[IntentDef]:
        return self.intents.get(intent_id) if intent_id else None

    def ids(self) -> list[str]:
        return list(self.intents)

    def prompt_table(self) -> str:
        """Tabla compacta para el prompt del agente semantico."""
        lines = []
        for d in self.intents.values():
            roles = [r.name + ("" if r.required else "?") for r in d.roles]
            if d.variadic_role:
                roles.append(f"{d.variadic_role}*(min {d.variadic_min})")
            props = "; ".join(
                f"{k}={'|'.join(v) if isinstance(v, list) else v}" for k, v in d.properties.items()
            )
            lines.append(f"- {d.id} [node_type={d.category}] roles: {', '.join(roles) or '(ninguno)'}"
                         + (f" | properties: {props}" if props else ""))
        return "\n".join(lines)


def _build(doc: dict) -> Vocabulary:
    node_types = tuple(doc.get("node_types", ()))
    if "unresolved" not in node_types:
        raise VocabularyError("node_types debe incluir 'unresolved'.")
    intents: dict[str, IntentDef] = {}
    for raw in doc.get("intents", []):
        iid = raw.get("id")
        if not iid or iid in intents:
            raise VocabularyError(f"Id de intent ausente o duplicado: {iid!r}")
        if raw["category"] not in node_types or raw["category"] == "unresolved":
            raise VocabularyError(f"{iid}: categoria invalida {raw['category']!r}")
        roles = tuple(RoleDef(r["name"], bool(r["required"])) for r in raw.get("roles", []))
        variadic = raw.get("variadic")
        names = [r.name for r in roles] + ([variadic["role"]] if variadic else [])
        if len(set(names)) != len(names):
            raise VocabularyError(f"{iid}: roles duplicados")
        for group in raw.get("both_or_none", []):
            if any(g not in names for g in group):
                raise VocabularyError(f"{iid}: both_or_none referencia un rol inexistente")
        for rp in raw.get("required_properties", []):
            if rp not in raw.get("properties", {}):
                raise VocabularyError(f"{iid}: propiedad requerida no declarada: {rp}")
        intents[iid] = IntentDef(
            id=iid, category=raw["category"], description=raw["description"], roles=roles,
            variadic_role=variadic["role"] if variadic else None,
            variadic_min=variadic["min"] if variadic else 0,
            properties=dict(raw.get("properties", {})),
            required_properties=tuple(raw.get("required_properties", ())),
            both_or_none=tuple(tuple(g) for g in raw.get("both_or_none", [])),
            mathml_hint=tuple(raw.get("mathml_hint", ())),
            example=raw.get("example", ""), note=raw.get("note", ""),
        )
    if not intents:
        raise VocabularyError("El vocabulario no contiene intents.")
    return Vocabulary(doc["vocabulary_version"], node_types, doc.get("unresolved_child_role", "component"), intents)


@lru_cache(maxsize=4)
def _load_cached(path: str) -> Vocabulary:
    return _build(json.loads(Path(path).read_text(encoding="utf-8")))


def load_vocabulary(path: Optional[Union[str, Path]] = None) -> Vocabulary:
    return _load_cached(str(path or VOCABULARY_PATH))
