"""Flow 2: latex2mathml skeleton followed by semantic JSON annotation."""

from __future__ import annotations

from collections.abc import Sequence

from ..config import Settings
from ..llm.base import LLMClient
from ..mathml.concept_cards import ConceptCard
from ..mathml.semantic import semantic_json_schema
from ..models import MathContext
from .prompts import semantic_messages


class HybridIntentAgent:
    role = "semantic-intent-annotator"

    def __init__(self, llm: LLMClient, settings: Settings | None = None) -> None:
        self.llm = llm
        self.settings = settings or Settings.from_env()

    def run(
        self,
        latex: str,
        base_mathml: str,
        context: MathContext,
        feedback: list[str] | None = None,
        *,
        allow_open_concepts: bool = True,
        allow_all_open_concepts: bool = False,
        concept_cards: Sequence[ConceptCard] | None = None,
    ) -> str:
        """Return a semantic JSON decision for a tree with temporary ids."""

        schema = None
        if self.settings.structured_output:
            names = [card.name for card in concept_cards] if concept_cards else None
            schema = semantic_json_schema(names)
        return self.llm.chat(
            semantic_messages(
                latex,
                base_mathml,
                context,
                feedback,
                allow_open_concepts=allow_open_concepts,
                allow_all_open_concepts=allow_all_open_concepts,
                concept_cards=concept_cards,
                allowed_open_concepts=self.settings.allowed_open_concepts,
                static_system=self.settings.static_system_prompt,
            ),
            temperature=self.settings.temperature,
            num_predict=self.settings.num_predict,
            json_schema=schema,
        )
