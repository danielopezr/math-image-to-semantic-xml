"""Flow 2: latex2mathml skeleton followed by semantic JSON annotation."""

from __future__ import annotations

from collections.abc import Sequence

from ..config import Settings
from ..llm.base import LLMClient
from ..mathml.concept_cards import ConceptCard
from ..models import MathContext
from .prompts import hybrid_messages, semantic_messages


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

        return self.llm.chat(
            semantic_messages(
                latex,
                base_mathml,
                context,
                feedback,
                allow_open_concepts=allow_open_concepts,
                allow_all_open_concepts=allow_all_open_concepts,
                concept_cards=concept_cards,
            ),
            temperature=0.0,
            num_predict=self.settings.num_predict,
        )

    def run_legacy(
        self,
        latex: str,
        base_mathml: str,
        context: MathContext,
        feedback: list[str] | None = None,
        *,
        allow_open_concepts: bool = True,
        allow_all_open_concepts: bool = False,
    ) -> str:
        """Generate annotated MathML directly (compatibility fallback)."""

        return self.llm.chat(
            hybrid_messages(
                latex,
                base_mathml,
                context,
                feedback,
                allow_open_concepts=allow_open_concepts,
                allow_all_open_concepts=allow_all_open_concepts,
            ),
            temperature=0.0,
            num_predict=self.settings.num_predict,
        )
