"""Flow 2: latex2mathml skeleton followed by Ollama intent annotation."""

from __future__ import annotations

from ..config import Settings
from ..llm.base import LLMClient
from ..models import MathContext
from .prompts import hybrid_messages


class HybridIntentAgent:
    role = "intent-annotator"

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
    ) -> str:
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
