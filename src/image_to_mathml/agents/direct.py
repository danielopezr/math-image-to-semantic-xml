"""Flow 1: LaTeX directly to intent-annotated MathML via Ollama."""

from __future__ import annotations

from ..config import Settings
from ..llm.base import LLMClient
from ..models import MathContext
from .prompts import direct_messages


class DirectMathMLAgent:
    role = "direct-mathml-generator"

    def __init__(self, llm: LLMClient, settings: Settings | None = None) -> None:
        self.llm = llm
        self.settings = settings or Settings.from_env()

    def run(
        self,
        latex: str,
        context: MathContext,
        feedback: list[str] | None = None,
        *,
        allow_open_concepts: bool = True,
        allow_all_open_concepts: bool = False,
    ) -> str:
        return self.llm.chat(
            direct_messages(
                latex,
                context,
                feedback,
                allow_open_concepts=allow_open_concepts,
                allow_all_open_concepts=allow_all_open_concepts,
            ),
            temperature=0.0,
            num_predict=self.settings.num_predict,
        )
