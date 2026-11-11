"""Direct image-to-MathML agent backed by an Ollama vision model."""

from __future__ import annotations

from ..config import Settings
from ..llm.ollama_vision import OllamaVisionClient
from ..models import MathContext
from ..ocr.preprocess import PreprocessConfig, preprocess_image
from .prompts import vision_messages


class VisionMathMLAgent:
    role = "vision-mathml-generator"

    def __init__(
        self,
        settings: Settings | None = None,
        client: OllamaVisionClient | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.client = client or OllamaVisionClient(
            model=self.settings.ollama_vision_model,
            base_url=self.settings.ollama_base_url,
            timeout=self.settings.ollama_timeout,
            num_predict=self.settings.ollama_vision_num_predict,
            num_ctx=self.settings.ollama_vision_num_ctx,
            max_side=self.settings.ollama_vision_max_side,
        )
        self.model = self.client.model

    def is_available(self) -> bool:
        return self.client.is_available()

    def has_model(self) -> bool:
        return self.client.has_model()

    def run(
        self,
        image_path: str,
        context: MathContext,
        feedback: list[str] | None = None,
        *,
        allow_open_concepts: bool = True,
        allow_all_open_concepts: bool = False,
    ) -> str:
        preprocess = PreprocessConfig.from_env(
            enabled=self.settings.ocr_preprocess,
            profile=self.settings.ocr_preprocess_profile,
        )
        image = preprocess_image(image_path, preprocess)
        return self.client.chat(
            vision_messages(
                context,
                feedback,
                allow_open_concepts=allow_open_concepts,
                allow_all_open_concepts=allow_all_open_concepts,
            ),
            image,
            temperature=0.0,
            num_predict=self.settings.ollama_vision_num_predict,
        )
