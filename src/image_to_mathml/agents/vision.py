"""Direct image-to-MathML agent backed by an Ollama vision model."""

from __future__ import annotations

import os

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
            base_url=self.settings.vision_base_url,
            timeout=self.settings.ollama_vision_timeout,
            num_predict=self.settings.ollama_vision_num_predict,
            num_ctx=self.settings.ollama_vision_num_ctx,
            max_side=self.settings.ollama_vision_max_side,
            stream=self.settings.ollama_stream,
            think=self.settings.ollama_think,
            retries=self.settings.llm_retries,
        )
        self.model = self.client.model
        # (path, mtime) -> preprocessed image, reused by the repair attempts.
        self._image_cache: tuple[tuple[str, float], object] | None = None

    def is_available(self) -> bool:
        return self.client.is_available()

    def has_model(self) -> bool:
        return self.client.has_model()

    def _image(self, image_path: str):
        try:
            key = (os.path.abspath(image_path), os.path.getmtime(image_path))
        except OSError:
            key = None
        if key is not None and self._image_cache is not None:
            cached_key, cached_image = self._image_cache
            if cached_key == key:
                return cached_image
        preprocess = PreprocessConfig.from_env(
            enabled=self.settings.ocr_preprocess,
            profile=self.settings.ocr_preprocess_profile,
        )
        image = preprocess_image(image_path, preprocess)
        if key is not None:
            self._image_cache = (key, image)
        return image

    def run(
        self,
        image_path: str,
        context: MathContext,
        feedback: list[str] | None = None,
        *,
        allow_open_concepts: bool = True,
        allow_all_open_concepts: bool = False,
    ) -> str:
        return self.client.chat(
            vision_messages(
                context,
                feedback,
                allow_open_concepts=allow_open_concepts,
                allow_all_open_concepts=allow_all_open_concepts,
                allowed_open_concepts=self.settings.allowed_open_concepts,
            ),
            self._image(image_path),
            temperature=self.settings.temperature,
            num_predict=self.settings.ollama_vision_num_predict,
        )
