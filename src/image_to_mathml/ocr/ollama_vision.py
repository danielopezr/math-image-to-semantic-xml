"""Ollama vision OCR backend (Qwen3-VL and other multimodal models)."""

from __future__ import annotations

import re

from ..llm.base import LLMError
from ..llm.ollama_vision import OllamaVisionClient
from .base import OCRError
from .preprocess import PreprocessConfig, preprocess_image

_SYSTEM_PROMPT = (
    "You are a precise mathematical OCR engine. The user sends an image of one "
    "equation. Return only the LaTeX source of that equation.\n"
    "Rules:\n"
    "- Do not use Markdown, code fences, dollar signs, \\[...\\], \\(...\\), or "
    "equation environments.\n"
    "- Output one LaTeX math expression and preserve fractions, roots, "
    "subscripts, superscripts, matrices, accents and delimiters.\n"
    "- Use standard LaTeX commands such as \\frac, \\sqrt, \\sum, \\int, "
    "\\left and \\right.\n"
    "- Do not add explanations, labels, equation numbers or surrounding prose.\n"
    "- If the image is not an equation or is unreadable, return an empty string."
)

_FENCE_RE = re.compile(r"```(?:latex|tex)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_LABEL_RE = re.compile(r"^\s*(?:latex|result)\s*[:=]\s*", re.IGNORECASE)
_DELIMITER_RE = re.compile(
    r"^\s*(?:\$\$?|\\\[|\\\()\s*(.*?)\s*(?:\$\$?|\\\]|\\\))\s*$",
    re.DOTALL,
)


def _extract_latex(text: str) -> str:
    cleaned = (text or "").strip()
    fenced = _FENCE_RE.search(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()
    cleaned = _LABEL_RE.sub("", cleaned).strip()
    delimited = _DELIMITER_RE.match(cleaned)
    if delimited:
        cleaned = delimited.group(1).strip()
    return cleaned


class OllamaVisionOCR:
    name = "ollama_vision"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        preprocess: PreprocessConfig | None = None,
        stream: bool | None = None,
    ) -> None:
        self.client = OllamaVisionClient(
            model=model, base_url=base_url, timeout=timeout, stream=stream
        )
        self.model = self.client.model
        self.base_url = self.client.base_url
        self.num_ctx = self.client.num_ctx
        self.stream = self.client.stream
        self.preprocess = preprocess or PreprocessConfig.from_env()

    def image_to_latex(self, image_path: str) -> str:
        image = preprocess_image(image_path, self.preprocess)
        try:
            content = self.client.chat(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": "Extract the equation from this image as LaTeX.",
                    },
                ],
                image,
            )
        except LLMError as exc:
            raise OCRError(str(exc)) from exc

        latex = _extract_latex(content)
        if not latex:
            raise OCRError(
                f"Ollama vision model {self.model!r} returned an empty LaTeX string"
            )
        return latex
