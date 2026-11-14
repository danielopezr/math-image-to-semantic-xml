"""Runtime configuration for the local pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    """Settings shared by the Ollama client and the repair loop."""

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_vision_model: str = "qwen3-vl:4b-instruct"
    ollama_vision_num_predict: int = 1024
    ollama_vision_num_ctx: int = 4096
    ollama_vision_max_side: int = 1600
    ollama_timeout: float = 180.0
    ollama_stream: bool = False
    temperature: float = 0.0
    num_predict: int = 1600
    num_ctx: int = 8192
    max_repair_attempts: int = 2
    allow_open_concepts: bool = True
    allow_all_open_concepts: bool = False
    allowed_open_concepts: tuple[str, ...] = (
        "gamma-function",
        "beta-function",
        "error-function",
    )
    ocr_backend: str = "auto"
    direct_vision: bool = False
    ocr_preprocess: bool = True
    ocr_preprocess_profile: str = "safe"
    llm_backend: str = "ollama"
    openai_base_url: str = "http://localhost:8000/v1"
    openai_api_key: str = "not-needed"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_base_url),
            ollama_model=os.getenv("OLLAMA_MODEL", cls.ollama_model),
            ollama_vision_model=os.getenv(
                "IMAGE_TO_MATHML_OLLAMA_VISION_MODEL", cls.ollama_vision_model
            ),
            ollama_vision_num_predict=_env_int(
                "IMAGE_TO_MATHML_OLLAMA_VISION_NUM_PREDICT",
                cls.ollama_vision_num_predict,
            ),
            ollama_vision_num_ctx=_env_int(
                "IMAGE_TO_MATHML_OLLAMA_VISION_NUM_CTX",
                cls.ollama_vision_num_ctx,
            ),
            ollama_vision_max_side=_env_int(
                "IMAGE_TO_MATHML_OLLAMA_VISION_MAX_SIDE",
                cls.ollama_vision_max_side,
            ),
            ollama_timeout=_env_float("OLLAMA_TIMEOUT", cls.ollama_timeout),
            ollama_stream=_env_bool("OLLAMA_STREAM", cls.ollama_stream),
            temperature=_env_float("OLLAMA_TEMPERATURE", cls.temperature),
            num_predict=_env_int("OLLAMA_NUM_PREDICT", cls.num_predict),
            num_ctx=_env_int("OLLAMA_NUM_CTX", cls.num_ctx),
            max_repair_attempts=_env_int(
                "IMAGE_TO_MATHML_MAX_REPAIRS", cls.max_repair_attempts
            ),
            allow_open_concepts=_env_bool(
                "IMAGE_TO_MATHML_ALLOW_OPEN_CONCEPTS", cls.allow_open_concepts
            ),
            allow_all_open_concepts=_env_bool(
                "IMAGE_TO_MATHML_ALLOW_ALL_OPEN_CONCEPTS",
                cls.allow_all_open_concepts,
            ),
            allowed_open_concepts=_env_csv(
                "IMAGE_TO_MATHML_ALLOWED_OPEN_CONCEPTS",
                cls.allowed_open_concepts,
            ),
            ocr_backend=os.getenv("IMAGE_TO_MATHML_OCR", cls.ocr_backend),
            direct_vision=_env_bool("IMAGE_TO_MATHML_DIRECT_VISION", cls.direct_vision),
            ocr_preprocess=_env_bool("IMAGE_TO_MATHML_PREPROCESS", cls.ocr_preprocess),
            ocr_preprocess_profile=os.getenv(
                "IMAGE_TO_MATHML_PREPROCESS_PROFILE",
                cls.ocr_preprocess_profile,
            ),
            llm_backend=os.getenv("IMAGE_TO_MATHML_LLM_BACKEND", cls.llm_backend),
            openai_base_url=os.getenv("OPENAI_BASE_URL", cls.openai_base_url),
            openai_api_key=os.getenv("OPENAI_API_KEY", cls.openai_api_key),
        )
