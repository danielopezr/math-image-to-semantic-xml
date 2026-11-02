"""OCR interface for equation images."""

from __future__ import annotations

from typing import Protocol


class OCRError(RuntimeError):
    """Raised when the image-to-LaTeX backend is unavailable or fails."""


class OCREngine(Protocol):
    name: str

    def image_to_latex(self, image_path: str) -> str:
        """Return a LaTeX representation of the equation in ``image_path``."""
