"""Image to accessible MathML with W3C intent annotations.

This package implements the equation-recognition stage of a larger
accessibility pipeline.  It turns an equation image (or an already recognised
LaTeX string) plus textbook context into a single presentation MathML
expression annotated with W3C ``intent`` attributes.
"""

from .config import Settings
from .models import (
    EquationRequest,
    Flow,
    MathContext,
    PipelineResult,
    ValidationIssue,
    ValidationReport,
)
from .pipeline import EquationPipeline, compare_flows

__all__ = [
    "EquationPipeline",
    "EquationRequest",
    "Flow",
    "MathContext",
    "PipelineResult",
    "Settings",
    "ValidationIssue",
    "ValidationReport",
    "compare_flows",
]

__version__ = "0.1.0"
