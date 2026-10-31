"""Contratos de datos entre etapas del pipeline.

Todos los modelos usan Pydantic v2. Cada etapa devuelve un resultado con
``status``, ``errors`` y ``warnings``; las etapas no propagan excepciones.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StageStatus(str, Enum):
    OK = "OK"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class StageIssue(BaseModel):
    """Error o advertencia asociado a una etapa."""

    model_config = ConfigDict(frozen=True)

    stage: str
    code: str
    message: str


# --------------------------------------------------------------------------- OCR
class OcrResult(BaseModel):
    image_ref: str = Field(description="Ruta o nombre de la imagen; no se incrusta su contenido.")
    status: StageStatus
    latex_raw: Optional[str] = Field(
        default=None, description="Salida original del OCR. Nunca se modifica (auditoria)."
    )
    latex_reviewed: Optional[str] = Field(
        default=None, description="LaTeX corregido manualmente, si hubo revision."
    )
    attempt: int = 1
    engine: str = "pix2tex"
    engine_version: Optional[str] = None
    reviewed_by_human: bool = False
    errors: list[StageIssue] = Field(default_factory=list)
    warnings: list[StageIssue] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def latex(self) -> Optional[str]:
        """LaTeX que debe usar la etapa siguiente: el revisado si existe."""
        return self.latex_reviewed if self.latex_reviewed is not None else self.latex_raw


# ------------------------------------------------------------ Semantica (salida LLM)
NodeKind = Literal[
    "equation", "inequality", "expression", "relation",
    "variable", "constant", "number", "operator",
    "fraction", "power", "root", "function", "function_application",
    "sum", "product", "integral", "limit", "derivative",
    "subscript", "group", "unknown",
]

Evidence = Literal["directly_supported", "context_dependent"]


class SemanticNode(BaseModel):
    """Nodo recursivo: permite expresiones anidadas, no solo listas planas."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: NodeKind
    symbol: Optional[str] = Field(default=None, description="Simbolo o valor literal si es hoja.")
    role: Optional[str] = None
    evidence: Evidence = "directly_supported"
    children: list["SemanticNode"] = Field(default_factory=list)


class SemanticRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = Field(description="p. ej. equality, less_than, divides, applies_function")
    left: str = Field(description="id de un SemanticNode")
    right: str = Field(description="id de un SemanticNode")
    evidence: Evidence = "directly_supported"


class SemanticOutput(BaseModel):
    """Esquema que debe cumplir la salida del LLM (se exporta a JSON Schema)."""

    model_config = ConfigDict(extra="forbid")

    expression_type: str
    root: SemanticNode
    relations: list[SemanticRelation] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    requires_review: bool
    confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Solo si existe una medida fiable; en caso contrario null.",
    )


SemanticNode.model_rebuild()


# ------------------------------------------------------------------- Conversion
class ConversionErrorKind(str, Enum):
    NONE = "NONE"
    LATEX_SYNTAX = "LATEX_SYNTAX"  # el LaTeX de entrada no es valido
    CONVERSION = "CONVERSION"      # fallo del conversor o MathML resultante invalido


class ConversionResult(BaseModel):
    latex: str
    status: StageStatus
    mathml: Optional[str] = None
    mathml_kind: Literal["presentation"] = "presentation"
    error_kind: ConversionErrorKind = ConversionErrorKind.NONE
    converter: str = "latex2mathml"
    converter_version: Optional[str] = None
    errors: list[StageIssue] = Field(default_factory=list)
    warnings: list[StageIssue] = Field(default_factory=list)


class SemanticResult(BaseModel):
    status: StageStatus
    output: Optional[SemanticOutput] = None
    raw_responses: list[str] = Field(
        default_factory=list, description="Respuestas originales del modelo, una por intento (auditoria)."
    )
    provider: Optional[str] = None
    model: Optional[str] = None
    attempts: int = 0
    errors: list[StageIssue] = Field(default_factory=list)
    warnings: list[StageIssue] = Field(default_factory=list)


# ------------------------------------------------------------------------ Reglas
class RuleOutcome(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class RuleOrigin(str, Enum):
    W3C_NORMATIVE = "w3c_normative"          # definido por una especificacion del W3C
    PROJECT_HEURISTIC = "project_heuristic"  # criterio propio del proyecto
    LLM_DERIVED = "llm_derived"              # conclusion generada por el LLM


class RuleSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class RuleResult(BaseModel):
    rule_id: str
    rule_version: str
    origin: RuleOrigin
    outcome: RuleOutcome
    severity: RuleSeverity
    message: str
    evidence_refs: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- XML
class XmlResult(BaseModel):
    status: StageStatus
    xml: Optional[bytes] = None
    well_formed: bool = False
    schema_valid: bool = False
    errors: list[StageIssue] = Field(default_factory=list)
    warnings: list[StageIssue] = Field(default_factory=list)


# ---------------------------------------------------------------------- Pipeline
class PipelineReport(BaseModel):
    run_id: str
    created_at: datetime = Field(default_factory=utc_now)
    image_ref: Optional[str] = None
    output_dir: Optional[str] = None
    ocr: Optional[OcrResult] = None
    conversion: Optional[ConversionResult] = None
    semantic: Optional[SemanticResult] = None
    rules: list[RuleResult] = Field(default_factory=list)
    rules_catalog_version: Optional[str] = None
    rules_status: StageStatus = StageStatus.SKIPPED
    xml: Optional[XmlResult] = None
    skipped: dict[str, str] = Field(default_factory=dict, description="Etapa -> motivo por el que no se ejecuto.")
    validated: bool = Field(
        default=False,
        description="True solo si todas las etapas estan OK y el XML supera XSD y coherencia.",
    )
