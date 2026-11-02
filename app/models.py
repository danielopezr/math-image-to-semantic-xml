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
# Los intents NO se enumeran aqui: viven en app/intent_vocabulary.json (fuente unica).
# El tipo de nodo si es cerrado porque es estructura del modelo, no vocabulario.
NodeType = Literal["element", "operation", "relation", "structure", "function", "calculus", "unresolved"]
Evidence = Literal["directly_supported", "context_dependent"]
LinkType = Literal["binds", "same_entity"]


class Operand(BaseModel):
    """Referencia ordenada a un nodo hijo, con el rol que desempena para el padre."""

    model_config = ConfigDict(extra="forbid")

    role: str = Field(description="Rol definido por el vocabulario (p. ej. base, exponent, numerator).")
    node_id: str


class SemanticNode(BaseModel):
    """Nodo del arbol semantico (representacion plana: los hijos se referencian por id).

    * ``symbol``: simbolo tal como se escribe en la formula (si existe).
    * ``intent``: funcion matematica del elemento dentro de la expresion (vocabulario controlado).
    * ``operands``: hijos ordenados con su rol (relacion padre-hijo).
    * Un nodo ambiguo no tiene ``intent`` definitivo: lista ``candidate_intents`` y exige revision.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    node_type: NodeType
    intent: Optional[str] = None
    symbol: Optional[str] = None
    operands: list[Operand] = Field(default_factory=list)
    properties: dict[str, str] = Field(default_factory=dict)
    implicit: bool = Field(default=False, description="Operador no escrito (multiplicacion o aplicacion implicita).")
    evaluation_order: Optional[int] = Field(
        default=None, description="Lo calcula el proyecto; el valor que envie el LLM se descarta."
    )
    evidence: Evidence = "directly_supported"
    ambiguous: bool = False
    candidate_intents: list[str] = Field(default_factory=list)
    justification: Optional[str] = None
    review_required: bool = False


class SemanticLink(BaseModel):
    """Relacion no jerarquica entre nodos (p. ej. un integral que liga a su variable)."""

    model_config = ConfigDict(extra="forbid")

    type: LinkType
    source: str
    target: str
    evidence: Evidence = "directly_supported"


class SemanticOutput(BaseModel):
    """Esquema que debe cumplir la salida del LLM (se exporta a JSON Schema).

    No contiene campos para LaTeX ni MathML: el LLM no puede devolverlos modificados.
    """

    model_config = ConfigDict(extra="forbid")

    vocabulary_version: str
    root_id: str
    nodes: list[SemanticNode]
    links: list[SemanticLink] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    review_required: bool
    confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="La confianza declarada por un LLM no es fiable: el proyecto la descarta.",
    )


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
    input_latex_sha256: Optional[str] = Field(default=None, description="Huella del LaTeX enviado al modelo.")
    input_mathml_sha256: Optional[str] = Field(default=None, description="Huella del MathML enviado al modelo.")
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


# ------------------------------------------------------------ MathCAT, braille y audio
class MathCATConfiguration(BaseModel):
    """Configuracion de MathCAT. Los valores por defecto viven AQUI y en nigun otro sitio."""

    rules_dir: Optional[str] = Field(default=None, description="Carpeta Rules de MathCAT.")
    braille_code: str = Field(default="CMU", description="nemeth | ueb_technical | cmu")
    speech_language: str = Field(default="es", description="Codigo de idioma de la lectura (p. ej. es, en).")
    speech_style: str = "ClearSpeak"
    verbosity: str = "Medium"
    python_module: Optional[str] = Field(default=None, description="Nombre del modulo nativo, si no es el habitual.")


class _StageOutput(BaseModel):
    """Campos comunes a las salidas de las etapas nuevas."""

    status: StageStatus
    errors: list[StageIssue] = Field(default_factory=list)
    warnings: list[StageIssue] = Field(default_factory=list)
    review_required: bool = False


class BrailleOutput(_StageOutput):
    """Transcripcion braille en Unicode (celdas U+2800..U+28FF). No es un archivo BRF ni salida para linea braille."""

    braille_code: Optional[str] = Field(default=None, description="Codigo solicitado (nemeth, ueb_technical, cmu).")
    mathcat_code: Optional[str] = Field(default=None, description="Valor de BrailleCode enviado a MathCAT.")
    language: Optional[str] = None
    text_unicode: Optional[str] = None
    text_unicode_reviewed: Optional[str] = None
    reviewed_by_human: bool = False
    mathcat_version: Optional[str] = None

    @property
    def text(self) -> Optional[str]:
        return self.text_unicode_reviewed if self.text_unicode_reviewed is not None else self.text_unicode


class SpeechOutput(_StageOutput):
    """Texto de lectura matematica generado por MathCAT (NO es audio)."""

    language: Optional[str] = None
    style: Optional[str] = None
    verbosity: Optional[str] = None
    text: Optional[str] = None
    mathcat_version: Optional[str] = None


class AudioOutput(_StageOutput):
    """Resultado de la sintesis de voz. Solo es OK si el archivo existe y se ha verificado."""

    provider: Optional[str] = None
    language: Optional[str] = None
    audio_format: Optional[str] = None
    file_name: Optional[str] = Field(default=None, description="Solo el nombre del archivo, sin ruta personal.")
    file_path: Optional[str] = Field(default=None, description="Ruta local; no se escribe en el XML.")
    size_bytes: Optional[int] = None
    voice: Optional[str] = None


class EnrichmentResult(_StageOutput):
    """Enriquecimiento opcional del MathML con atributos intent (desactivado por defecto)."""

    enabled: bool = False
    mathml: Optional[str] = Field(default=None, description="MathML enriquecido; None si no se aplico.")
    applied: list[str] = Field(default_factory=list)
    mathcat_checked: bool = False


class StageResult(BaseModel):
    """Resumen uniforme de una etapa del pipeline."""

    stage: str
    status: StageStatus
    required: bool = True
    review_required: bool = False
    skipped_reason: Optional[str] = None
    error_count: int = 0
    warning_count: int = 0


# ---------------------------------------------------------------------- Pipeline
class PipelineResult(BaseModel):
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
    enrichment: Optional[EnrichmentResult] = None
    braille: Optional[BrailleOutput] = None
    speech: Optional[SpeechOutput] = None
    audio: Optional[AudioOutput] = None
    stages: list[StageResult] = Field(default_factory=list)
    xml: Optional[XmlResult] = None
    skipped: dict[str, str] = Field(default_factory=dict, description="Etapa -> motivo por el que no se ejecuto.")
    validated: bool = Field(
        default=False,
        description="True solo si todas las etapas estan OK y el XML supera XSD y coherencia.",
    )


PipelineReport = PipelineResult  # alias de compatibilidad con la version anterior
