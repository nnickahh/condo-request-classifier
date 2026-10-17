from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Category(StrEnum):
    MAINTENANCE = "maintenance"
    SECURITY = "security"
    NOISE = "noise"
    CLEANLINESS = "cleanliness"
    FACILITIES = "facilities"
    PARKING = "parking"
    BILLING = "billing"
    RENOVATION = "renovation"
    GENERAL = "general_enquiry"
    OTHER = "other"


class Priority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EMERGENCY = "emergency"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @model_validator(mode="before")
    @classmethod
    def _coerce_reloaded_models(cls, data: Any) -> Any:
        if hasattr(data, "model_dump") and not isinstance(data, cls):
            return data.model_dump()
        if isinstance(data, dict):
            normalized: dict[str, Any] = {}
            for k, v in data.items():
                if hasattr(v, "model_dump"):
                    normalized[k] = v.model_dump()
                elif isinstance(v, dict):
                    normalized[k] = {
                        dk: (dv.model_dump() if hasattr(dv, "model_dump") else dv)
                        for dk, dv in v.items()
                    }
                else:
                    normalized[k] = v
            return normalized
        return data



class ResidentRequest(StrictModel):
    """Input state passed into the Jev System-One triage pipeline."""

    text: str = Field(min_length=3, max_length=4000, description="Raw message sent by the resident.")
    channel: Literal[
        "Web",
        "WhatsApp",
        "Email",
        "Phone transcript",
        "Resident App",
        "Intercom / Call Log",
        "Telegram",
        "REST API",
        "External CMMS",
    ] = "Web"
    location: str | None = Field(default=None, max_length=120, description="Optional unit number, block, or common area.")

    @field_validator("channel", mode="before")
    @classmethod
    def normalize_channel(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            return "Web"
        val = value.strip()
        allowed = {
            "Web",
            "WhatsApp",
            "Email",
            "Phone transcript",
            "Resident App",
            "Intercom / Call Log",
            "Telegram",
            "REST API",
            "External CMMS",
        }
        if val in allowed:
            return val
        if "api" in val.lower():
            return "REST API"
        if "cmms" in val.lower():
            return "External CMMS"
        if "app" in val.lower():
            return "Resident App"
        return "Web"

    @field_validator("text")
    @classmethod
    def validate_meaningful_text(cls, value: str) -> str:
        if not any(ch.isalnum() for ch in value):
            raise ValueError("Request text must include alphanumeric words.")
        return value

    @field_validator("location")
    @classmethod
    def normalize_location(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return value.strip()


class JevChoiceResult(StrictModel):
    """Represents a Jev `Choice` primitive output: selected option, concentration confidence, and full distribution."""

    question_key: str
    instructions: str
    choice: Category
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]


class JevScoreResult(StrictModel):
    """Represents a Jev `Score` primitive output: continuous rubric score [0, 1], mapped level, confidence, and distribution."""

    question_key: str
    instructions: str
    score: float = Field(ge=0.0, le=1.0)
    level: Priority
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]


class JevNoulResult(StrictModel):
    """Represents a Jev `Noul` primitive output: binary yes/no question returning P(yes) in [0, 1]."""

    question_key: str
    instructions: str
    noul: float = Field(ge=0.0, le=1.0, description="Probability in [0, 1] that the answer to the binary question is YES.")


class JevSystemOneBundle(StrictModel):
    """Structured container mirroring the `typesafe_sdk` System-One response (`choices`, `scores`, `nouls`)."""

    state_summary: str
    choices: dict[str, JevChoiceResult]
    scores: dict[str, JevScoreResult]
    nouls: dict[str, JevNoulResult]


class LLMJevEmulationPayload(StrictModel):
    """Strict JSON schema validated when an Ollama/local LLM emulates Jev primitives."""

    category: Category
    priority: Priority
    choice_confidence: float = Field(ge=0.0, le=1.0)
    urgency_score: float = Field(ge=0.0, le=1.0)
    noul_life_safety_hazard: float = Field(ge=0.0, le=1.0)
    noul_multi_intent_request: float = Field(ge=0.0, le=1.0)
    noul_adversarial_or_sensitive: float = Field(ge=0.0, le=1.0)
    noul_requires_human_triage: float = Field(ge=0.0, le=1.0)
    evidence_quotes: list[str] = Field(default_factory=list, max_length=4)
    rationale_note: str = Field(min_length=5, max_length=320)


class TriagePrediction(StrictModel):
    """Raw backend output before deterministic LangGraph policy enforcement."""

    category: Category
    priority: Priority
    choice_confidence: float = Field(ge=0.0, le=1.0)
    urgency_score: float = Field(ge=0.0, le=1.0)
    category_probabilities: dict[str, float]
    priority_probabilities: dict[str, float]
    nouls: dict[str, float]
    jev_bundle: JevSystemOneBundle
    extracted_location: str | None = None
    evidence_quotes: list[str] = Field(default_factory=list)
    rationale_note: str | None = None
    backend: str
    model: str
    calibration_status: Literal[
        "calibrated_jev_api",
        "platt_calibrated_local",
        "uncalibrated_llm_self_report",
        "deterministic_safety_rule",
    ]


class Classification(StrictModel):
    """Final validated triage decision emitted by the LangGraph workflow."""

    category: Category
    category_title: str
    assigned_team: str
    sla_target: str
    priority: Priority
    choice_confidence: float = Field(ge=0.0, le=1.0)
    urgency_score: float = Field(ge=0.0, le=1.0)
    calibration_status: Literal[
        "calibrated_jev_api",
        "platt_calibrated_local",
        "uncalibrated_llm_self_report",
        "deterministic_safety_rule",
    ]
    category_probabilities: dict[str, float]
    priority_probabilities: dict[str, float]
    nouls: dict[str, float]
    jev_bundle: JevSystemOneBundle
    needs_human_review: bool
    review_reasons: list[str]
    missing_details: list[str] = Field(default_factory=list)
    extracted_location: str | None = None
    evidence_quotes: list[str] = Field(default_factory=list)
    rationale_note: str | None = None
    backend: str
    model: str
    latency_ms: float = Field(ge=0.0)
