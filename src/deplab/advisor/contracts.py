from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class AnalysisRequest:
    requirements_text: str
    question: str
    python_version: str
    platform: str = "linux-x86_64"
    conversation_context: tuple["ConversationTurn", ...] = ()


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    content: str


@dataclass(frozen=True)
class RequirementPin:
    name: str
    version: str | None
    raw: str


@dataclass(frozen=True)
class ParsedIntent:
    requirements: list[RequirementPin]
    target_package: str
    requested_version: str
    action: str = "upgrade"
    assumptions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConstraintConflict:
    evidence_type: str
    certainty: str
    declaring_package: str
    declaring_version: str
    dependency_package: str
    dependency_version: str
    requirement: str
    blocking_specifiers: list[str]


@dataclass(frozen=True)
class ResolverCheck:
    environment: dict[str, str]
    status: str
    resolvable: bool | None
    duration_seconds: float
    cache_hit: bool
    explanation: str
    evidence_type: str = "uv_resolution"
    installed: bool = False


@dataclass(frozen=True)
class PairRisk:
    family: str
    target_package: str
    target_version: str
    related_package: str
    related_version: str
    risk_score: float
    predicted_failure: bool
    logistic_probability: float
    resolution_probability: float
    post_install_probability: float
    evidence_source: str
    model_id: str
    likely_stage: str
    published_constraints_allow: bool
    risk_detected: bool
    evidence_type: str
    constraint_conflicts: list[ConstraintConflict]
    import_probability: float | None = None
    smoke_probability: float | None = None
    model_coverage: str = "covered"


@dataclass(frozen=True)
class SuggestedChange:
    package: str
    from_version: str
    to_version: str


@dataclass(frozen=True)
class Alternative:
    target_version: str
    keeps_requested_target: bool
    category: str
    changes: list[SuggestedChange]
    maximum_risk_score: float
    predicted_failure: bool
    covered_pairs: int
    reason: str
    resolver_status: str = "not_checked"
    resolver_duration_seconds: float | None = None
    post_install_risk: float | None = None
    verification_status: str = "not_checked"


@dataclass
class AdvisoryResult:
    status: str
    summary: str
    parsed_intent: ParsedIntent | None = None
    pair_risks: list[PairRisk] = field(default_factory=list)
    resolver_checks: list[ResolverCheck] = field(default_factory=list)
    alternatives: list[Alternative] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    answer: str = ""
    model_scope: str = "prediction_only"
    verification_status: str = "not_runtime_verified"
    candidates_considered: int = 0
    candidates_resolver_checked: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
