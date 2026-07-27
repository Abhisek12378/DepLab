from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

from .contracts import (
    AdvisoryResult,
    Alternative,
    AnalysisRequest,
    ConstraintConflict,
    PairRisk,
    ParsedIntent,
    ResolverCheck,
    SuggestedChange,
)
from .model_runtime import FeatureTable
from .ports import AnswerComposer, IntentParser
from .production_runtime import (
    ModernBertPostInstallModel,
    PostInstallRisk,
    StructuredRankingModel,
)
from .resolver import ResolverResult, ResolverVerifier
from .validation import canonical_name, validate_intent


MAXIMUM_CANDIDATES = 50
MAXIMUM_RESOLVER_CHECKS = 5
MAXIMUM_RECOMMENDATIONS = 3
OPTIONS_PER_RELATED_PACKAGE = 3


@dataclass(frozen=True)
class Candidate:
    environment: dict[str, str]
    target_version: str
    category: str
    structured_risk: float
    changes: list[SuggestedChange]


@dataclass(frozen=True)
class EnvironmentAssessment:
    pair_risks: list[PairRisk]
    warnings: list[str]
    maximum_post_install_risk: float | None
    post_install_covered_pairs: int
    post_install_risk_detected: bool
    deterministic_conflicts: bool


class CascadeAdvisorService:
    """Resolver-authoritative advisor with post-install prediction."""

    def __init__(
        self,
        parser: IntentParser,
        features: FeatureTable,
        structured_model: StructuredRankingModel,
        post_install_model: ModernBertPostInstallModel,
        resolver: ResolverVerifier,
        composer: AnswerComposer | None = None,
    ) -> None:
        self.parser = parser
        self.features = features
        self.structured_model = structured_model
        self.post_install_model = post_install_model
        self.resolver = resolver
        self.composer = composer

    def analyze(self, request: AnalysisRequest) -> AdvisoryResult:
        parsed = self._parse_and_validate(request)
        if isinstance(parsed, AdvisoryResult):
            return parsed
        intent = parsed
        current = self._exact_environment(intent)
        proposed = {**current, canonical_name(intent.target_package): intent.requested_version}
        direct = self._assess_environment(
            proposed,
            request.python_version,
            resolver_succeeded=False,
        )
        resolver_result = self._verify_proposed(
            proposed,
            request,
            direct.deterministic_conflicts,
        )
        assessment = (
            self._assess_environment(
                proposed,
                request.python_version,
                resolver_succeeded=True,
            )
            if resolver_result.resolvable is True
            else direct
        )
        risk_found = (
            assessment.deterministic_conflicts
            or resolver_result.resolvable is False
            or assessment.post_install_risk_detected
        )
        alternatives, checks, considered = self._recommendations(
            intent,
            request,
            current,
        ) if risk_found else ([], [], 0)
        result = self._result(
            intent,
            resolver_result,
            proposed,
            assessment,
            alternatives,
            checks,
            considered,
        )
        payload = result.to_dict()
        result.answer = self.composer.compose(payload) if self.composer else ""
        if not result.answer:
            result.answer = self._fallback_answer(result)
        return result

    def _parse_and_validate(
        self,
        request: AnalysisRequest,
    ) -> ParsedIntent | AdvisoryResult:
        try:
            intent = self.parser.parse(request)
        except Exception as exc:
            return AdvisoryResult(
                status="input_error",
                summary="DepLab could not understand the request.",
                errors=[str(exc)],
            )
        errors = validate_intent(request, intent)
        errors.extend(self._exact_pin_errors(intent))
        if errors:
            return AdvisoryResult(
                status="input_error",
                summary="Please correct the request before compatibility analysis.",
                parsed_intent=intent,
                errors=errors,
            )
        return intent

    @staticmethod
    def _exact_pin_errors(intent: ParsedIntent) -> list[str]:
        errors = []
        for requirement in intent.requirements:
            if not requirement.version:
                errors.append(
                    f"{requirement.name} must use an exact == version for resolver analysis."
                )
                continue
            try:
                Version(requirement.version)
            except InvalidVersion:
                errors.append(
                    f"{requirement.name} has an invalid exact version."
                )
        try:
            Version(intent.requested_version)
        except InvalidVersion:
            errors.append("The requested target version is invalid.")
        return errors

    @staticmethod
    def _exact_environment(intent: ParsedIntent) -> dict[str, str]:
        return {
            canonical_name(requirement.name): str(requirement.version)
            for requirement in intent.requirements
            if requirement.version
        }

    def _verify_proposed(
        self,
        environment: dict[str, str],
        request: AnalysisRequest,
        deterministic_conflicts: bool,
    ) -> ResolverResult:
        if deterministic_conflicts:
            return ResolverResult(
                status="direct_constraint_blocked",
                resolvable=False,
                duration_seconds=0.0,
                explanation=(
                    "Published direct constraints already prove that the proposed "
                    "environment cannot resolve."
                ),
            )
        return self.resolver.verify(
            environment,
            request.python_version,
            request.platform,
        )

    def _assess_environment(
        self,
        environment: dict[str, str],
        python_version: str,
        resolver_succeeded: bool,
    ) -> EnvironmentAssessment:
        risks: list[PairRisk] = []
        warnings: list[str] = []
        post_scores: list[float] = []
        post_risk = False
        deterministic = False
        for left, right in combinations(sorted(environment), 2):
            if not self.features.has_family(left, right, python_version):
                continue
            row = self.features.find(
                left,
                environment[left],
                right,
                environment[right],
                python_version,
            )
            if row is None:
                warnings.append(
                    f"Post-install model coverage is unavailable for "
                    f"{left}=={environment[left]} with {right}=={environment[right]} "
                    f"on Python {python_version}."
                )
                continue
            constraints_allow = self.features.published_constraints_allow(row)
            conflicts = self.features.constraint_conflict_records(row)
            structured = self.structured_model.score(row)
            post = (
                self.post_install_model.score(row)
                if constraints_allow and resolver_succeeded
                else None
            )
            if constraints_allow and resolver_succeeded and post is None:
                warnings.append(
                    "Post-install model embeddings are unavailable for "
                    f"{left}=={environment[left]} with "
                    f"{right}=={environment[right]} on Python {python_version}."
                )
            deterministic = deterministic or not constraints_allow
            if post is not None:
                post_scores.append(post.combined_probability)
                post_risk = post_risk or post.risk_detected
            risks.append(
                self._pair_risk(
                    row,
                    left,
                    environment[left],
                    right,
                    environment[right],
                    structured.probability,
                    constraints_allow,
                    conflicts,
                    post,
                )
            )
        return EnvironmentAssessment(
            pair_risks=risks,
            warnings=warnings,
            maximum_post_install_risk=max(post_scores) if post_scores else None,
            post_install_covered_pairs=len(post_scores),
            post_install_risk_detected=post_risk,
            deterministic_conflicts=deterministic,
        )

    def _pair_risk(
        self,
        row: dict[str, str],
        left: str,
        left_version: str,
        right: str,
        right_version: str,
        structured_probability: float,
        constraints_allow: bool,
        conflicts: list[dict[str, Any]],
        post: PostInstallRisk | None,
    ) -> PairRisk:
        if not constraints_allow:
            evidence_type = "published_constraint_conflict"
            risk_score = 1.0
            likely_stage = "dependency resolution"
            model_id = self.structured_model.model_id
        elif post is None:
            evidence_type = "structured_ranking_signal"
            risk_score = structured_probability
            likely_stage = "not assessed"
            model_id = self.structured_model.model_id
        else:
            evidence_type = "post_install_prediction"
            risk_score = post.combined_probability
            likely_stage = (
                "import"
                if post.import_probability >= post.smoke_probability
                else "smoke test"
            )
            model_id = self.post_install_model.model_id
        return PairRisk(
            family=row["family"],
            target_package=left,
            target_version=left_version,
            related_package=right,
            related_version=right_version,
            risk_score=risk_score,
            predicted_failure=bool(post and post.risk_detected),
            logistic_probability=structured_probability,
            resolution_probability=structured_probability,
            post_install_probability=(
                post.combined_probability if post is not None else 0.0
            ),
            import_probability=(
                post.import_probability if post is not None else None
            ),
            smoke_probability=(
                post.smoke_probability if post is not None else None
            ),
            evidence_source=(
                "published metadata and direct requirements"
                if not constraints_allow
                else "frozen structured features and ModernBERT release embeddings"
            ),
            model_id=model_id,
            likely_stage=likely_stage,
            published_constraints_allow=constraints_allow,
            risk_detected=not constraints_allow or bool(post and post.risk_detected),
            evidence_type=evidence_type,
            constraint_conflicts=[
                ConstraintConflict(
                    evidence_type="published_constraint_conflict",
                    certainty="deterministic",
                    **record,
                )
                for record in conflicts
            ],
            model_coverage="covered" if post is not None else "not_scored",
        )

    def _recommendations(
        self,
        intent: ParsedIntent,
        request: AnalysisRequest,
        current: dict[str, str],
    ) -> tuple[list[Alternative], list[ResolverCheck], int]:
        candidates = self._generate_candidates(intent, request.python_version, current)
        accepted: list[Alternative] = []
        checks: list[ResolverCheck] = []
        for candidate in candidates[:MAXIMUM_RESOLVER_CHECKS]:
            assessment = self._assess_environment(
                candidate.environment,
                request.python_version,
                resolver_succeeded=False,
            )
            if assessment.deterministic_conflicts:
                continue
            resolver = self.resolver.verify(
                candidate.environment,
                request.python_version,
                request.platform,
            )
            checks.append(_resolver_check(candidate.environment, resolver))
            if resolver.resolvable is not True:
                continue
            post = self._assess_environment(
                candidate.environment,
                request.python_version,
                resolver_succeeded=True,
            )
            if (
                post.post_install_covered_pairs < 1
                or post.post_install_risk_detected
                or post.warnings
            ):
                continue
            accepted.append(
                Alternative(
                    target_version=candidate.target_version,
                    keeps_requested_target=(
                        candidate.target_version == intent.requested_version
                    ),
                    category=candidate.category,
                    changes=candidate.changes,
                    maximum_risk_score=post.maximum_post_install_risk or 0.0,
                    predicted_failure=False,
                    covered_pairs=post.post_install_covered_pairs,
                    reason=(
                        "uv produced a complete dependency lock, and the "
                        "post-install model is below its frozen warning threshold "
                        "for every covered pair. No packages were installed."
                    ),
                    resolver_status=resolver.status,
                    resolver_duration_seconds=resolver.duration_seconds,
                    post_install_risk=post.maximum_post_install_risk,
                    verification_status="resolver_verified",
                )
            )
            if len(accepted) >= MAXIMUM_RECOMMENDATIONS:
                break
        return accepted, checks, len(candidates)

    def _generate_candidates(
        self,
        intent: ParsedIntent,
        python_version: str,
        current: dict[str, str],
    ) -> list[Candidate]:
        target = canonical_name(intent.target_package)
        related = [
            name
            for name in sorted(current)
            if name != target and self.features.has_family(
                target, name, python_version
            )
        ]
        target_versions = self._target_versions(
            target,
            related,
            python_version,
        )
        candidates: list[Candidate] = []
        for target_version in target_versions:
            states: list[tuple[dict[str, str], float]] = [
                ({**current, target: target_version}, 0.0)
            ]
            for package in related:
                states = self._expand_related(
                    states,
                    target,
                    target_version,
                    package,
                    current[package],
                    python_version,
                    current,
                )
                if not states:
                    break
            for environment, risk in states:
                candidates.append(
                    Candidate(
                        environment=environment,
                        target_version=target_version,
                        category=_category(
                            target_version,
                            intent.requested_version,
                            current.get(target),
                        ),
                        structured_risk=risk,
                        changes=_changes(current, environment),
                    )
                )
        unique = {
            tuple(sorted(candidate.environment.items())): candidate
            for candidate in candidates
        }
        return sorted(
            unique.values(),
            key=lambda candidate: (
                _category_priority(candidate.category),
                len(candidate.changes),
                candidate.structured_risk,
                _version_distance(candidate.target_version, intent.requested_version),
            ),
        )[:MAXIMUM_CANDIDATES]

    def _target_versions(
        self,
        target: str,
        related: list[str],
        python_version: str,
    ) -> list[str]:
        if not related:
            return []
        sets = [
            {
                target_version
                for _, target_version, _ in self.features.pair_rows(
                    target,
                    None,
                    package,
                    python_version,
                )
            }
            for package in related
        ]
        versions = set.intersection(*sets) if sets else set()
        return sorted(versions, key=_version_key)

    def _expand_related(
        self,
        states: list[tuple[dict[str, str], float]],
        target: str,
        target_version: str,
        related: str,
        current_version: str,
        python_version: str,
        original_environment: dict[str, str],
    ) -> list[tuple[dict[str, str], float]]:
        options: list[tuple[tuple[Any, ...], str, float]] = []
        for row, _, related_version in self.features.pair_rows(
            target,
            target_version,
            related,
            python_version,
        ):
            if not self.features.published_constraints_allow(row):
                continue
            structured = self.structured_model.score(row).probability
            options.append(
                (
                    (
                        related_version != current_version,
                        _version_distance(related_version, current_version),
                        structured,
                    ),
                    related_version,
                    structured,
                )
            )
        selected = sorted(options, key=lambda item: item[0])[
            :OPTIONS_PER_RELATED_PACKAGE
        ]
        expanded = [
            (
                {**environment, related: version},
                max(risk, option_risk),
            )
            for environment, risk in states
            for _, version, option_risk in selected
        ]
        return sorted(
            expanded,
            key=lambda item: (
                _change_count(item[0], original_environment),
                item[1],
                tuple(sorted(item[0].items())),
            ),
        )[:MAXIMUM_CANDIDATES]

    def _result(
        self,
        intent: ParsedIntent,
        resolver: ResolverResult,
        proposed: dict[str, str],
        assessment: EnvironmentAssessment,
        alternatives: list[Alternative],
        alternative_checks: list[ResolverCheck],
        considered: int,
    ) -> AdvisoryResult:
        proposed_check = _resolver_check(proposed, resolver)
        status = _status(resolver, assessment)
        summary = _summary(resolver, assessment)
        warnings = list(dict.fromkeys(assessment.warnings))
        if resolver.resolvable is None:
            warnings.append(resolver.explanation)
        warnings.extend(
            _recommendation_warnings(
                alternatives,
                alternative_checks,
                considered,
            )
        )
        warnings = list(dict.fromkeys(warnings))
        result = AdvisoryResult(
            status=status,
            summary=summary,
            parsed_intent=intent,
            pair_risks=sorted(
                assessment.pair_risks,
                key=lambda item: item.risk_score,
                reverse=True,
            ),
            resolver_checks=[proposed_check, *alternative_checks],
            alternatives=alternatives,
            warnings=warnings,
            model_scope="resolver_and_post_install_prediction",
            verification_status=(
                "resolver_verified"
                if resolver.resolvable is True
                else "resolver_rejected"
                if resolver.resolvable is False
                else "resolver_unavailable"
            ),
            candidates_considered=considered,
            candidates_resolver_checked=len(alternative_checks) + (
                1 if resolver.status not in {"direct_constraint_blocked"} else 0
            ),
        )
        return result

    @staticmethod
    def _fallback_answer(result: AdvisoryResult) -> str:
        check = result.resolver_checks[0] if result.resolver_checks else None
        if check and check.resolvable is False:
            answer = (
                "The proposed environment will not resolve. "
                f"{check.explanation}"
            )
        elif check and check.resolvable is True:
            risky = [
                risk
                for risk in result.pair_risks
                if risk.evidence_type == "post_install_prediction"
                and risk.risk_detected
            ]
            answer = (
                "uv successfully produced a complete dependency lock. "
                + (
                    f"DepLab predicts post-install risk in {len(risky)} "
                    "covered package pair(s)."
                    if risky
                    else "DepLab did not predict post-install risk in the covered pairs."
                )
            )
        else:
            answer = result.summary
        if result.alternatives:
            changes = ", ".join(
                f"{change.package}=={change.to_version}"
                for change in result.alternatives[0].changes
            )
            answer += f" The best resolver-checked alternative is: {changes}."
        return answer + " No packages were installed or executed."


def _resolver_check(
    environment: dict[str, str],
    result: ResolverResult,
) -> ResolverCheck:
    return ResolverCheck(
        environment=dict(sorted(environment.items())),
        status=result.status,
        resolvable=result.resolvable,
        duration_seconds=result.duration_seconds,
        cache_hit=result.cache_hit,
        explanation=result.explanation,
    )


def _recommendation_warnings(
    alternatives: list[Alternative],
    checks: list[ResolverCheck],
    considered: int,
) -> list[str]:
    if alternatives or considered == 0:
        return []
    unavailable = sum(check.resolvable is None for check in checks)
    rejected = sum(check.resolvable is False for check in checks)
    resolved = sum(check.resolvable is True for check in checks)
    if unavailable and resolved == 0:
        return [
            f"DepLab generated {considered} candidate environments, but uv "
            f"could not verify {unavailable} checked candidate(s). No "
            "recommendation is shown without a successful resolver check."
        ]
    if rejected and resolved == 0:
        return [
            f"DepLab generated {considered} candidate environments, but uv "
            f"rejected all {rejected} checked candidate(s)."
        ]
    if resolved:
        return [
            f"uv resolved {resolved} candidate environment(s), but each was "
            "removed by the frozen post-install risk or coverage policy."
        ]
    return [
        f"DepLab generated {considered} candidate environments, but none "
        "reached the resolver-verification stage."
    ]


def _status(
    resolver: ResolverResult,
    assessment: EnvironmentAssessment,
) -> str:
    if resolver.resolvable is None:
        return "resolver_unavailable"
    if (
        resolver.resolvable is False
        or assessment.deterministic_conflicts
        or assessment.post_install_risk_detected
    ):
        return "risk_found"
    if assessment.post_install_covered_pairs < 1:
        return "coverage_unavailable"
    return "no_risk_predicted"


def _summary(
    resolver: ResolverResult,
    assessment: EnvironmentAssessment,
) -> str:
    if resolver.resolvable is None:
        return "DepLab could not complete the resolver check."
    if resolver.resolvable is False:
        return "The proposed environment cannot be resolved."
    if assessment.post_install_risk_detected:
        count = sum(
            risk.evidence_type == "post_install_prediction"
            and risk.risk_detected
            for risk in assessment.pair_risks
        )
        return (
            "uv resolved the environment, but DepLab predicts post-install "
            f"risk in {count} covered package pair(s)."
        )
    if assessment.post_install_covered_pairs < 1:
        return (
            "uv resolved the environment, but DepLab has no post-install "
            "model coverage for its package pairs."
        )
    return (
        "uv resolved the environment, and DepLab did not predict "
        "post-install failure in the covered package pairs."
    )


def _changes(
    current: dict[str, str],
    candidate: dict[str, str],
) -> list[SuggestedChange]:
    return [
        SuggestedChange(
            package=package,
            from_version=current.get(package, "unpinned"),
            to_version=version,
        )
        for package, version in sorted(candidate.items())
        if current.get(package) != version
    ]


def _change_count(
    candidate: dict[str, str],
    reference: dict[str, str],
) -> int:
    return sum(reference.get(name) != version for name, version in candidate.items())


def _category(candidate: str, requested: str, current: str | None) -> str:
    if candidate == requested:
        return "achieves_requested_change"
    if current and candidate == current:
        return "keeps_current_version"
    if current and _version_key(candidate) < _version_key(current):
        return "downgrade_fallback"
    return "different_upgrade_fallback"


def _category_priority(category: str) -> int:
    return {
        "achieves_requested_change": 0,
        "keeps_current_version": 1,
        "different_upgrade_fallback": 2,
        "downgrade_fallback": 3,
    }[category]


def _version_key(value: str) -> Version:
    try:
        return Version(value)
    except InvalidVersion:
        return Version("0")


def _version_distance(left: str, right: str) -> int:
    a = _release_tuple(left)
    b = _release_tuple(right)
    return (
        abs(a[0] - b[0]) * 1_000_000
        + abs(a[1] - b[1]) * 1_000
        + abs(a[2] - b[2])
    )


def _release_tuple(value: str) -> tuple[int, int, int]:
    try:
        release = Version(value).release
    except InvalidVersion:
        release = ()
    return tuple((list(release) + [0, 0, 0])[:3])  # type: ignore[return-value]


def load_post_install_threshold(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("policy_id") != "deplab-post-install-cascade-v1.0.0":
        raise ValueError("unexpected post-install policy")
    return float(payload["post_install_threshold"])
