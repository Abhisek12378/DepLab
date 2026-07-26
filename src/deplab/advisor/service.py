from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .contracts import (
    AdvisoryResult,
    Alternative,
    AnalysisRequest,
    ConstraintConflict,
    PairRisk,
    ParsedIntent,
    SuggestedChange,
)
from .model_runtime import FeatureTable, HybridModel
from .openai_agent import OpenAIAnswerComposer, OpenAIRequirementsAgent
from .ports import AnswerComposer, IntentParser
from .validation import canonical_name, validate_intent


class AdvisorService:
    def __init__(
        self,
        parser: IntentParser,
        features: FeatureTable,
        model: HybridModel,
        composer: AnswerComposer | None = None,
    ) -> None:
        self.parser = parser
        self.features = features
        self.model = model
        self.composer = composer

    def analyze(self, request: AnalysisRequest) -> AdvisoryResult:
        try:
            intent = self.parser.parse(request)
        except Exception as exc:
            return AdvisoryResult(
                status="input_error",
                summary="DepLab could not understand the request.",
                errors=[str(exc)],
            )
        errors = validate_intent(request, intent)
        if errors:
            return AdvisoryResult(
                status="input_error",
                summary="Please correct the request before compatibility scoring.",
                parsed_intent=intent,
                errors=errors,
            )

        pair_risks, warnings = self._score_intent(intent, request.python_version)
        if not pair_risks:
            result = AdvisoryResult(
                status="coverage_unavailable",
                summary="The frozen DepLab feature table does not cover this exact combination yet.",
                parsed_intent=intent,
                warnings=warnings,
            )
            result.answer = self._fallback_answer(result)
            return result

        failed = [risk for risk in pair_risks if risk.risk_detected]
        alternatives = self._alternatives(intent, request.python_version) if failed else []
        deterministic = [risk for risk in failed if not risk.published_constraints_allow]
        model_only = [risk for risk in failed if risk.published_constraints_allow]
        if deterministic and model_only:
            summary = (
                f"DepLab found {len(deterministic)} published-constraint conflict(s) and "
                f"{len(model_only)} additional model warning(s) in {len(pair_risks)} covered pairs."
            )
        elif deterministic:
            summary = (
                f"DepLab found {len(deterministic)} deterministic published-constraint "
                f"conflict(s) in {len(pair_risks)} covered pairs."
            )
        elif model_only:
            summary = (
                f"DepLab predicts risk in {len(model_only)} of {len(pair_risks)} covered package pairs."
            )
        else:
            summary = f"DepLab did not predict failure in {len(pair_risks)} covered package pairs."
        result = AdvisoryResult(
            status="risk_found" if failed else "no_risk_predicted",
            summary=summary,
            parsed_intent=intent,
            pair_risks=sorted(pair_risks, key=lambda item: item.risk_score, reverse=True),
            alternatives=alternatives,
            warnings=warnings,
            model_scope=(
                "deterministic_constraints_and_prediction"
                if deterministic
                else "prediction_only"
            ),
        )
        payload = result.to_dict()
        result.answer = self.composer.compose(payload) if self.composer else ""
        if not result.answer:
            result.answer = self._fallback_answer(result)
        return result

    def _score_intent(self, intent: ParsedIntent, python_version: str) -> tuple[list[PairRisk], list[str]]:
        target = canonical_name(intent.target_package)
        risks: list[PairRisk] = []
        warnings: list[str] = []
        for requirement in intent.requirements:
            related = canonical_name(requirement.name)
            if related == target:
                continue
            if not requirement.version:
                warnings.append(
                    f"{requirement.name} is not exactly pinned, so its pair could not be scored."
                )
                continue
            row = self.features.find(
                target,
                intent.requested_version,
                related,
                requirement.version,
                python_version,
            )
            if row is None:
                warnings.append(
                    f"No frozen feature row exists for {target}=={intent.requested_version} "
                    f"with {related}=={requirement.version} on Python {python_version}."
                )
                continue
            score = self.model.score(row)
            constraints_allow = self.features.published_constraints_allow(row)
            conflict_records = self.features.constraint_conflict_records(row)
            risks.append(
                PairRisk(
                    family=row["family"],
                    target_package=target,
                    target_version=intent.requested_version,
                    related_package=related,
                    related_version=requirement.version,
                    risk_score=score.risk_score,
                    predicted_failure=score.predicted_failure,
                    logistic_probability=score.logistic_probability,
                    resolution_probability=score.resolution_probability,
                    post_install_probability=score.post_install_probability,
                    evidence_source="frozen release metadata, wheels, constraints, and changelog features",
                    model_id=self.model.model_id,
                    likely_stage=score.likely_stage,
                    published_constraints_allow=constraints_allow,
                    risk_detected=not constraints_allow or score.predicted_failure,
                    evidence_type=(
                        "published_constraint_conflict"
                        if not constraints_allow
                        else "model_prediction"
                    ),
                    constraint_conflicts=[
                        ConstraintConflict(
                            evidence_type="published_constraint_conflict",
                            certainty="deterministic",
                            **record,
                        )
                        for record in conflict_records
                    ],
                )
            )
        return risks, warnings

    def _alternatives(self, intent: ParsedIntent, python_version: str) -> list[Alternative]:
        target = canonical_name(intent.target_package)
        pinned_related = [
            item for item in intent.requirements
            if (
                canonical_name(item.name) != target
                and item.version
                and self.features.has_family(target, item.name, python_version)
            )
        ]
        if not pinned_related:
            return []

        target_requirement = next(
            item for item in intent.requirements if canonical_name(item.name) == target
        )
        target_sets = [
            {
                target_version
                for _, target_version, _ in self.features.pair_rows(
                    target, None, item.name, python_version
                )
            }
            for item in pinned_related
        ]
        versions = set.intersection(*target_sets) if target_sets else set()
        candidates: list[Alternative] = []
        ordered_versions = sorted(
            versions,
            key=lambda version: (
                version != intent.requested_version,
                self._version_distance(version, intent.requested_version),
                self._version_key(version),
            ),
        )
        for version in ordered_versions:
            scores: list[float] = []
            selected_related: list[tuple[str, str, str]] = []
            complete = True
            constraints_allow = True
            for item in pinned_related:
                safe_options: list[tuple[tuple[Any, ...], dict[str, str], str, float]] = []
                for row, _, related_version in self.features.pair_rows(
                    target, version, item.name, python_version
                ):
                    if not self.features.published_constraints_allow(row):
                        continue
                    score = self.model.score(row)
                    if score.predicted_failure:
                        continue
                    current_version = item.version or ""
                    related_key = self._version_key(related_version)
                    current_key = self._version_key(current_version)
                    is_downgrade = related_key < current_key
                    option_rank = (
                        related_version != current_version,
                        is_downgrade,
                        related_key if not is_downgrade else tuple(-part for part in related_key),
                        score.risk_score,
                    )
                    safe_options.append((option_rank, row, related_version, score.risk_score))
                if not safe_options:
                    complete = False
                    break
                _, selected_row, selected_version, selected_score = min(
                    safe_options, key=lambda option: option[0]
                )
                constraints_allow = constraints_allow and self.features.published_constraints_allow(
                    selected_row
                )
                scores.append(selected_score)
                selected_related.append(
                    (canonical_name(item.name), item.version or "", selected_version)
                )
            if complete and constraints_allow:
                changes: list[SuggestedChange] = []
                if target_requirement.version != version:
                    changes.append(
                        SuggestedChange(
                            package=target,
                            from_version=target_requirement.version or "unpinned",
                            to_version=version,
                        )
                    )
                for package, current, selected in selected_related:
                    if current != selected:
                        changes.append(
                            SuggestedChange(
                                package=package,
                                from_version=current,
                                to_version=selected,
                            )
                        )
                candidates.append(
                    
                    Alternative(
                        target_version=version,
                        keeps_requested_target=version == intent.requested_version,
                        category=self._alternative_category(
                            version,
                            intent.requested_version,
                            target_requirement.version,
                        ),
                        changes=changes,
                        maximum_risk_score=max(scores),
                        predicted_failure=False,
                        covered_pairs=len(selected_related),
                        reason=(
                            "Direct published constraints allow every covered pair and DepLab "
                            "does not predict failure for the suggested versions."
                        ),
                    )
                )
        return sorted(
            candidates,
            key=lambda item: self._alternative_sort_key(item, intent.requested_version),
        )[:3]

    @classmethod
    def _alternative_sort_key(
        cls, item: Alternative, requested_version: str
    ) -> tuple[int, int, float, int]:
        return (
            cls._category_priority(item.category),
            len(item.changes),
            item.maximum_risk_score,
            cls._version_distance(item.target_version, requested_version),
        )

    @classmethod
    def _alternative_category(
        cls,
        candidate: str,
        requested: str,
        current: str | None,
    ) -> str:
        if candidate == requested:
            return "achieves_requested_change"
        if current and candidate == current:
            return "keeps_current_version"
        if current and cls._version_key(candidate) < cls._version_key(current):
            return "downgrade_fallback"
        return "different_upgrade_fallback"

    @staticmethod
    def _category_priority(category: str) -> int:
        return {
            "achieves_requested_change": 0,
            "keeps_current_version": 1,
            "different_upgrade_fallback": 2,
            "downgrade_fallback": 3,
        }[category]

    @staticmethod
    def _version_key(version: str) -> tuple[int, ...]:
        parts = []
        for component in version.replace("-", ".").split("."):
            digits = "".join(character for character in component if character.isdigit())
            parts.append(int(digits) if digits else 0)
        return tuple((parts + [0, 0, 0])[:3])

    @classmethod
    def _version_distance(cls, left: str, right: str) -> int:
        a = cls._version_key(left)
        b = cls._version_key(right)
        return abs(a[0] - b[0]) * 1_000_000 + abs(a[1] - b[1]) * 1_000 + abs(a[2] - b[2])

    @staticmethod
    def _fallback_answer(result: AdvisoryResult) -> str:
        if result.status == "coverage_unavailable":
            return (
                "I cannot give a model-backed answer for this exact combination yet. "
                "The package pair or versions are outside the frozen feature table."
            )
        risky = [item for item in result.pair_risks if item.risk_detected]
        if not risky:
            return (
                f"{result.summary} This is a model prediction, not an actual installation test."
            )
        deterministic = [item for item in risky if not item.published_constraints_allow]
        model_only = [item for item in risky if item.published_constraints_allow]
        recommendation = (
            " A covered lower-risk environment is: "
            + ", ".join(
                f"{change.package}=={change.to_version}"
                for change in result.alternatives[0].changes
            )
            + "."
            if result.alternatives and not result.alternatives[0].predicted_failure
            else " No covered low-risk alternative was found in the current frozen table."
        )
        statements: list[str] = []
        for risk in deterministic:
            if risk.constraint_conflicts:
                conflict = risk.constraint_conflicts[0]
                blocking = ", ".join(conflict.blocking_specifiers) or conflict.requirement
                statements.append(
                    f"{conflict.dependency_package}=={conflict.dependency_version} will not resolve "
                    f"with {conflict.declaring_package}=={conflict.declaring_version}: "
                    f"{conflict.declaring_package} declares {conflict.dependency_package}{blocking}."
                )
            else:
                statements.append(
                    f"{risk.target_package}=={risk.target_version} will not resolve with "
                    f"{risk.related_package}=={risk.related_version} under the recorded published constraints."
                )
        for risk in model_only:
            statements.append(
                f"DepLab predicts that {risk.target_package}=={risk.target_version} may fail with "
                f"{risk.related_package}=={risk.related_version} during {risk.likely_stage}."
            )
        evidence_note = (
            " The constraint conflicts above are deterministic facts from published metadata."
            if deterministic
            else ""
        )
        if model_only:
            evidence_note += " The remaining compatibility warnings are model predictions."
        return " ".join(statements) + recommendation + evidence_note + " No environment was installed."


def build_default_service(project_root: Path | None = None) -> Any:
    root = project_root or Path(os.getenv("DEPLAB_PROJECT_ROOT", Path.cwd()))
    model_root = Path(os.getenv("DEPLAB_MODEL_ROOT", str(root)))
    private_outputs = model_root / "outputs"
    feature_paths = [
        private_outputs
        / "deplab-large-features-v3.0.0/development-features.csv",
        private_outputs
        / "deplab-large-features-v3.0.0/validation-inputs.csv",
    ]
    missing = [str(path) for path in feature_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing production DepLab feature table: " + ", ".join(missing)
        )
    from .cascade_service import CascadeAdvisorService, load_post_install_threshold
    from .production_runtime import (
        ModernBertPostInstallModel,
        StructuredRankingModel,
    )
    from .resolver import UVCompileVerifier

    candidate_root = (
        private_outputs / "deplab-large-candidate-freeze-v3.0.0"
    )
    structured = StructuredRankingModel(
        candidate_root / "candidate-structured_weighted_logistic.json"
    )
    post_install = ModernBertPostInstallModel(
        candidate_root / "candidate-modernbert_stage_aware_hybrid.json",
        private_outputs / "large-release-modernbert-v3.0.0.jsonl",
        load_post_install_threshold(
            root / "configs/post-install-policy-v1.0.0.json"
        ),
    )
    resolver = UVCompileVerifier(
        uv_command=os.getenv("DEPLAB_UV_COMMAND", "uv"),
        timeout_seconds=float(os.getenv("DEPLAB_RESOLVER_TIMEOUT_SECONDS", "15")),
        maximum_concurrency=int(
            os.getenv("DEPLAB_RESOLVER_MAXIMUM_CONCURRENCY", "2")
        ),
        uv_cache_dir=Path(
            os.getenv(
                "DEPLAB_UV_CACHE_DIR",
                "/opt/deplab/shared/uv-cache",
            )
        ),
    )
    api_key = os.getenv("OPENAI_API_KEY")
    return CascadeAdvisorService(
        parser=OpenAIRequirementsAgent(api_key=api_key),
        features=FeatureTable(feature_paths),
        structured_model=structured,
        post_install_model=post_install,
        resolver=resolver,
        composer=OpenAIAnswerComposer(api_key=api_key) if api_key else None,
    )
