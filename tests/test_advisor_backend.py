from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deplab.advisor.contracts import (
    Alternative,
    AnalysisRequest,
    ParsedIntent,
    RequirementPin,
    SuggestedChange,
)
from deplab.advisor.model_runtime import FeatureTable, HybridModel
from deplab.advisor.service import AdvisorService
from deplab.advisor.validation import validate_intent


ROOT = Path(__file__).resolve().parents[1]


class StaticParser:
    def parse(self, request: AnalysisRequest) -> ParsedIntent:
        return ParsedIntent(
            requirements=[
                RequirementPin("numpy", "1.21.6", "numpy==1.21.6"),
                RequirementPin("pandas", "1.3.5", "pandas==1.3.5"),
            ],
            target_package="numpy",
            requested_version="1.21.6",
        )


class UpgradeParser:
    def parse(self, request: AnalysisRequest) -> ParsedIntent:
        return ParsedIntent(
            requirements=[
                RequirementPin("numpy", "1.26.4", "numpy==1.26.4"),
                RequirementPin("pandas", "2.1.4", "pandas==2.1.4"),
                RequirementPin("scipy", "1.11.4", "scipy==1.11.4"),
                RequirementPin("requests", "2.31.0", "requests==2.31.0"),
            ],
            target_package="numpy",
            requested_version="2.0.2",
        )


class ModelOnlyWarningParser:
    def parse(self, request: AnalysisRequest) -> ParsedIntent:
        return ParsedIntent(
            requirements=[
                RequirementPin("numpy", "1.21.6", "numpy==1.21.6"),
                RequirementPin("pandas", "2.0.3", "pandas==2.0.3"),
            ],
            target_package="numpy",
            requested_version="1.23.5",
        )


class AdvisorBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.features_path = ROOT / "outputs/deplab-expanded-development-v2.0.0/features.csv"
        cls.model = HybridModel(
            ROOT / "outputs/deplab-expanded-weighted-logistic-v2.0.0/model.json",
            ROOT / "outputs/deplab-advanced-model-comparison-v3.0.0/model.json",
            ROOT / "outputs/deplab-hybrid-validation-v3.0.0/metrics.json",
        )

    def test_real_frozen_row_is_scored(self) -> None:
        table = FeatureTable([self.features_path])
        row = table.find("numpy", "1.21.6", "pandas", "1.3.5", "3.10")
        self.assertIsNotNone(row)
        score = self.model.score(row or {})
        self.assertGreaterEqual(score.risk_score, 0.0)
        self.assertLessEqual(score.risk_score, 1.0)

    def test_service_returns_structured_prediction(self) -> None:
        service = AdvisorService(
            parser=StaticParser(),
            features=FeatureTable([self.features_path]),
            model=self.model,
        )
        result = service.analyze(
            AnalysisRequest("numpy==1.21.6\npandas==1.3.5", "Can I use NumPy 1.21.6?", "3.10")
        )
        self.assertIn(result.status, {"risk_found", "no_risk_predicted"})
        self.assertEqual(len(result.pair_risks), 1)
        self.assertEqual(result.verification_status, "not_runtime_verified")

    def test_feature_table_accepts_reversed_lookup(self) -> None:
        table = FeatureTable([self.features_path])
        direct = table.find("numpy", "1.21.6", "pandas", "1.3.5", "3.10")
        reversed_row = table.find("pandas", "1.3.5", "numpy", "1.21.6", "3.10")
        self.assertEqual(direct and direct["experiment_id"], reversed_row and reversed_row["experiment_id"])

    def test_validator_rejects_a_requirement_invented_by_the_llm(self) -> None:
        intent = ParsedIntent(
            requirements=[RequirementPin("numpy", "1.21.6", "numpy==1.21.6")],
            target_package="numpy",
            requested_version="1.24.4",
        )
        errors = validate_intent(
            AnalysisRequest("pandas==1.3.5", "Upgrade NumPy", "3.10"),
            intent,
        )
        self.assertTrue(any("not found verbatim" in message for message in errors))

    def test_solver_changes_conflicting_dependents_and_keeps_requested_numpy(self) -> None:
        service = AdvisorService(
            parser=UpgradeParser(),
            features=FeatureTable([self.features_path]),
            model=self.model,
        )
        result = service.analyze(
            AnalysisRequest(
                "numpy==1.26.4\npandas==2.1.4\nscipy==1.11.4\nrequests==2.31.0",
                "Can I upgrade NumPy to 2.0.2?",
                "3.11",
            )
        )
        self.assertEqual(result.status, "risk_found")
        self.assertTrue(result.alternatives)
        recommendation = result.alternatives[0]
        self.assertTrue(recommendation.keeps_requested_target)
        changes = {item.package: item.to_version for item in recommendation.changes}
        self.assertEqual(changes["numpy"], "2.0.2")
        self.assertEqual(changes["pandas"], "2.2.2")
        self.assertEqual(changes["scipy"], "1.13.1")
        self.assertFalse(recommendation.predicted_failure)
        self.assertLessEqual(len(result.alternatives), 3)
        self.assertEqual(
            [item.category for item in result.alternatives[:2]],
            ["achieves_requested_change", "keeps_current_version"],
        )
        self.assertEqual(result.model_scope, "deterministic_constraints_and_prediction")
        self.assertTrue(all(not risk.published_constraints_allow for risk in result.pair_risks))
        blocking = {
            clause
            for risk in result.pair_risks
            for conflict in risk.constraint_conflicts
            for clause in conflict.blocking_specifiers
        }
        self.assertEqual(blocking, {"<2", "<1.28.0"})
        self.assertIn("will not resolve", result.answer)
        self.assertNotIn("may fail", result.answer)

    def test_equal_category_and_change_count_are_ranked_by_risk(self) -> None:
        def fallback(version: str, risk: float) -> Alternative:
            return Alternative(
                target_version=version,
                keeps_requested_target=False,
                category="downgrade_fallback",
                changes=[SuggestedChange("numpy", "1.26.4", version)],
                maximum_risk_score=risk,
                predicted_failure=False,
                covered_pairs=2,
                reason="test",
            )

        alternatives = [
            fallback("1.23.5", 0.105),
            fallback("1.24.4", 0.040),
            fallback("1.25.2", 0.049),
        ]
        ordered = sorted(
            alternatives,
            key=lambda item: AdvisorService._alternative_sort_key(item, "2.0.2"),
        )
        self.assertEqual(
            [item.target_version for item in ordered],
            ["1.24.4", "1.25.2", "1.23.5"],
        )

    def test_model_only_warning_uses_prediction_language(self) -> None:
        service = AdvisorService(
            parser=ModelOnlyWarningParser(),
            features=FeatureTable([self.features_path]),
            model=self.model,
        )
        result = service.analyze(
            AnalysisRequest(
                "numpy==1.21.6\npandas==2.0.3",
                "Can I upgrade NumPy to 1.23.5?",
                "3.10",
            )
        )
        self.assertEqual(result.status, "risk_found")
        self.assertTrue(all(risk.published_constraints_allow for risk in result.pair_risks))
        self.assertTrue(all(risk.evidence_type == "model_prediction" for risk in result.pair_risks))
        self.assertIn("may fail", result.answer)
        self.assertNotIn("will not resolve", result.answer)


if __name__ == "__main__":
    unittest.main()
