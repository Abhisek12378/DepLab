from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deplab.advisor.cascade_service import (
    CascadeAdvisorService,
    load_post_install_threshold,
)
from deplab.advisor.contracts import (
    AnalysisRequest,
    ParsedIntent,
    RequirementPin,
)
from deplab.advisor.model_runtime import FeatureTable
from deplab.advisor.production_runtime import (
    ModernBertPostInstallModel,
    StructuredRankingModel,
)
from deplab.advisor.resolver import ResolverResult, UVCompileVerifier


class StaticParser:
    def parse(self, request: AnalysisRequest) -> ParsedIntent:
        return ParsedIntent(
            requirements=[
                RequirementPin("alpha", "1.0", "alpha==1.0"),
                RequirementPin("beta", "1.0", "beta==1.0"),
            ],
            target_package="alpha",
            requested_version="2.0",
        )


class FakeStructuredModel:
    model_id = "structured-ranking"

    def score(self, row: dict[str, str]) -> object:
        return type(
            "Score",
            (),
            {
                "probability": float(row.get("ranking_risk", "0.1")),
                "predicted_failure": False,
            },
        )()


class FakePostInstallModel:
    model_id = "post-install"

    def score(self, row: dict[str, str]) -> object:
        risk = float(row.get("post_risk", "0.1"))
        return type(
            "Score",
            (),
            {
                "import_probability": risk,
                "smoke_probability": 0.01,
                "combined_probability": risk,
                "predicted_import_failure": risk >= 0.5,
                "predicted_smoke_failure": False,
                "risk_detected": risk >= 0.5,
            },
        )()


class FakeResolver:
    def __init__(self, result: ResolverResult) -> None:
        self.result = result
        self.calls: list[dict[str, str]] = []

    def verify(
        self,
        requirements: dict[str, str],
        python_version: str,
        platform: str,
    ) -> ResolverResult:
        self.calls.append(dict(requirements))
        return self.result


class ProductionAgentTests(unittest.TestCase):
    def test_uv_compile_uses_arguments_without_a_shell_and_caches(self) -> None:
        verifier = UVCompileVerifier(cache_ttl_seconds=60)
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="",
        )
        with patch("subprocess.run", return_value=completed) as run:
            first = verifier.verify(
                {"Alpha": "1.0", "Beta": "2.0"},
                "3.11",
                "linux-x86_64",
            )
            second = verifier.verify(
                {"beta": "2.0", "alpha": "1.0"},
                "3.11",
                "linux-x86_64",
            )
        self.assertTrue(first.resolvable)
        self.assertTrue(second.cache_hit)
        self.assertEqual(run.call_count, 1)
        command = run.call_args.args[0]
        self.assertIn("--python-version", command)
        self.assertIn("x86_64-unknown-linux-gnu", command)
        self.assertFalse(run.call_args.kwargs.get("shell", False))

    def test_uv_compile_does_not_call_network_for_an_invalid_version(self) -> None:
        verifier = UVCompileVerifier()
        with patch("subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "invalid exact version"):
                verifier.verify(
                    {"alpha": "1.0; touch unsafe"},
                    "3.11",
                    "linux-x86_64",
                )
        run.assert_not_called()

    def test_frozen_models_score_without_torch_or_transformers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            structured_path = root / "structured.json"
            stage_path = root / "stage.json"
            embeddings_path = root / "embeddings.jsonl"
            preprocessor = {
                "numeric_columns": ["value"],
                "categorical_columns": [],
                "numeric_medians": {"value": 0.0},
                "numeric_means": {"value": 0.0},
                "numeric_scales": {"value": 1.0},
                "category_levels": {},
            }
            structured_path.write_text(
                json.dumps(
                    {
                        "pipeline_id": "deplab-large-hybrid-v3.0.0",
                        "training_rows": 21490,
                        "candidate": "structured_weighted_logistic",
                        "threshold": 0.5,
                        "preprocessor": preprocessor,
                        "model": {
                            "type": "weighted_logistic",
                            "weights": [1.0],
                            "intercept": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            projection = {
                "mean": [0.0] * 768,
                "components": [[1.0] + [0.0] * 767],
            }
            leaf = {
                "base_logit": 0.0,
                "learning_rate": 1.0,
                "trees": [
                    {
                        "nodes": [
                            {
                                "feature": -1,
                                "split_bin": -1,
                                "left": -1,
                                "right": -1,
                                "value": 1.0,
                            }
                        ]
                    }
                ],
            }
            stage_path.write_text(
                json.dumps(
                    {
                        "pipeline_id": "deplab-large-hybrid-v3.0.0",
                        "training_rows": 21490,
                        "candidate": "modernbert_stage_aware_hybrid",
                        "threshold": 0.5,
                        "preprocessor": preprocessor,
                        "text_features": {
                            "encoder": "answerdotai/ModernBERT-base",
                            "encoder_frozen": True,
                            "release_projector": projection,
                            "text_scaler": {
                                "mean": [0.0] * 4,
                                "scale": [1.0] * 4,
                            },
                        },
                        "binner": {"thresholds": [[] for _ in range(5)]},
                        "model": {
                            "type": "stage_aware_histogram_gradient_boosting",
                            "import_failure_head": leaf,
                            "smoke_failure_head": leaf,
                        },
                    }
                ),
                encoding="utf-8",
            )
            embedding_rows = []
            for package, first in (("alpha", 1.0), ("beta", 0.0)):
                vector = [first] + [0.0] * 767
                embedding_rows.append(
                    json.dumps(
                        {
                            "package": package,
                            "version": "1.0",
                            "model": "answerdotai/ModernBERT-base",
                            "embedding_dimension": 768,
                            "embedding": vector,
                        }
                    )
                )
            embeddings_path.write_text(
                "\n".join(embedding_rows) + "\n",
                encoding="utf-8",
            )
            structured = StructuredRankingModel(structured_path)
            stage = ModernBertPostInstallModel(
                stage_path,
                embeddings_path,
                post_install_threshold=0.5,
            )
            row = {
                "value": "1",
                "package_a_name": "alpha",
                "package_a_version": "1.0",
                "package_b_name": "beta",
                "package_b_version": "1.0",
            }
            self.assertTrue(structured.score(row).predicted_failure)
            self.assertTrue(stage.score(row).risk_detected)  # type: ignore[union-attr]

    def test_cascade_requires_uv_success_before_recommendation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            table = _feature_table(Path(directory))
            resolver = FakeResolver(
                ResolverResult(
                    status="unresolvable",
                    resolvable=False,
                    duration_seconds=0.01,
                    explanation="No solution.",
                )
            )
            service = CascadeAdvisorService(
                parser=StaticParser(),
                features=table,
                structured_model=FakeStructuredModel(),  # type: ignore[arg-type]
                post_install_model=FakePostInstallModel(),  # type: ignore[arg-type]
                resolver=resolver,
            )
            result = service.analyze(
                AnalysisRequest(
                    "alpha==1.0\nbeta==1.0",
                    "Can I upgrade alpha to 2.0?",
                    "3.11",
                )
            )
        self.assertEqual(result.status, "risk_found")
        self.assertEqual(result.verification_status, "resolver_rejected")
        self.assertFalse(result.alternatives)

    def test_cascade_returns_only_resolver_checked_low_risk_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            table = _feature_table(Path(directory))
            resolver = FakeResolver(
                ResolverResult(
                    status="resolved",
                    resolvable=True,
                    duration_seconds=0.01,
                    explanation="Lock created.",
                )
            )
            service = CascadeAdvisorService(
                parser=StaticParser(),
                features=table,
                structured_model=FakeStructuredModel(),  # type: ignore[arg-type]
                post_install_model=FakePostInstallModel(),  # type: ignore[arg-type]
                resolver=resolver,
            )
            result = service.analyze(
                AnalysisRequest(
                    "alpha==1.0\nbeta==1.0",
                    "Can I upgrade alpha to 2.0?",
                    "3.11",
                )
            )
        self.assertEqual(result.status, "no_risk_predicted")
        self.assertEqual(result.verification_status, "resolver_verified")
        self.assertIn("No packages were installed", result.answer)

    def test_frozen_post_install_policy_is_valid(self) -> None:
        root = Path(__file__).resolve().parents[1]
        threshold = load_post_install_threshold(
            root / "configs/post-install-policy-v1.0.0.json"
        )
        self.assertAlmostEqual(threshold, 0.4726886805608258)


def _feature_table(root: Path) -> FeatureTable:
    path = root / "features.csv"
    columns = [
        "family",
        "package_a_name",
        "package_a_version",
        "package_b_name",
        "package_b_version",
        "python_version",
        "package_a_declares_package_b",
        "package_a_requirement_allows_b",
        "package_a_requirement_on_b",
        "package_b_declares_package_a",
        "package_b_requirement_allows_a",
        "package_b_requirement_on_a",
        "ranking_risk",
        "post_risk",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for alpha in ("1.0", "2.0", "3.0"):
            writer.writerow(
                {
                    "family": "alpha-beta",
                    "package_a_name": "alpha",
                    "package_a_version": alpha,
                    "package_b_name": "beta",
                    "package_b_version": "1.0",
                    "python_version": "3.11",
                    "package_a_declares_package_b": "false",
                    "package_a_requirement_allows_b": "true",
                    "package_a_requirement_on_b": "",
                    "package_b_declares_package_a": "false",
                    "package_b_requirement_allows_a": "true",
                    "package_b_requirement_on_a": "",
                    "ranking_risk": "0.1",
                    "post_risk": "0.1",
                }
            )
    return FeatureTable([path])


if __name__ == "__main__":
    unittest.main()
