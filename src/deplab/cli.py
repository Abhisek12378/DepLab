from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .batch import ManifestError, run_batch
from .catalog import ScopeError, collect_catalog
from .changelogs import ChangelogError, collect_changelogs
from .matrix import MatrixError, generate_matrix
from .models import ExperimentSpec, PackagePin
from .pypi import PyPIClient, PyPIError
from .runner import ExperimentRunner
from .scope_audit import ScopeAuditError, audit_scope
from .scope_plan import ScopePlanError, build_scope_draft
from .shards import ShardError, shard_manifest
from .storage import append_jsonl, completed_ids


PYTHON_CHOICES = ("3.8", "3.9", "3.10", "3.11", "3.12", "3.13", "3.14")


def _pin(value: str) -> PackagePin:
    if "==" not in value:
        raise argparse.ArgumentTypeError("package pins must use NAME==VERSION")
    name, version = value.split("==", 1)
    if not name or not version:
        raise argparse.ArgumentTypeError("package pins must use NAME==VERSION")
    return PackagePin(name=name, version=version)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deplab", description="Empirical dependency compatibility lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="collect PyPI metadata and wheel eligibility")
    inspect_parser.add_argument("package", type=_pin)
    inspect_parser.add_argument("--python", required=True, choices=PYTHON_CHOICES)

    catalog_parser = subparsers.add_parser("catalog", help="collect the audited PyPI package scope")
    catalog_parser.add_argument("--scope", type=Path, required=True)
    catalog_parser.add_argument("--output", type=Path, default=Path("outputs/package-catalog.jsonl"))

    changelog_parser = subparsers.add_parser(
        "changelogs", help="collect deterministic release-note compatibility signals"
    )
    changelog_parser.add_argument("--scope", type=Path, required=True)
    changelog_parser.add_argument("--sources", type=Path, required=True)
    changelog_parser.add_argument(
        "--output", type=Path, default=Path("outputs/changelog-catalog.jsonl")
    )

    scope_audit_parser = subparsers.add_parser(
        "scope-audit", help="add verified PyPI wheel coverage to a version scope draft"
    )
    scope_audit_parser.add_argument("--input", type=Path, required=True)
    scope_audit_parser.add_argument("--output", type=Path, required=True)

    scope_plan_parser = subparsers.add_parser(
        "scope-plan", help="select a reproducible, stable release scope from a large-dataset plan"
    )
    scope_plan_parser.add_argument("--plan", type=Path, required=True)
    scope_plan_parser.add_argument("--output", type=Path, required=True)

    matrix_parser = subparsers.add_parser(
        "matrix", help="generate the systematic wheel-eligible experiment matrix"
    )
    matrix_parser.add_argument("--scope", type=Path, required=True)
    matrix_parser.add_argument("--pairs", type=Path, required=True)
    matrix_parser.add_argument("--output", type=Path, default=Path("configs/systematic-matrix.json"))

    shard_parser = subparsers.add_parser(
        "shard", help="divide a validated experiment manifest into smaller manifests"
    )
    shard_parser.add_argument("--manifest", type=Path, required=True)
    shard_parser.add_argument("--output-dir", type=Path, required=True)
    shard_parser.add_argument("--size", type=int, default=50)

    run_parser = subparsers.add_parser("run", help="run one isolated, wheel-only experiment")
    run_parser.add_argument("package_a", type=_pin)
    run_parser.add_argument("package_b", type=_pin)
    run_parser.add_argument("--python", required=True, choices=PYTHON_CHOICES)
    run_parser.add_argument("--output", type=Path, default=Path("outputs/results.jsonl"))
    run_parser.add_argument("--run-root", type=Path, default=Path("work/runs"))
    run_parser.add_argument("--timeout", type=float, default=180.0)
    run_parser.add_argument("--cache-dir", type=Path, help="use and measure this uv cache directory")
    run_parser.add_argument(
        "--cache-scope", choices=("shared", "experiment"), default="shared"
    )
    run_parser.add_argument("--force", action="store_true", help="rerun even if the experiment ID exists")
    run_parser.add_argument(
        "--cleanup-environments",
        action="store_true",
        help="remove each temporary virtual environment after its result is captured",
    )

    batch_parser = subparsers.add_parser("batch", help="run a resumable experiment manifest")
    batch_parser.add_argument("--manifest", type=Path, required=True)
    batch_parser.add_argument("--output", type=Path, default=Path("outputs/results.jsonl"))
    batch_parser.add_argument("--run-root", type=Path, default=Path("work/runs"))
    batch_parser.add_argument("--workers", type=int, default=1)
    batch_parser.add_argument("--timeout", type=float, default=180.0)
    batch_parser.add_argument("--cache-dir", type=Path, help="use and measure this shared uv cache directory")
    batch_parser.add_argument(
        "--cache-scope",
        choices=("shared", "experiment"),
        default="shared",
        help="give all runs one cache or a separate cache per experiment",
    )
    batch_parser.add_argument(
        "--cleanup-environments",
        action="store_true",
        help="remove each temporary virtual environment after its result is captured",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = PyPIClient()
    try:
        if args.command == "inspect":
            release = client.release(args.package.name, args.package.version, args.python)
            print(json.dumps(asdict(release), indent=2, sort_keys=True))
            return 0

        if args.command == "catalog":
            summary = collect_catalog(args.scope, args.output, client=client)
            print(json.dumps(asdict(summary), indent=2, sort_keys=True))
            return 0

        if args.command == "changelogs":
            summary = collect_changelogs(args.scope, args.sources, args.output)
            print(json.dumps(asdict(summary), indent=2, sort_keys=True))
            return 0

        if args.command == "scope-audit":
            summary = audit_scope(args.input, args.output, client=client)
            print(json.dumps(asdict(summary), indent=2, sort_keys=True))
            return 0

        if args.command == "scope-plan":
            summary = build_scope_draft(args.plan, args.output, client=client)
            print(json.dumps(asdict(summary), indent=2, sort_keys=True))
            return 0

        if args.command == "matrix":
            summary = generate_matrix(args.scope, args.pairs, args.output)
            print(json.dumps(asdict(summary), indent=2, sort_keys=True))
            return 0

        if args.command == "shard":
            summary = shard_manifest(args.manifest, args.output_dir, args.size)
            print(json.dumps(asdict(summary), indent=2, sort_keys=True))
            return 0

        if args.command == "batch":
            runner = ExperimentRunner(
                args.run_root,
                timeout_seconds=args.timeout,
                uv_cache_dir=args.cache_dir,
                cache_scope=args.cache_scope,
                cleanup_environments=args.cleanup_environments,
                measure_cache_contents=not (
                    args.workers > 1 and args.cache_scope == "shared"
                ),
            )
            summary = run_batch(
                args.manifest, args.output, runner, client=client, workers=args.workers
            )
            print(json.dumps(asdict(summary), indent=2, sort_keys=True))
            return 3 if "infrastructure_failure" in summary.outcome_counts else 0

        spec = ExperimentSpec(args.package_a, args.package_b, args.python)
        if not args.force and spec.experiment_id in completed_ids(args.output):
            print(json.dumps({"experiment_id": spec.experiment_id, "status": "already_completed"}))
            return 0
        release_a = client.release(args.package_a.name, args.package_a.version, args.python)
        release_b = client.release(args.package_b.name, args.package_b.version, args.python)
        result = ExperimentRunner(
            args.run_root,
            timeout_seconds=args.timeout,
            uv_cache_dir=args.cache_dir,
            cache_scope=args.cache_scope,
            cleanup_environments=args.cleanup_environments,
        ).run(
            spec, release_a, release_b
        )
        append_jsonl(args.output, result.to_dict())
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0 if result.outcome == "pass" else 2
    except (
        PyPIError,
        ManifestError,
        ScopeError,
        ChangelogError,
        ScopeAuditError,
        ScopePlanError,
        MatrixError,
        ShardError,
        ValueError,
    ) as exc:
        print(json.dumps({"outcome": "infrastructure_failure", "error": str(exc)}), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
