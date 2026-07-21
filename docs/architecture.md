# DepLab architecture and milestone plan

## Smallest credible end-to-end MVP

One exact package pair on one target Python version is enough to prove the full data path, but not enough to support compatibility claims. The vertical slice must:

1. fetch both exact releases from PyPI;
2. retain metadata and compatible Linux x86_64 wheel artifacts;
3. classify missing wheels as coverage, not incompatibility;
4. create a fresh uv environment for the target Python;
5. resolve a wheel-only PEP 751 lock, choose one compatible artifact per package, and install those exact URLs with SHA-256 validation and dependency resolution disabled;
6. capture the actual Python patch, implementation, Linux kernel, architecture, libc and uv version;
7. import both packages and run an interoperability smoke test when registered;
8. capture stage logs, durations, the resolved environment, outcome, and artifact identity in JSONL;
9. skip an experiment ID already present in the output, while leaving infrastructure failures retryable.

The included `requests`/`urllib3` slice is a pipeline check. Any output from it is a measured observation only when run on the declared Linux platform; fixture-driven tests are labeled test data and are not experimental evidence.

## Components

| Component | Current responsibility | Next extension |
|---|---|---|
| `pypi.py` | Exact-release metadata and files | cache responses; release discovery |
| `wheels.py` | Conservative CPython/glibc Linux x86_64 eligibility | use `packaging.tags` in a locked runtime; record the runner's glibc baseline |
| `smoke.py` | imports plus pair-specific interoperability | registry modules, severity and test versioning |
| `artifacts.py` | select and validate every exact top-level and transitive wheel from the PEP 751 lock | stronger tag ranking with `packaging.tags` |
| `runner.py` | isolated uv lifecycle, exact-artifact installation and stage classification | retry policy and cloud telemetry |
| `matrix.py` | deterministic Cartesian generation with audited wheel-coverage filtering | partition and shard planning |
| `shards.py` | lossless, deterministic execution shards | distributed shard leases |
| `measurements.py` | cache size, disk, memory, load, network and process-tree peak RSS | CPU-credit and EBS telemetry from the cloud provider |
| `changelogs.py` | version-pinned official release-note collection, hashing, release-series selection and deterministic compatibility signals | add source definitions as the package scope expands |
| `storage.py` | append-only JSONL and resume IDs | partitioned Parquet and manifest checksums |
| CLI | inspect or run one pair | schedule manifests and pilot batches |

## Observation contract

The experiment ID is deterministic over both package pins, Python, OS, and architecture. A schema 1.3 observation stores the full spec, outcome taxonomy, selected top-level metadata, exact installed wheel artifacts and hashes, lock hash, installed package versions, command logs, exit codes, exceptions, resources, and timings. OS and architecture are explicit even though fixed in the MVP.

The lock is used for resolution evidence. Installation receives the chosen wheel URLs directly with `--no-deps`, so the recorded artifact list is the list actually installed rather than a list of possible candidates.

## Milestones

### M0 — vertical slice (implemented)

- metadata collection, wheel prefilter, isolated wheel-only run;
- imports/interoperability registry;
- structured/resumable JSONL;
- deterministic tests and honest infrastructure failures.

Exit criterion: unit tests pass and one real Linux run produces a `pass` or correctly categorized failure with no source build.

### M1 — 50-combination pilot

- choose 8–10 package families and 6–8 releases using release history rather than guesses;
- generate eligible pair/Python combinations;
- add bounded parallelism, shared uv cache, network/infrastructure-only retries;
- measure cold/warm time, cache and run-disk growth, peak RSS, download bytes, failure rates;
- export JSONL and Parquet with a versioned run manifest.

Exit criterion: observed measurements justify worker count, storage, IOPS, instance size, and main-run cost.

### Measurement interpretation

- A per-experiment cache directory that starts empty is a true cold-cache observation for that experiment.
- Reusing the same per-experiment directory for a second run is the controlled warm-cache observation.
- Cache bytes are logical file sizes. The disk-free delta is the host filesystem's physical change and can include other host activity.
- Peak RSS is sampled from each command and all of its visible Linux child processes.
- Systematic runs may remove the temporary `.venv` after measurements; the lock, result and stage logs are retained.
- Memory availability, load average and network byte counters are host-level snapshots. Run one worker on an otherwise idle instance when establishing the baseline.
- T3 CPU credits and EBS behavior are not yet recorded, so measurements from the `t3.medium` validate instrumentation but must not be used to size the final 30–40-worker machine.

### M2 — main empirical dataset

- run the eligible matrix on Linux x86_64 for Python 3.10–3.12;
- validate completeness, duplicates, schema, log retention and artifact hashes;
- extract deterministic changelog sections and label their provenance.

Exit criterion: reproducible, versioned dataset with an audit report. No simulated rows mixed with measured data.

### Changelog scaling and leakage policy

Changelogs are normalized at the package-release level, then joined into experiment rows. A release record is fetched once and reused for every pair and Python version that contains it. The source registry is explicit and version-pinned so a later documentation edit cannot silently change an old training row. Stored hashes provide provenance.

Only information available before an experiment runs may become a model input. Changelog keyword and co-occurrence signals are allowed; failure messages, tracebacks, outcomes and post-install evidence are not. When a new signal is designed after inspecting test errors, that test set becomes development evidence and a later untouched holdout is required for the next honest final evaluation.

### M3 — evaluation and demo

- package-family grouped splits with reverse/near-duplicate containment;
- metadata, timing, wheel/ABI and changelog baselines before learned models;
- dashboard for exact-result lookup and clearly labeled risk estimates.

Exit criterion: held-out-family metrics, baseline comparison, and a demo that distinguishes observed compatibility from prediction or missing coverage.

## Cost guardrails

Do not launch the large run until the pilot records cold/warm distributions and disk contention. Default to local deterministic tests and a small Linux runner. Changelog LLM processing should receive only keyword/heading-selected excerpts, have cached outputs, and stop at the stated budget. Fine-tuning remains optional until dataset validity and leakage checks pass.
