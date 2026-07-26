# DepLab

**Empirical Python dependency compatibility—tested, not guessed.**

DepLab builds compatibility evidence by installing exact Python package versions into fresh environments and exercising them. A successful resolver is not considered proof of compatibility: both imports and the registered smoke test must pass.

This repository is a clean-room implementation created for the OpenAI Build Week project. It contains no DepDoctor code.

## Current vertical slice

- collects exact-release metadata from PyPI;
- filters for non-yanked wheels compatible with Python 3.10, 3.11, or 3.12 on Linux x86_64;
- resolves a standard wheel-only `pylock.toml`, then installs every exact wheel URL with SHA-256 verification;
- categorizes coverage, resolution, installation, import, smoke, timeout, and infrastructure outcomes;
- captures logs, timings, every top-level and transitive wheel artifact, and the installed environment;
- appends versioned JSONL and skips completed experiment IDs.
- runs validated manifests with bounded workers; infrastructure failures remain retryable.
- safely replaces a stale environment only inside the matching experiment directory when retrying.

The architecture and staged hackathon plan are in [docs/architecture.md](docs/architecture.md).

## Streamlit dependency advisor

The first chatbot prototype is a thin Streamlit screen over the reusable
`deplab.advisor` backend. GPT-5.6 parses `requirements.txt` and the user's
question into a strict structure. DepLab then validates that structure and
scores every exact package pair covered by the frozen feature table. It can
search and rank complete alternative environments across all exactly pinned
related packages. The solver first tries to keep the requested target version
and change conflicting related packages; only then does it try another target
version. Suggested pairs must satisfy recorded direct published constraints
and must not be classified as failures by the frozen model.

Responses preserve evidence certainty: a direct published constraint conflict
is reported as a deterministic non-resolving combination, while a warning from
the compatibility model is reported as a prediction. Recommendations are
grouped by whether they achieve the requested change, keep the current target,
or require a fallback/downgrade; risk breaks ties before version distance.

The application does **not** install the proposed environment. The structured
model ranks candidate environments, `uv pip compile` verifies dependency
resolution for a bounded number of candidates, and the ModernBERT stage-aware
model predicts post-install import and smoke-test risk. Deterministic
constraint facts, resolver results, and model predictions are labelled
separately. Unsupported package pairs,
versions, or Python versions return an explicit coverage message instead of a
guessed score.

Install the optional application dependencies in a virtual environment:

```bash
python -m pip install -e ".[app]"
```

Set the API key and start the app:

```bash
export OPENAI_API_KEY="your-key"
streamlit run apps/streamlit_app.py
```

On Windows PowerShell, set the key with
`$env:OPENAI_API_KEY="your-key"` before running the same Streamlit command.
The default parser model is `gpt-5.6`; it can be changed with
`DEPLAB_OPENAI_MODEL`. The backend service returns dataclasses that can later
be exposed directly through FastAPI without moving logic into the route or the
React client.

## React chatbot and FastAPI

The production-facing prototype lives in `frontend/` and talks only to the
versioned FastAPI surface in `deplab.api`. Conversation history is held on the
server for two hours by default. The browser stores only an opaque conversation
ID in session storage, so a refresh restores the chat without putting the
requirements file or OpenAI key into persistent browser storage.

Install the backend dependencies and start the API:

```bash
python -m pip install -e ".[api]"
cp .env.example .env
# Put OPENAI_API_KEY in .env. Never commit that file.
deplab-api
```

In a second terminal, start the React/TypeScript application:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The local Vite server proxies `/api` to
`http://127.0.0.1:8000`; deployment can instead set `VITE_API_BASE_URL`.
Interactive API documentation is available at `http://127.0.0.1:8000/api/docs`.

The API includes bounded input sizes, TTL and message limits, idempotent message
IDs, per-conversation serialization, CORS allowlisting, request IDs, safe error
responses, and no-store security headers. The in-memory conversation store can
be replaced with Redis before horizontal scaling without changing the React
client or advisory service.

## CI/CD and private model artifacts

GitHub Actions runs source-only backend tests, the public-artifact safety check,
TypeScript checks, and the production frontend build on every pull request and
push to `main`. Deployment remains disabled until the EC2 host is provisioned
and the repository variable `EC2_DEPLOY_ENABLED` is set to `true`.

The trained model, feature table, experiment outputs, `.env`, SSH keys, caches,
and local environments are intentionally excluded from Git. EC2 stores the
private model files under `/opt/deplab/shared/models/current`; a deployment changes
only the versioned source release and preserves those private files. See
[`deploy/ec2/README.md`](deploy/ec2/README.md) for the deployment contract and
required GitHub secrets.

The audited ten-package/seven-version scope is in `configs/package-scope.json`. The first real pilot is `configs/pilot-50.json`: 50 wheel-eligible experiments across six package-pair families. Each row's `selection_hypothesis` explains why the row is informative but is never used as an outcome label.

Generate the full systematic matrix from the audited scope and registered pair families:

```bash
PYTHONPATH=src python3 -m deplab matrix \
  --scope configs/package-scope.json \
  --pairs configs/pair-families.json \
  --output configs/systematic-matrix.json
```

The current scope produces 646 runnable rows. Another 236 of the original 882 Cartesian rows are excluded because at least one top-level wheel is unavailable for that Python version. This is a coverage filter, not a compatibility result.

Divide the generated matrix into resumable 50-row execution shards:

```bash
PYTHONPATH=src python3 -m deplab shard \
  --manifest configs/systematic-matrix.json \
  --output-dir configs/systematic-shards \
  --size 50
```

This produces 13 manifests: twelve contain 50 experiments and the last contains 46. On the small EC2 validation machine, run a chosen inclusive range with `bash scripts/run_systematic_shards.sh 1 1`. The script uses one worker, one shared cache, one resumable result file, and stops before starting another shard if free disk falls below 4 GiB.

After downloading the completed JSONL, run the strict dataset audit:

```bash
python scripts/audit_systematic_dataset.py \
  --manifest configs/systematic-matrix.json \
  --results outputs/systematic-main-full.jsonl \
  --summary outputs/systematic-main-audit-summary.json \
  --report outputs/systematic-main-audit-report.md
```

The audit rejects missing or duplicate experiment IDs, schema or runtime mismatches, invalid locks or artifacts, inconsistent installed environments, failed cleanup, and outcome/stage contradictions. Measured dependency, import and smoke failures remain valid dataset evidence.

Collect the full version/Python PyPI catalog before running the pilot:

```bash
PYTHONPATH=src python3 -m deplab catalog \
  --scope configs/package-scope.json \
  --output outputs/package-catalog.jsonl
```

## Deterministic changelog signals

DepLab collects official, version-pinned release notes once per distinct package release. It stores source URLs and SHA-256 hashes, selects only the matching release series, and extracts deterministic counts and flags for breaking changes, removed or deprecated APIs, ABI changes, dependency compatibility, Python support, and wheel/build changes. It does not send changelogs to an LLM.

```bash
PYTHONPATH=src python3 -m deplab changelogs \
  --scope configs/package-scope.json \
  --sources configs/changelog-sources.json \
  --output outputs/changelog-catalog-v1.1.0.jsonl
```

The output is append-only and resumable. Its cost grows with unique package releases, not experiment combinations: a 5,000-row matrix containing 200 distinct releases needs approximately 200 release records, not 5,000 changelog downloads.

Join the release-level signals to a feature table:

```bash
python scripts/build_changelog_features.py \
  --features outputs/deplab-systematic-v1.0.0/features.csv \
  --changelogs outputs/changelog-catalog-v1.1.0.jsonl \
  --output outputs/deplab-systematic-v1.0.0/features-with-changelogs-v1.1.0.csv \
  --summary outputs/deplab-systematic-v1.0.0/changelog-feature-summary-v1.1.0.json
```

Changelog features are public pre-run inputs. Experiment errors, outcomes, logs, and installed-environment evidence remain excluded to prevent label leakage.

## Reproducible commands

The experiment runner requires Linux and [uv](https://docs.astral.sh/uv/). The unit suite needs only Python 3.10+ and does not access the network.

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Inspect wheel coverage without running an experiment:

```bash
PYTHONPATH=src python -m deplab inspect requests==2.32.3 --python 3.11
```

Run the first real vertical slice on Linux:

```bash
PYTHONPATH=src python -m deplab run requests==2.32.3 urllib3==2.2.2 \
  --python 3.11 --output outputs/results.jsonl
```

Re-running the same command reports `already_completed`. Pass `--force` only when intentionally replacing the append-only experiment policy during development.

Run the six-observation EC2 rehearsal sequentially:

```bash
PYTHONPATH=src python3 -m deplab batch \
  --manifest configs/rehearsal.json \
  --output outputs/rehearsal.jsonl \
  --workers 1
```

Use one worker on a `t3.medium`. The manifest deliberately combines a pure-Python pair with a native-wheel NumPy/Pandas pair across all three target Python versions.

## Cold and warm cache measurement

Use `--cache-scope experiment` to give every experiment its own cache. An empty cache directory makes the first batch cold. Run the same manifest again with the same cache directory but a different output file to measure the warm case. A populated directory alone is not proof that a particular wheel was cached, so DepLab records exact cache byte and file-count changes rather than guessing.

Each schema 1.3 observation also stores before/after host memory, disk, load and network counters, plus sampled peak process-tree RSS for every stage. Its `installed_wheel_artifacts` list records the exact filename, URL, size and SHA-256 for every installed package, including transitive dependencies. Host counters may include unrelated machine activity; use one worker on an otherwise idle machine for the first measurement.

For the systematic run, `--cleanup-environments` removes only the experiment's temporary `.venv` after evidence and resource measurements are captured. The small `requirements.in`, `pylock.toml`, smoke script, JSONL result, and shared wheel cache remain available for auditing and resume.

After completing repetition 01 manually, collect repetitions 02 and 03 with:

```bash
bash scripts/run_cache_benchmark.sh 2 3
```

The script refuses to overwrite an existing cache or result file. Each repetition gets a new cache root; its cold and warm runs share only that repetition's matching per-experiment caches.

## Evidence labels

- `measured: true` means the runner actually attempted the declared environment; preflight coverage and host/tool failures remain `false`.
- Unit tests use deterministic fixtures and never constitute compatibility data.
- `wheel_unavailable` is a coverage result, not a negative compatibility label.
- The example configuration is a pipeline validation case, not a benchmark dataset.
