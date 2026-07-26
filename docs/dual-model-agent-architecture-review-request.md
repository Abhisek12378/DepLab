# Request for a Second Opinion: DepLab Dual-Model Agent Architecture

> Historical design note: the implemented production architecture no longer
> treats the structured model as a resolution gate. It ranks candidates, while
> `uv pip compile` makes the dependency-resolution decision. See
> `docs/production-agent-architecture.md` for the current design.

## What DepLab does

DepLab is a prediction system for Python dependency compatibility.

A user provides a `requirements.txt`, a target Python version and a question such as:

> Can I upgrade NumPy from 1.26.4 to 2.0.2?

DepLab should:

1. Understand the requested change.
2. Check relevant published dependency constraints.
3. Predict whether candidate package combinations will resolve and install.
4. Predict whether installed packages may fail during import or a package-specific smoke test.
5. Generate and rank safer complete environments.
6. Clearly separate deterministic facts from model predictions.

DepLab does **not** install packages during a live user request. Runtime installation and verification are deliberately outside the current product scope.

## Dataset

All experiments use Linux x86_64 with glibc. Each experiment contains two pinned packages and a pinned Python version.

Every experiment was executed in an isolated `uv` environment. Its measured outcome is one of:

- `pass`
- `resolution_failure`
- `import_failure`
- `smoke_test_failure`

Import and smoke-test results are labels only. They are not allowed as model inputs because they would not be available during inference.

### Development dataset

- Total experiments: **21,490**
- Package-pair families: **36**
- Pass: **12,031**
- Resolution failure: **6,410**
- Import failure: **2,448**
- Smoke-test failure: **601**

### Sealed validation dataset

The validation packages and package families were excluded from development. This tests generalization to package names not seen during training.

Validation packages include:

- `boto3`
- `botocore`
- `s3transfer`
- `celery`
- `kombu`
- `billiard`

Validation outcomes:

- Total experiments: **3,432**
- Pass: **639**
- Resolution failure: **2,714**
- Import failure: **60**
- Smoke-test failure: **19**

The final test dataset remains separate and is not part of this architecture decision.

## Inference-safe inputs

The models may use only information available before installation:

- Package and release metadata from PyPI
- Exact versions requested by the user
- Target Python version
- `Requires-Python`
- Published dependency declarations and version specifiers
- Wheel tags and wheel availability
- Release dates and classifiers
- Frozen, version-specific release-note or changelog text
- Pre-generated release-text embeddings

The models must not use:

- Installation outcome
- Import result
- Smoke-test result
- Installed environment
- Resolver output or error message
- Runtime timings
- Cache or machine measurements

## Model 1: Structured weighted-logistic model

This model uses structured metadata, constraints, Python compatibility, wheel information and other inference-safe numeric or categorical features.

Validation results:

| Metric | Result |
|---|---:|
| Accuracy | 90.01% |
| Balanced accuracy | 93.86% |
| Failure precision | 100.00% |
| Failure recall | 87.72% |
| Failure F1 | 93.46% |
| ROC AUC | 98.90% |
| False failure warnings | 0 |
| Missed failures | 343 |
| Pass recall | 100.00% |
| Resolution-failure recall | 90.27% |
| Import-failure recall | 0.00% |
| Smoke-test-failure recall | 0.00% |

Frozen threshold:

```text
0.6431916234021728
```

### Interpretation

The structured model is strong at detecting dependency-resolution problems without producing false warnings in this validation set. However, it did not identify any import or smoke-test failures.

## Model 2: ModernBERT stage-aware hybrid model

Release text was embedded offline using:

```text
answerdotai/ModernBERT-base
```

The encoder is not loaded during user requests. The production server reads previously generated release embeddings and runs a lightweight stage-aware model with separate heads for:

- Resolution failure
- Import failure
- Smoke-test failure

Validation results:

| Metric | Result |
|---|---:|
| Accuracy | 80.04% |
| Balanced accuracy | 67.34% |
| Failure precision | 87.86% |
| Failure recall | 87.58% |
| Failure F1 | 87.72% |
| ROC AUC | 83.04% |
| False failure warnings | 338 |
| Missed failures | 347 |
| Pass recall | 47.10% |
| Resolution-failure recall | 87.21% |
| Import-failure recall | 100.00% (60 of 60) |
| Smoke-test-failure recall | 100.00% (19 of 19) |

Frozen overall threshold:

```text
0.45767843402863695
```

### Interpretation

The ModernBERT stage-aware model detects the rare import and smoke-test failures that the structured model misses. However, using its overall failure decision directly produces too many false warnings and rejects many passing combinations.

## Current production state

Both frozen lightweight predictors are installed on a small AWS EC2 `t3.medium`.

The server currently has:

- Structured model artifact
- ModernBERT stage-aware model artifact
- Model input policy
- 593 pre-generated ModernBERT release embeddings

The full ModernBERT encoder is not required on the production server.

A separate offline job is generating the same ModernBERT embeddings for 14,148 stable releases across 100 popular packages.

The chatbot is not yet connected to these two new predictors. This review is intended to improve the orchestration policy before that connection is implemented.

## Proposed architecture

### Responsibility of GPT

GPT should:

- Parse the user's `requirements.txt` and natural-language question.
- Resolve references in follow-up questions using conversation memory.
- Produce a structured intent.
- Explain the deterministic and model results in simple language.

GPT should not:

- Invent candidate compatibility scores.
- Decide whether a published constraint allows a version.
- Control an unbounded retry loop.
- Describe a prediction as runtime-verified.

### Responsibility of the deterministic backend

The backend should:

- Generate complete candidate environments.
- Check published dependency constraints.
- Select the appropriate model stages.
- Score candidates in batches.
- Reject unsafe candidates.
- Rank surviving candidates.
- Enforce retry and candidate limits.
- Return a structured evidence record to GPT.

## Proposed sequential decision flow

### Stage 1: Parse intent

Convert the request into structured data:

```json
{
  "python_version": "3.11",
  "platform": "linux-x86_64-glibc",
  "current_requirements": {
    "numpy": "1.26.4",
    "pandas": "2.1.4",
    "scipy": "1.11.4"
  },
  "target_package": "numpy",
  "requested_version": "2.0.2",
  "action": "upgrade",
  "packages_that_must_not_change": []
}
```

### Stage 2: Generate candidate environments

Generate a bounded batch of complete alternatives rather than evaluating one candidate through an open-ended agent loop.

Example:

```text
Candidate A: numpy 2.0.2, pandas 2.2.2, scipy 1.13.1
Candidate B: numpy 1.26.4, pandas 2.1.4, scipy 1.11.4
Candidate C: numpy 1.26.4, pandas 2.2.2, scipy 1.12.0
```

Proposed limits:

- Maximum candidates evaluated: 50
- Maximum recommendations returned: 3

### Stage 3: Published-constraint gate

Check direct published constraints before using either model.

If metadata states:

```text
pandas 2.1.4 requires numpy < 2
```

then `numpy==2.0.2` is rejected as a deterministic constraint conflict.

The response must call this:

```text
BLOCKED by a published constraint
```

It must not call it a model prediction.

### Stage 4: Structured resolution-risk gate

Candidates that pass published constraints are scored by the structured model.

If the structured model predicts high failure risk, reject the candidate and move to the next candidate.

If it predicts low risk, pass the candidate to the ModernBERT stage.

The structured result remains a prediction. Low predicted risk is not proof that installation will succeed.

### Stage 5: ModernBERT post-install gate

For candidates that survive the structured gate, use only the relevant stage-aware outputs:

- Import-failure probability
- Smoke-test-failure probability

The ModernBERT resolution head would not be used as the main resolution decision because the structured model performed better for resolution failures.

If either post-install risk exceeds its calibrated threshold:

1. Record the package pair and likely failure stage.
2. Reject that exact candidate.
3. Return to the existing candidate batch.
4. Evaluate the next candidate.

The system should not reparse the user's request or restart GPT reasoning.

### Stage 6: Rank survivors

Proposed ranking order:

1. Achieves the user's requested change.
2. Has no deterministic constraint conflict.
3. Lowest import and smoke-test risk.
4. Lowest structured resolution risk.
5. Fewest package changes.
6. Smallest version movement.

Possible response groups:

- Achieves your requested change
- Keeps the current target version
- Fallback that does not achieve the requested goal

### Stage 7: Explain evidence

Every conclusion should preserve the distinction between:

- **Fact:** published constraints block the environment.
- **Prediction:** a model estimates that the environment may fail.
- **Unknown:** model coverage is missing.
- **Verified:** reserved for actual runtime installation testing, which the current system does not perform.

## Proposed pair-level result contract

```json
{
  "package_pair": "numpy-pandas",
  "constraint_status": "allowed",
  "constraint_conflicts": [],
  "structured_model": {
    "resolution_risk": 0.08,
    "threshold": 0.6431916234021728,
    "decision": "continue"
  },
  "modernbert_model": {
    "import_risk": 0.04,
    "smoke_test_risk": 0.02,
    "decision": "continue"
  },
  "final_decision": "recommended",
  "evidence_types": [
    "published_metadata",
    "structured_model_prediction",
    "modernbert_model_prediction"
  ],
  "runtime_verified": false
}
```

## Coverage policy

If a required structured feature row or release embedding is unavailable:

- Do not call the combination safe.
- Continue reporting any deterministic constraint facts that are available.
- Return `insufficient_model_coverage` for the missing prediction.
- Do not silently substitute a different embedding model.

All release embeddings used for training and inference must come from the exact same embedding model and preprocessing procedure.

## Important unresolved decision

The sequential cascade has not yet been evaluated as one combined policy.

Before connecting it to the live chatbot, we plan to replay all 3,432 sealed validation rows through the proposed cascade and measure:

- Overall accuracy
- Balanced accuracy
- Failure precision and recall
- Pass recall
- Resolution-failure recall
- Import-failure recall
- Smoke-test-failure recall
- False warnings
- Missed failures
- How many rows reach the ModernBERT stage
- Performance and latency saved by the cascade

The import and smoke-test thresholds should be chosen using validation results rather than arbitrary values.

## Questions for the reviewing AI

Please critically review this architecture and answer the following:

1. Is the deterministic constraint → structured model → ModernBERT post-install model cascade technically sound?
2. Are there cases where the ModernBERT stage should run even when the structured model predicts high resolution risk?
3. Should the structured model's binary failure probability be treated as a resolution gate, given that it was trained on all failure labels but mainly learned resolution failures?
4. What is the safest method for calibrating separate import and smoke-test thresholds when those validation classes contain only 60 and 19 examples?
5. How should the system combine pair-level risks into one complete-environment decision?
6. Should candidates be rejected when any pair crosses a threshold, or should the system use an aggregate environment-level score?
7. How can we reduce ModernBERT's false warnings without losing its strong import and smoke-test recall?
8. Is the proposed candidate ranking policy appropriate for an upgrade advisor?
9. What additional safety rules are needed for missing embeddings, unsupported package pairs or unsupported Python versions?
10. What failure modes could cause the retry loop to reject every candidate or recommend unnecessary downgrades?
11. What structured result fields should be returned to GPT so that it cannot confuse deterministic facts with model predictions?
12. What evaluation tests should be completed before enabling this architecture in production?

## Requested response format

Please provide:

1. Overall assessment
2. Strong parts of the design
3. Main technical risks
4. Recommended architecture changes
5. Recommended threshold and calibration strategy
6. Suggested pseudocode for the final orchestrator
7. Minimum test plan before production

Please be critical. We are specifically looking for incorrect assumptions, hidden failure modes and simpler alternatives.
