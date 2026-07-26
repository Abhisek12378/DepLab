# DepLab production agent

## Purpose

DepLab answers upgrade questions without installing packages during a live
request. It separates four kinds of evidence so the chatbot cannot describe a
prediction as a fact.

## Request flow

1. GPT parses the user's `requirements.txt`, target package and requested
   version into a strict schema. Backend validation rejects invented packages,
   non-exact pins and unsupported targets.
2. Published direct constraints are checked first. A conflicting requirement is
   a deterministic fact and is labelled `BLOCKED`.
3. The structured model scores at most 50 complete candidate environments. Its
   score is used only for ranking; it is not accepted as proof that a candidate
   resolves.
4. `uv pip compile` checks at most the top 5 candidates for Linux x86_64 and the
   requested Python version. The command is wheel-only, has a 15-second timeout,
   uses a two-request concurrency limit and caches identical checks for 15
   minutes. It resolves dependencies but does not install packages.
5. Only resolver-successful candidates reach the ModernBERT stage-aware model.
   The model uses frozen release embeddings and predicts import or smoke-test
   risk. The full ModernBERT encoder is not loaded in production.
6. DepLab returns at most 3 complete environments, ordered by goal alignment,
   number of changes and predicted post-install risk.
7. GPT explains the structured result. It cannot generate candidates, override
   the resolver or change evidence labels.

## Evidence labels

- `published_constraint_conflict`: deterministic metadata fact.
- `uv_resolution`: resolver result for the complete pinned environment.
- `structured_ranking_signal`: internal candidate-ordering score only.
- `post_install_prediction`: ModernBERT import/smoke prediction.
- Missing features or embeddings: unknown coverage, never called safe.

`resolver_verified` means dependency resolution was checked. It does not mean
the environment was installed or runtime-verified.

## Model data and validation

The production candidates were trained on 21,490 development experiments:

- 12,031 passes
- 6,410 resolution failures
- 2,448 import failures
- 601 smoke-test failures

The frozen validation set contains 3,432 experiments. The structured candidate
reached 90.01% accuracy and 93.86% balanced accuracy, but detected none of the
import or smoke-test failures; therefore it is not used for the post-install
decision. On the 718 validation rows that resolved, the calibrated ModernBERT
post-install policy detected 70 of 79 import/smoke failures:

- Recall: 88.61%
- Precision: 44.59%
- Specificity: 86.38%
- Balanced accuracy: 87.50%
- Frozen combined threshold: 0.4726886805608258

These are validation results, not a guarantee for every package family.

## Safety and runtime limits

- No shell execution: `uv` receives a validated argument list.
- 1 to 100 exact package pins per resolver request.
- 50 generated candidates, 5 resolver checks and 3 returned recommendations.
- Two concurrent resolver processes per API instance.
- Resolver output is cached; temporary input and lock files are removed.
- No package installation and no execution of package code.
- Unsupported platforms, resolver infrastructure errors and missing model
  coverage return an explicit unknown state.
