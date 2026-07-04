# TabFM Financial Risk Benchmark — Project Objectives

## Thesis

Do zero-shot tabular foundation models — Google's TabFM and SAP's
sap-rpt-1-oss — hold up against tuned gradient boosting on real financial
risk tasks, well enough to justify a real production decision — not just on
accuracy, but on the things that actually decide whether a model ships to
production? The deliverable is production advice grounded in experimental
evidence, not an academic leaderboard.

## Production Constraint: Licensing (read this first)

Before any benchmark result matters, both foundation models have a
checkpoint-level licensing restriction that blocks commercial deployment
outright, independent of performance:

| | Code license | Pretrained weights license |
|---|---|---|
| **TabFM** | Apache 2.0 (permissive) | **TabFM Non-Commercial License v1.0** |
| **SAP-RPT** | Apache 2.0 (permissive) | **Research-only** (inherited from the T4/TabLib training-data lineage) |

Confirmed against both HuggingFace model cards and both underlying GitHub
repos (`google-research/tabfm`, `SAP-samples/sap-rpt-1-oss`). The permissive
code license is a red herring — you can't use either model without its
checkpoint (retraining an equivalent from scratch erases the "zero-shot, no
training" value proposition entirely), and the checkpoint itself is
restricted to non-commercial/research use in both cases. XGBoost, LightGBM,
and logistic regression carry no such restriction: you train on your own
data and own the resulting model outright.

**Headline finding this produces:** regardless of benchmark performance,
neither TabFM nor SAP-RPT can be legally deployed in a commercial financial
product today. A team would need to either negotiate separate commercial
licensing directly with Google/SAP, or train an equivalent model from
scratch on properly-licensed data. This benchmark project itself is
squarely research use and unaffected; it's a live production deployment
serving real customer decisions that the license would block. This finding
should lead the final write-up, ahead of any accuracy/calibration numbers.

## Primary Objective

Produce a rigorous, reproducible benchmark of TabFM and SAP-RPT vs.
established baselines on the FinBench suite (credit default, loan default,
fraud, churn), evaluated across predictive performance, calibration,
operational cost, and explainability — and turn the results into an honest
production recommendation, including the licensing constraint above and
where the foundation models lose on the merits.

## Research Questions

### 1. How do TabFM and SAP-RPT's zero-shot performance compare to tuned XGBoost/LightGBM across all 10 FinBench datasets?

The foundational question. Foundation models are marketed as "no more manual
tuning," but that claim only matters if it holds up against models that *are*
tuned — the comparison a hiring manager would actually make. Running it
across all 10 datasets (not just one favorable one) guards against
cherry-picking and shows the benchmark's credibility comes from breadth, not
a single good number. Comparing *two* independently-built foundation models
against the same baselines also tests whether any finding is specific to one
architecture or a more general property of zero-shot tabular ICL.

### 2. Are TabFM and SAP-RPT well-calibrated, or do they need post-hoc calibration?

Accuracy/AUC only measure whether the model ranks risky vs. safe cases
correctly — they say nothing about whether "0.73 probability of default"
actually means 73% of those cases default. In finance, calibration is what
risk scoring, pricing, and capital-reserve decisions are built on. A model
can have great AUC and still be unusable in production if its probabilities
are distorted. Most portfolio projects stop at accuracy; including
calibration signals an understanding of what "good enough for production"
really means.

**Calibration, concretely:** gather every case where the model predicted
~0.73. If roughly 73% of them actually defaulted, the model is
well-calibrated. If only 40% did, the model is overconfident — it may still
rank cases correctly (good AUC) while the actual numbers are meaningless for
anything beyond ranking. Checked visually with a **reliability diagram**
(predicted probability vs. observed frequency per bin — perfect calibration
traces the diagonal) and numerically with **Brier score** (mean squared
error between predicted probability and actual 0/1 outcome; lower is
better).

### 3. What's the real inference cost of in-context learning vs. a pre-trained GBM?

This attacks the model's core marketing claim directly. "No training needed"
sounds free, but both TabFM and SAP-RPT process the entire context (training
set) on every prediction call — so the cost a GBM pays once, upfront, during
training, they may pay repeatedly, at inference time. If true, "zero-shot"
doesn't mean "cheaper," it means "cost moved elsewhere" — exactly the kind of
nuance teams miss when adopting new foundation models without checking total
cost of ownership. This question also surfaces a real, concrete finding from
scaffolding this project: TabFM's checkpoint alone needs ~12-16GB of RAM to
load (no GPU required), while SAP-RPT's weights are tiny (~65MB) but its
default settings can need up to ~80GB of GPU memory for inference-time
activations — two very different cost profiles hiding behind the same
"zero-shot, no training" pitch.

### 4. Does SHAP work on TabFM and SAP-RPT the way it does on GBMs?

The highest-upside, most differentiated question — likely nobody else has
written about it yet, which is what makes a portfolio piece get noticed
instead of blending into the pile of "I benchmarked model X" posts.
Explainability tooling was built assuming tree-based or simple parametric
models; in-context-learning transformers are a very different computational
object, and it's an open question whether SHAP's assumptions (feature
independence approximations, background datasets, etc.) transfer cleanly to
either of them. A concrete finding — even "it runs but attributions are
unstable/misleading" — is a real, citable, non-obvious result, and comparing
two architecturally-different foundation models strengthens the claim if
both show the same weakness.

**SHAP (SHapley Additive exPlanations), concretely:** explains an individual
prediction by quantifying how much each input feature pushed that specific
prediction up or down. Built on Shapley values (cooperative game theory):
imagine features as "players" joining the prediction one at a time, in every
possible order, and measure each feature's average marginal contribution
across all orderings. The result is a signed number per feature per
prediction (e.g., "income: +0.12, missed payment history: +0.30") that sums
exactly to the gap between the model's prediction and its baseline average.
It's the standard explainability tool in regulated finance because laws like
the US Equal Credit Opportunity Act require lenders to give applicants a
specific reason for a denial — SHAP is what generates that reason from an
otherwise black-box model. Whether it transfers cleanly to TabFM is directly
relevant to real-world adoption.

### Why this set as a whole

Questions 1-3 establish rigorous, unbiased evaluation — the baseline
expectation for any ML role. Question 4 is what makes the project memorable
rather than merely competent — the difference between "solid portfolio
project" and "the project a hiring manager remembers after 50 other
candidates."

## Scope — In

- Full FinBench suite (10 datasets: credit default, loan default, fraud, churn)
- Foundation models: TabFM and SAP-RPT (sap-rpt-1-oss), fed raw/native input
  (real column names, real category values) matching how each is actually
  documented to be used in production, not the same pre-encoded array fed to
  the classical baselines
- Baselines: XGBoost, LightGBM (categorical-aware), and a logistic regression
  floor (one-hot encoded categoricals) — realistic production-grade
  preprocessing for each, not a strawman
- Fixed random seeds on all classical baselines for reproducibility
- Metrics: AUC-ROC, PR-AUC, recall, F1, log-loss, Brier score / calibration
  curves, cost-weighted score with an actual threshold sweep to find the
  cost-minimizing operating point (not just a fixed 0.5 cutoff), inference
  latency/memory
- Explainability comparison (SHAP feasibility + agreement across models)
- Deliverables: public GitHub repo (clean, reusable eval harness), written
  narrative/production recommendation, small interactive demo

## Scope — Out

- No fine-tuning of either foundation model (both are zero-shot by design)
- No multi-table/relational datasets (e.g., full Home Credit with bureau
  data) — single-table tabular only
- No production deployment/serving system — benchmarking only
- No hyperparameter search beyond sane defaults for baselines (not a
  "who can tune harder" contest)
- No repeated-run/bootstrap confidence intervals in this pass — single-split
  results should be treated as directional, not proof, for any two models
  that land close together. Documented as a caveat in the write-up rather
  than built now.
- No per-request (P50/P99) latency breakdown — cost.py measures whole-batch
  fit/predict time, which conflates a live-API scoring scenario with a batch
  scoring scenario. Noted as a limitation, not built now.
- No determinism/consistency check on repeated foundation-model calls
  (relevant given SAP-RPT's bagging) — a real production/fair-lending
  question, deferred to a later pass.

## Definition of Done

Repo runs end-to-end from a single command per dataset. Results table,
calibration plots, and SHAP comparison are reproducible. A written narrative
delivers a clear, honest production recommendation — including the
licensing constraint above — not just a leaderboard.
