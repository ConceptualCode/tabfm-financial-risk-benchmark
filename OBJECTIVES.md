# TabFM Financial Risk Benchmark — Project Objectives

## Thesis

Do zero-shot tabular foundation models (TabFM) hold up against tuned gradient
boosting on real financial risk tasks — not just on accuracy, but on the
things that actually decide whether a model ships to production?

## Primary Objective

Produce a rigorous, reproducible benchmark of TabFM vs. established baselines
on the FinBench suite (credit default, loan default, fraud, churn), evaluated
across predictive performance, calibration, operational cost, and
explainability — and publish honest findings, including where TabFM loses.

## Research Questions

### 1. How does TabFM's zero-shot performance compare to tuned XGBoost/LightGBM across all 10 FinBench datasets?

The foundational question. Foundation models are marketed as "no more manual
tuning," but that claim only matters if it holds up against models that *are*
tuned — the comparison a hiring manager would actually make. Running it
across all 10 datasets (not just one favorable one) guards against
cherry-picking and shows the benchmark's credibility comes from breadth, not
a single good number.

### 2. Is TabFM well-calibrated, or does it need post-hoc calibration?

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
sounds free, but TabFM processes the entire context (training set) on every
prediction call — so the cost a GBM pays once, upfront, during training,
TabFM may pay repeatedly, at inference time. If true, "zero-shot" doesn't
mean "cheaper," it means "cost moved elsewhere" — exactly the kind of nuance
teams miss when adopting new foundation models without checking total cost
of ownership.

### 4. Does SHAP work on TabFM the way it does on GBMs?

The highest-upside, most differentiated question — likely nobody else has
written about it yet, which is what makes a portfolio piece get noticed
instead of blending into the pile of "I benchmarked model X" posts.
Explainability tooling was built assuming tree-based or simple parametric
models; a 24-block causal transformer doing in-context learning is a very
different computational object, and it's an open question whether SHAP's
assumptions (feature independence approximations, background datasets, etc.)
transfer cleanly. A concrete finding — even "it runs but attributions are
unstable/misleading" — is a real, citable, non-obvious result.

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
- Baselines: XGBoost, LightGBM, and a logistic regression floor
- Metrics: AUC-ROC, PR-AUC, log-loss, Brier score / calibration curves,
  cost-weighted metric, inference latency/memory
- Explainability comparison (SHAP feasibility + agreement across models)
- Deliverables: public GitHub repo (clean, reusable eval harness), written
  narrative/blog post, small interactive demo

## Scope — Out

- No TabFM fine-tuning (it's zero-shot by design)
- No multi-table/relational datasets (e.g., full Home Credit with bureau
  data) — single-table tabular only
- No production deployment/serving system — benchmarking only
- No hyperparameter search beyond sane defaults for baselines (not a
  "who can tune harder" contest)

## Definition of Done

Repo runs end-to-end from a single command per dataset. Results table,
calibration plots, and SHAP comparison are reproducible. A written narrative
delivers a clear, honest verdict — not just a leaderboard.
