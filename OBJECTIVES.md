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

## Related Work (read before treating any finding as novel)

This is a fast-moving space -- TabFM shipped days before this project
started -- and parts of it are already covered elsewhere. Checked directly
rather than assumed:

- ["Evaluating SAP RPT-1 for Enterprise Business Process Prediction"](https://arxiv.org/abs/2602.19237)
  already benchmarks SAP-RPT-1-OSS against tuned XGBoost/LightGBM/CatBoost
  on SAP's own financial-risk business scenarios (not FinBench). RQ1's
  SAP-RPT-vs-GBM comparison substantially overlaps with this paper --
  differentiate by citing it and by using FinBench (an independent,
  reproducible academic benchmark) instead of SAP's internal scenarios.
- ["High Performance, Low Reliability: Uncertainty Benchmarking for Tabular
  Foundation Models"](https://arxiv.org/pdf/2605.28554) already covers
  general calibration/uncertainty benchmarking for tabular FMs -- read this
  before finalizing RQ2's narrative; don't duplicate its methodology
  uncritically.
- Explainability critiques of TabFM already exist (commentary + a proposed
  purpose-built model, ShapPFN, integrating Shapley values directly into
  the architecture) -- RQ4 should be framed as "does SHAP transfer to
  these two specific models on financial risk data," not "is this an
  unexplored question."

**What's still a genuinely distinctive combination:** both foundation
models together (not one), on FinBench specifically, plus the licensing
finding above, plus RQ5-8 below (fairness, the preprocessing-honesty
finding, missing-data robustness, and dataset-size sensitivity), none of
which turned up in this search.

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

Explainability tooling was built assuming tree-based or simple parametric
models; in-context-learning transformers are a very different computational
object, and it's an open question whether SHAP's assumptions (feature
independence approximations, background datasets, etc.) transfer cleanly to
either of them.

**Answer for this article, established through direct experience rather
than a planned experiment:** SHAP is either infeasible or prohibitively
expensive on realistic hardware for both models. TabFM's SHAP step OOMs
even at the minimum ensemble size (`n_estimators=1`) on an 8GB GPU, and only
survives on a 16GB T4 after shrinking the ensemble 8x (32 -> 4). SAP-RPT's
SHAP technically completes but costs ~45-71 minutes per dataset, versus
seconds for the GBMs. In practice this collapsed into a cost finding
overlapping with RQ3 -- "does it run at all, and at what price" -- rather
than the deeper question below.

**Deferred to a follow-up phase, after this article is published, once
there's budget for real GPU time (e.g. Colab Pro+/A100):** whether the
explanations SAP-RPT's SHAP *does* produce are actually stable and
trustworthy, or noisy/arbitrary compared to a GBM's -- and whether TabFM's
SHAP can be evaluated at all with proper hardware headroom instead of a
crippled ensemble size. This needs full, unconstrained runs to answer
properly rather than more workarounds on limited hardware. See "Future
Work" below.

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

### 5. Do TabFM and SAP-RPT encode different bias patterns than a GBM trained on your own data?

TabFM is trained on synthetic SCM-generated data; SAP-RPT is trained on
scraped real-world tables (T4/TabLib). A GBM is trained fresh, only on your
own population. Whether those very different training corpora produce
different fair-lending outcomes -- not just different accuracy -- is a real,
untested question none of the related work above addresses (each existing
benchmark evaluates one model's performance, not cross-model bias).
Evaluated via the standard disparate-impact ratio (US EEOC "four-fifths
rule": ratio under 0.8 flags potential concern) and equalized-odds gap
(difference in true/false-positive rates across groups), on the FinBench
configs with a clean binary protected attribute (`gender`, present and
verified binary in 6 of the 10 configs). Directly relevant to the SHAP/ECOA
framing already established in RQ4 -- a model can be technically explainable
and still produce disparate outcomes.

### 6. Is the "zero preprocessing needed" pitch actually true in practice?

Both models are marketed on skipping manual feature engineering — just feed
raw data, no one-hot encoding or scaling required. Building this benchmark
surfaced real, concrete counter-evidence, not a theoretical objection:
`cf2`'s own metadata had duplicate column names that broke naive DataFrame
reconstruction; numeric columns silently lost their dtype not once but
*twice* in the pipeline (once in our own raw-DataFrame reconstruction, again
inside `shap.KernelExplainer` itself); SAP-RPT's own tokenizer threw dtype
warnings on data that had already been carefully, correctly typed. None of
these are exotic edge cases — they're exactly the kind of friction any team
would hit trying to actually implement "raw input" the way the model cards
describe it.

**The finding:** "no manual feature engineering" is a real and valuable
claim (you genuinely don't need to hand-encode categoricals or scale
numerics) — but it is not the same claim as "no engineering effort."
Correctly serving these models well-typed raw data required real,
non-obvious debugging work that a team adopting either model would have to
rediscover for themselves, since it isn't mentioned in either model's
documentation.

### 7. How gracefully does each model handle incomplete data at prediction time?

Real applicants and transactions in production often have incomplete
profiles — a field wasn't collected, a system didn't report a value in
time. TabFM and SAP-RPT both advertise automatic handling of missing data
as a built-in feature; XGBoost and LightGBM also handle missing values
natively via learned split directions. Tested by progressively masking an
increasing fraction of *test-time* features (5%, 20%, 50%) to missing while
keeping training data intact — simulating a real scenario where a subset of
applicants show up with incomplete data at scoring time, not a data-quality
problem baked into training. A model that degrades gracefully under
increasing missingness is more production-ready than one that falls apart
the moment a field is absent, regardless of how good its clean-data
accuracy looks.

### 8. Does the zero-shot "advantage" shrink as dataset size grows?

FinBench's 10 datasets vary meaningfully in size (row counts from ~2,700 to
5,400+, feature counts from 9 to 120). Zero-shot in-context learning models
are commonly assumed to have an edge on small datasets, where gradient-
boosted trees have less data to learn stable splits from — and TabFM's own
model card notes "memory scales with training row count" as a real
constraint. Correlating each model's relative performance (against the best
baseline) with each dataset's size, once the full 10-dataset run is
complete, tests whether that assumed advantage is real, and whether it
shrinks or reverses as datasets grow. Needs no new infrastructure — it's a
post-hoc analysis of results already being collected for RQ1.

### Why this set as a whole

Questions 1-3 establish rigorous, unbiased evaluation — the baseline
expectation for any ML role. Questions 4-8 are what make the project
memorable rather than merely competent — each pushes past what the existing
literature (see Related Work) has already covered on these two models, and
several (6, 7) come directly from friction hit while actually building this,
not from a planned experiment design.

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
- Explainability comparison (SHAP feasibility across models)
- Fair-lending audit (disparate impact ratio, equalized-odds gap) on the 6
  FinBench configs with a clean binary protected attribute
- Missing-data robustness: performance degradation as an increasing fraction
  of test-time features are masked (5%/20%/50%)
- Dataset-size sensitivity: post-hoc correlation of relative performance
  against each dataset's size (no new infrastructure needed)
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

## Future Work (planned follow-up, not part of this article)

Unlike "Scope — Out" above (things deliberately not pursued), this is work
that's genuinely planned, just sequenced after the current deliverable and
gated on investing in paid GPU time (e.g. Colab Pro+/A100) rather than
fighting free-tier hardware limits:

- **RQ4 deep dive**: whether SAP-RPT's SHAP explanations are actually
  stable/trustworthy (not just "did it finish running"), and evaluating
  TabFM's SHAP at a real ensemble size instead of a crippled one. Needs
  unconstrained GPU headroom to do properly.
- Repeated-run/bootstrap confidence intervals (currently deferred, see
  Scope — Out) would benefit from the same GPU-budget investment, since
  each foundation-model rerun is expensive.
- Determinism/consistency check on repeated foundation-model calls
  (currently deferred, see Scope — Out) — same reasoning.

## Definition of Done

Repo runs end-to-end from a single command per dataset. Results table,
calibration plots, and SHAP comparison are reproducible. A written narrative
delivers a clear, honest production recommendation — including the
licensing constraint above — not just a leaderboard.
