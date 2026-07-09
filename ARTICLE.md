# TabFM and SAP-RPT Outperform XGBoost on Credit Risk — But Neither Can Be Deployed Commercially

Zero-shot tabular foundation models — pretrained transformers that generate predictions on structured data without task-specific training — are being positioned as a replacement for gradient-boosted trees in production ML pipelines. I benchmarked two of them, Google's TabFM (released last week) and SAP's SAP-RPT, against tuned XGBoost, LightGBM, and logistic regression on FinBench, a 10-dataset suite spanning credit default, loan default, fraud, and churn prediction.

One finding overrides everything else below: **neither model's pretrained weights are licensed for commercial use.** TabFM ships under a non-commercial license. SAP-RPT's weights carry a research-only restriction inherited from its training data (T4/TabLib). Both codebases are Apache 2.0 — irrelevant, since the code has no function without the weights, and retraining an equivalent model from scratch eliminates the zero-shot value proposition the entire category is built on. This is a hard constraint on production use, independent of the performance results that follow.

The rest of this piece is what happens when you run the benchmark anyway — because the performance, cost, and reliability findings are still useful for anyone tracking where this category is headed, licensing aside.

## Performance: does zero-shot hold up against tuned gradient boosting?

On the 6 datasets TabFM completed, both foundation models outperformed both tuned GBMs on every metric measured:

| Model | AUC-ROC | PR-AUC | Brier score |
|---|---|---|---|
| TabFM | 0.850 | 0.630 | 0.076 |
| SAP-RPT | 0.845 | 0.594 | 0.081 |
| LightGBM | 0.816 | 0.558 | 0.087 |
| XGBoost | 0.814 | 0.542 | 0.086 |
| Logistic regression | 0.803 | 0.494 | 0.091 |

This is not a dataset-selection artifact: GBM performance on this same 6-dataset subset (AUC 0.81–0.82) exceeds their performance on the full 10-dataset set (AUC 0.797–0.799) — the subset is slightly easier for every model, not specifically flattering to the foundation models.

TabFM completed 6 of 10 datasets. The 4 failures were CUDA out-of-memory errors concentrated on the widest datasets by column count — a direct consequence of how in-context learning scales, covered next.

## Cost: what "zero-shot" actually means at inference time

Methodology note: all runs executed on a free-tier Google Colab T4 GPU (16GB VRAM). Absolute failure thresholds are hardware-dependent; the relative cost gap between architectures is not.

TabFM's `fit()` call averaged 0.36 seconds because it performs no gradient-based training — the training set is retained as in-context examples and processed at inference time instead. Mean prediction time per dataset:

| Model | Mean fit time | Mean predict time |
|---|---|---|
| XGBoost | 1.24s | 0.057s |
| LightGBM | 1.44s | 0.172s |
| Logistic regression | 0.49s | 0.048s |
| SAP-RPT | 0.002s | 55.2s |
| TabFM | 0.36s | 362s |

Per-dataset, TabFM's prediction time was 17,000x–35,000x XGBoost's; SAP-RPT's was 264x–1,107x, with the ratio increasing directly with column count (264x at 9 columns, 1,107x at 35). This scaling is the direct cause of TabFM's 4 dataset failures: both models attend over the full training context on every prediction, so cost scales with context size in a way tree-based inference does not.

This compounds for explainability. SHAP requires hundreds of prediction calls per explained instance. At TabFM's per-call cost, this is not a slower operation — SHAP failed with an out-of-memory error on every configuration tested, including after an 8x ensemble-size reduction. SAP-RPT's SHAP step completed but required 45–71 minutes per dataset.

Zero-shot eliminates training cost. It does not eliminate compute cost — it relocates it to inference, at a substantially higher total.

## Fairness: does model complexity correlate with disparate impact?

Evaluated via disparate impact ratio (EEOC four-fifths rule: ratio <0.8 flags concern) and equalized-odds gap, on the 6 FinBench datasets with gender as a usable protected attribute.

| Model | Datasets checked | Flagged below 0.8 | Mean equalized-odds gap |
|---|---|---|---|
| LightGBM | 6 | 0/6 | 0.061 |
| XGBoost | 6 | 1/6 | 0.087 |
| SAP-RPT | 5 | 1/5 | 0.102 |
| TabFM | 4 | 1/4 | 0.158 |
| Logistic regression | 6 | 2/6 | 0.169 |

Two results stand out. One dataset (`cc1`, customer churn) produced a disparate-impact violation for every model except LightGBM; logistic regression's ratio (0.43) was roughly half the regulatory floor. Logistic regression had the worst overall fairness profile — 2 of 6 flagged, highest mean equalized-odds gap — not the foundation models. Model simplicity does not correlate with fairness here; a linear model encodes the same disparity as a more complex one, through a coefficient rather than a split.

TabFM's mean equalized-odds gap (0.158) is second-highest, based on 4 of 6 datasets (the other 2 failed for the reasons above) — a directional signal, not a confirmed result, given the sample size.

Passing an accuracy benchmark, or being technically explainable via SHAP, does not establish fairness. It is a separate, unrelated check, and here the simplest model failed it worst.

## Preprocessing: does "zero feature engineering" hold up in practice?

Both models market raw-data ingestion as a core feature. One FinBench dataset, `cf2`, contains an ordinary upstream data-quality issue: two distinct source columns collapsed to an identical human-readable name during preprocessing — a common occurrence, not a constructed edge case.

Feeding this dataset to either model in its documented raw format produced an identical crash: `AttributeError: 'DataFrame' object has no attribute 'dtype'`. This indicates internal column lookup by name, which returns multiple columns instead of one when names collide — neither model's internal code handles this, and neither model card documents it as a constraint. Both models are missing complete results on `cf2` as a direct result; every classical baseline, which performs no named-column lookups, ran on it without issue.

Two further defects surfaced independently while implementing the raw-input path: a numeric column silently lost its correct dtype twice — once during DataFrame reconstruction, again inside SHAP's `KernelExplainer`, which flattens a DataFrame to a single-dtype array before invoking the model. Both would have silently fed incorrectly-typed data to the model without raising an error.

"No feature engineering" is accurate. "No engineering effort" is not — the effort moves from building a preprocessing pipeline to debugging failures the documentation does not anticipate.

## What this doesn't cover

Two additional analyses (missing-data robustness across all 10 datasets, and performance sensitivity to dataset size) are in the full technical writeup and repo, omitted here for length. The headline from the robustness analysis: XGBoost's AUC degraded roughly twice as fast as LightGBM's under increasing test-time missingness (0.703→0.575 vs. 0.722→0.654 at 50% missing), despite both claiming native missing-value handling — a reminder that "handles missing data" is not a single, uniform property even among architecturally similar models.

## Limitations

- Main results: 45/50 (dataset, model) combinations completed. The 5 gaps are TabFM/SAP-RPT only, attributable to the GPU memory ceiling or the `cf2` defect above.
- Single train/test split per dataset; no bootstrap confidence intervals. Close rankings should be treated as directional.
- SHAP was evaluated for feasibility and cost, not attribution stability. Cross-model explanation agreement data exists for one dataset (`cd1`) only.
- No test of prediction consistency across repeated calls on identical input. SAP-RPT's bagging step introduces randomness; its effect on repeated scoring of the same applicant is untested.
- All experiments ran on one hardware configuration (Colab T4, 16GB). Relative cost findings should generalize; absolute failure points will not.
- This evaluation was conducted under the research-use exemption of both licenses. Production deployment is the scenario being evaluated, not the scenario this project operates under.

## Verdict

Absent the licensing constraint, this is a genuine trade-off, not a clear result in either direction. Both foundation models outperformed tuned GBMs on every accuracy and calibration metric, on every dataset completed. That performance carries a cost that is not tunable away: 17,000x–35,000x higher inference cost than XGBoost, a memory ceiling that made 4 of 10 datasets unrunnable, and an explainability step that is either infeasible or takes up to an hour per dataset. Add the fairness result and the preprocessing-effort finding, and the accurate summary is: more accurate, substantially more expensive, and less production-ready than the pitch suggests.

None of that is the deciding factor. **Neither model can be legally deployed in a commercial product today.** A model that cannot be shipped is a research artifact, not a production option.

- **Production system today:** tuned XGBoost or LightGBM. No licensing exposure, substantially lower inference cost, modest (not disqualifying) performance gap.
- **Evaluating the category:** the performance case justifies monitoring, and a pilot the moment commercial licensing is available.
- **Already committed to one of these models:** budget for inference cost as a first-class constraint, and do not assume SHAP is a low-cost addition for compliance purposes.

Full methodology, code, and results: [GitHub repo link].
