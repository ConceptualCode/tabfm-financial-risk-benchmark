# TabFM Financial Risk Benchmark

Do zero-shot tabular foundation models — Google's
[TabFM](https://huggingface.co/google/tabfm-1.0.0-pytorch) and SAP's
[sap-rpt-1-oss](https://huggingface.co/SAP/sap-rpt-1-oss) — hold up against
tuned gradient boosting on real financial risk tasks? Not just on accuracy,
but on the things that decide whether a model ships to production:
calibration, inference cost, and explainability.

See [`OBJECTIVES.md`](OBJECTIVES.md) for the full thesis, research questions,
and scope.

## Setup

Requires **Python 3.11+** (`tabfm` on PyPI has `requires_python >= 3.11`;
on 3.10 or older, `pip install tabfm` fails with a misleading
"no matching distribution" error rather than a clear version message).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`sap_rpt` requires HuggingFace access approval — request access at
[SAP/sap-rpt-1-oss](https://huggingface.co/SAP/sap-rpt-1-oss), then
`huggingface-cli login` (or set `HF_TOKEN`) before running it. Never commit
that token — it's covered by `.gitignore` (`.env*`, `*.token`), but keep it
out of code and configs too.

**Hardware note:** TabFM's checkpoint alone is ~6.5GB — needs roughly
12-16GB+ available RAM to load and run, no GPU required. `sap_rpt`'s weights
are tiny (~65MB) but its default settings (`bagging=8`,
`max_context_size=8192`) can need up to ~80GB of GPU memory for inference-time
activations at scale; reduce `bagging`/`max_context_size` in
`models.py::build_sap_rpt` for smaller hardware. Check `free -h` (or
`nvidia-smi` for GPU memory) before running the full grid.

## Data

Benchmarked on [FinBench](https://huggingface.co/datasets/yuweiyin/FinBench)
(10 binary-classification tasks: credit-card default, loan default,
credit-card fraud, customer churn). Downloaded automatically via
`huggingface_hub` the first time you run the benchmark -- FinBench ships as
a legacy HF "dataset script," which current `datasets` versions no longer
support, so `data.py` fetches the underlying per-config `.npy` files
directly instead of using `datasets.load_dataset`.

## Usage

```bash
# full suite, all models
python scripts/run_benchmark.py

# a quick subset
python scripts/run_benchmark.py --datasets cd1 ld1 --models tabfm sap_rpt xgboost --skip-shap

# missing-data robustness sweep (RQ7): degrades test-time features to 0/5/20/50% missing
python scripts/run_robustness.py

# verify a robustness.csv run actually covers the full expected (dataset, model, missing_rate) grid
python scripts/check_robustness_coverage.py
```

### Real hyperparameter search for the classical baselines

`run_benchmark.py`/`run_robustness.py` run XGBoost, LightGBM, and logistic
regression at a single fixed config. `tune_classical.py` replaces that with
a genuine per-dataset random search (uses the FinBench validation split,
already loaded by `data.py` but otherwise unused), then re-evaluates on the
held-out test set. CPU only, no GPU needed.

```bash
# hyperparameter search + re-evaluation for XGBoost/LightGBM/logreg, all 10 datasets
python scripts/tune_classical.py

# a quick subset, fewer trials
python scripts/tune_classical.py --datasets cd1 ld1 --models xgboost --n-trials 10

# re-run the missingness robustness sweep using the tuned configs above
python scripts/run_robustness_tuned.py
```

Writes to `results/results_tuned.csv`, `results/tuned_best_params.csv`
(the winning hyperparameters found per dataset/model), and
`results/robustness_tuned.csv` — new files, none of the original
`results/*.csv` are overwritten.

### Regenerating figures

```bash
python scripts/generate_paper_figures.py              # original 5-model figure set
python scripts/generate_article_figures_tabfm_only.py  # TabFM vs. tuned-classical figure set
```

Results are written to `results/results.csv` (predictive metrics,
calibration points, fit/inference cost, cost-minimizing threshold, and
fairness metrics where applicable), `results/shap_agreement.csv`
(cross-model explainability agreement), `results/prediction_agreement.csv`
(cross-model prediction agreement -- do models with similar accuracy
actually agree on individual applicants?), and `results/robustness.csv`
(performance at increasing test-time missing-data rates).

## Project layout

```
configs/datasets.yaml     FinBench task list, model registry, protected attributes
src/tabfm_bench/
  data.py                 FinBench loader (raw DataFrame + pre-encoded array per split)
  models.py                TabFM + SAP-RPT + XGBoost + LightGBM + logreg, common interface
  metrics.py                AUC/PR-AUC/log-loss/Brier + calibration curve + cost-minimizing threshold
  cost.py                   fit/inference wall-clock + peak memory
  explain.py                 SHAP wrapper + cross-model SHAP agreement
  fairness.py                disparate impact ratio + equalized-odds gap
  agreement.py                cross-model prediction agreement
  robustness.py               missing-data injection for the RQ7 degradation test
  run.py                     runs one (dataset, model) pair end to end
scripts/run_benchmark.py             CLI: runs the full grid, writes results/
scripts/run_robustness.py            CLI: missing-data robustness sweep, writes results/robustness.csv
scripts/check_robustness_coverage.py CLI: verifies a robustness.csv run covers the full expected grid
scripts/tune_classical.py            CLI: real per-dataset hyperparameter search for XGBoost/LightGBM/logreg
scripts/run_robustness_tuned.py      CLI: robustness sweep using tune_classical.py's tuned configs
scripts/generate_paper_figures.py              regenerates the original 5-model figure set
scripts/generate_article_figures_tabfm_only.py regenerates the TabFM-vs-tuned-classical figure set
tests/                     smoke tests (no network/heavy deps required)
```
