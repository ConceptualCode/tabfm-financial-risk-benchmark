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
credit-card fraud, customer churn). Downloaded automatically via the
`datasets` library the first time you run the benchmark.

## Usage

```bash
# full suite, all models
python scripts/run_benchmark.py

# a quick subset
python scripts/run_benchmark.py --datasets cd1 ld1 --models tabfm sap_rpt xgboost --skip-shap
```

Results are written to `results/results.csv` (predictive metrics, calibration
points, fit/inference cost) and `results/shap_agreement.csv`
(cross-model explainability agreement).

## Project layout

```
configs/datasets.yaml     FinBench task list + model registry
src/tabfm_bench/
  data.py                 FinBench loader
  models.py                TabFM + SAP-RPT + XGBoost + LightGBM + logreg, common interface
  metrics.py                AUC/PR-AUC/log-loss/Brier + calibration curve
  cost.py                   fit/inference wall-clock + peak memory
  explain.py                 SHAP wrapper + cross-model agreement
  run.py                     runs one (dataset, model) pair end to end
scripts/run_benchmark.py   CLI: runs the full grid, writes results/
tests/                     smoke tests (no network/heavy deps required)
```
