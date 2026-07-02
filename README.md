# TabFM Financial Risk Benchmark

Does Google's zero-shot tabular foundation model ([TabFM](https://huggingface.co/google/tabfm-1.0.0-pytorch))
hold up against tuned gradient boosting on real financial risk tasks — not
just on accuracy, but on the things that decide whether a model ships to
production: calibration, inference cost, and explainability?

See [`OBJECTIVES.md`](OBJECTIVES.md) for the full thesis, research questions,
and scope.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

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
python scripts/run_benchmark.py --datasets cd1 ld1 --models tabfm xgboost --skip-shap
```

Results are written to `results/results.csv` (predictive metrics, calibration
points, fit/inference cost) and `results/shap_agreement.csv`
(cross-model explainability agreement).

## Project layout

```
configs/datasets.yaml     FinBench task list + model registry
src/tabfm_bench/
  data.py                 FinBench loader
  models.py                TabFM + XGBoost + LightGBM + logreg, common interface
  metrics.py                AUC/PR-AUC/log-loss/Brier + calibration curve
  cost.py                   fit/inference wall-clock + peak memory
  explain.py                 SHAP wrapper + cross-model agreement
  run.py                     runs one (dataset, model) pair end to end
scripts/run_benchmark.py   CLI: runs the full grid, writes results/
tests/                     smoke tests (no network/heavy deps required)
```
