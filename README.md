# HR Attrition Prediction - COMP10086 Artificial Intelligence Applications Project

This repository is the project for **COMP10086 Artificial Intelligence Applications**. It builds a machine learning solution that predicts employee attrition on the IBM HR Analytics dataset (fictional, 1,470 employees, around 16% leavers). The risk scores are meant to support, not replace, human retention conversations. See `docs/problem_framing.md` for the problem framing, scope and success criteria.

## Project Structure

The repository is organised into the following folders:

```text
hrattrition/
│
├── .github/workflows/       # CI (GitHub Actions)
├── config/                  # Configuration (config.yaml)
├── data/
│   ├── raw/                 # Original dataset (not committed)
│   └── processed/           # Cleaned data and batch scores (not committed)
├── docs/                    # Problem framing, figures and results
├── models/                  # Saved model (not committed), metrics.json, fairness.json
├── notebooks/               # Jupyter notebooks (01_eda.ipynb)
├── src/                     # Pipeline scripts and API
├── tests/                   # Automated tests
│
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
└── requirements-serve.txt
```

## Folders

### `config/`

`config.yaml` is the single source of truth for paths, the random seed, the train/test split, cross-validation folds, success criteria and the fairness groups (including the age bands `<30`, `30-45`, `>45`).

### `data/`

Data used during the project. The original dataset goes in `data/raw/`. Keep the raw data unchanged. `data/processed/` holds the cleaned data and batch scores produced by the scripts.

### `docs/`

Documentation for the project, including the problem framing, figures and results tables produced by the notebook and the scripts.

### `models/`

The trained model (`model.joblib`, not committed) and the committed results files `metrics.json` and `fairness.json`.

### `notebooks/`

Jupyter notebooks for exploring the data. `01_eda.ipynb` is the exploratory data analysis and saves its figures to `docs/`.

### `src/`

The main Python code for the project: the pipeline scripts, the shared config loader and the API.

### `tests/`

Tests for the Python code, run with pytest.

A simple way to think about the split between `notebooks/` and `src/` is:

> **Use notebooks to explore. Use `src/` for code you want to keep and reuse.**

## Dataset

The project uses the **IBM HR Analytics Employee Attrition & Performance** dataset (Kaggle file `WA_Fn-UseC_-HR-Employee-Attrition.csv`).

Rename the file and place it here:

```text
data/raw/hrattrition.csv
```

> **Please don't edit the original CSV file.** Any cleaning or preprocessing is done in code (`src/preprocess.py`).

## How to run

All commands are run from the repository root. The examples are for Windows (PowerShell or cmd).

### 1. Set up the environment

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

On Linux / macOS activate with `source .venv/bin/activate` instead.

### 2. Add the data

Put the raw CSV in `data/raw/hrattrition.csv` (see Dataset above).

### 3. Run the pipeline (in this order)

```powershell
python src/preprocess.py
python src/features.py
python src/train.py
python src/evaluate.py
python src/fairness.py
python src/explain.py
python src/fairness_experiments.py
python src/tune.py
python src/score_batch.py
python src/monitor.py
```

| Script | What it does / produces |
|---|---|
| `preprocess.py` | Validates the raw CSV and writes `data/processed/hrattrition_cleaned.csv` |
| `features.py` | Check only: prints the split sizes and number of model input features |
| `train.py` | Cross-validates the models on the training set, chooses the decision threshold, saves `models/model.joblib`, `docs/cv_results.csv` and `docs/calibration_curve.png`, logs runs to MLflow |
| `evaluate.py` | One-time evaluation on the held-out test set: `models/metrics.json`, `docs/pr_curve_test.png`, `docs/confusion_matrix_test.png`. Do not re-run it to tune anything |
| `fairness.py` | Recall and false positive rate gaps by Gender, AgeBand and MaritalStatus: `models/fairness.json`, `docs/fairness_results.csv`, `docs/fairness_test.png` |
| `explain.py` | SHAP explanations: `docs/shap_beeswarm.png`, `docs/shap_importance.csv`, `docs/lr_coefficients.csv`, `docs/shap_waterfall_example.png` |
| `fairness_experiments.py` | Age-bias mitigation experiments on training-set out-of-fold predictions only: `docs/fairness_experiments.csv` |
| `tune.py` | Hyperparameter experiment using training-set cross-validation only; does not change the saved model |
| `score_batch.py` | Batch-scores current employees: `data/processed/attrition_scores.csv` (use `--input` / `--output` to change the files) |
| `monitor.py` | Data drift demo using PSI and KS tests |

### 4. Run the tests

```powershell
python -m pytest -v
```

### 5. MLflow UI

```powershell
python -m mlflow ui
```

Run this from the repository root (it reads `mlflow.db` there), then open http://127.0.0.1:5000.

### 6. API

```powershell
python -m uvicorn api:app --app-dir src --reload
```

Then open http://127.0.0.1:8000/docs. The API needs `models/model.joblib`, so run `train.py` first.

### 7. Docker

`models/model.joblib` must exist before building (run the pipeline first). The image installs `requirements-serve.txt`.

```powershell
docker build -t hr-attrition-api .
docker run -p 8000:8000 hr-attrition-api
```

### 8. Continuous integration

`.github/workflows/ci.yml` runs on GitHub Actions: it installs the requirements, runs the test suite and then `src/quality_gate.py`, which checks `models/metrics.json` and `models/fairness.json` against the success criteria in `config/config.yaml`. To run the gate locally:

```powershell
python src/quality_gate.py
```

## Reproducibility

- The random seed, train/test split, cross-validation folds and success-criteria thresholds (minimum recall and precision, maximum CV standard deviation and maximum fairness gap) live in `config/config.yaml`. The decision threshold is chosen by `train.py` from training out-of-fold predictions and saved with the model.
- `requirements.txt` lists the packages without versions. To record the exact versions used, generate a lock file on your own machine after installing:

```powershell
pip freeze > requirements-lock.txt
```

## Working practices

Keep the repository tidy as you work. In particular:

- Don't modify the raw dataset.
- Don't put all of your code in notebooks.
- Put reusable Python code in `src/`.
- Put tests in `tests/`.
- Save trained models in `models/`.
- Keep project-specific configuration in `config/`.
- Use `docs/` for useful project documentation.
