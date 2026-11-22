"""Batch-score current employees with the saved model (the recommended deployment pattern)."""

import argparse
import logging
import sys
import time
import joblib
import pandas as pd
from datetime import datetime, timezone
from config_loader import PROJECT_ROOT, load_config
from features import NOMINAL_COLUMNS, NUMERIC_COLUMNS, ORDINAL_COLUMNS
from preprocess import TARGET_COLUMN, VALUE_RANGES

ID_COLUMN = "EmployeeNumber"
# The model only needs these. Gender and MaritalStatus are deliberately not required.
MODEL_INPUT_COLUMNS = NUMERIC_COLUMNS + ORDINAL_COLUMNS + NOMINAL_COLUMNS

# Named explicitly because __name__ is "__main__" when this file is run as a script
logger = logging.getLogger("score_batch")

def model_version(saved: dict) -> str:
    """Return the saved model's version, or "unversioned" for models saved without metadata."""
    return saved.get("metadata", {}).get("version", "unversioned")

def validate_for_scoring(df: pd.DataFrame) -> None:
    """Raise ValueError if current-employee data can't be scored safely."""
    missing = set(MODEL_INPUT_COLUMNS + [ID_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns needed for scoring: {sorted(missing)}")

    nulls = df[MODEL_INPUT_COLUMNS].isnull().sum()
    if nulls.any():
        raise ValueError(f"Nulls in model input columns:\n{nulls[nulls > 0]}")

    if df[ID_COLUMN].duplicated().any():
        raise ValueError(f"Duplicate {ID_COLUMN} values found")

    for col, (low, high) in VALUE_RANGES.items():
        if low is not None and (df[col] < low).any():
            raise ValueError(f"Values in {col} below minimum {low}")
        if high is not None and (df[col] > high).any():
            raise ValueError(f"Values in {col} above maximum {high}")

def score_employees(saved: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Return a risk score and a yes/no flag for each row of X."""
    scores = saved["pipeline"].predict_proba(X[MODEL_INPUT_COLUMNS])[:, 1]
    return pd.DataFrame({
        "risk_score": scores.round(4),
        "flagged": scores >= saved["threshold"],
    }, index=X.index)

if __name__ == "__main__":
    # Time, level and logger name on every line, so operators can trace what happened and when
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Score current employees for attrition risk.")
    parser.add_argument("--input", default=cfg["paths"]["raw_data"],
                        help="CSV of current employees (defaults to the raw dataset for a demo)")
    parser.add_argument("--output",
                        default=PROJECT_ROOT / "data" / "processed" / "attrition_scores.csv",
                        help="Where to write the scores")
    args = parser.parse_args()

    start = time.perf_counter()
    logger.info("Batch scoring started, input: %s", args.input)

    df = pd.read_csv(args.input)
    logger.info("Read %d input rows", len(df))
    if TARGET_COLUMN in df.columns:
        # Current employees have no outcome yet; drop it so the demo matches real use
        df = df.drop(columns=[TARGET_COLUMN])
        logger.info("Dropped %s column (not available for current employees)", TARGET_COLUMN)

    try:
        validate_for_scoring(df)
    except ValueError as error:
        # The check is all-or-nothing: give the operator one clear reason, not a traceback
        logger.error("Validation failed, all %d rows rejected and nothing scored: %s",
                     len(df), error)
        sys.exit(1)
    logger.info("Validation passed: %d rows accepted, 0 rejected", len(df))

    saved = joblib.load(cfg["paths"]["model"])
    version = model_version(saved)
    logger.info("Loaded model %s (version %s, threshold %.3f)",
                saved["model_name"], version, saved["threshold"])
    results = score_employees(saved, df)

    # Output only the ID and the scores: no protected attributes or other personal data
    output = pd.concat([df[[ID_COLUMN]], results], axis=1)
    output["model_name"] = saved["model_name"]
    output["model_version"] = version
    output["scored_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    output = output.sort_values("risk_score", ascending=False)
    output.to_csv(args.output, index=False)

    # Counts only: individual employees' scores stay in the output file, not the log
    n_flagged = int(output["flagged"].sum())
    logger.info("Scored %d employees, flagged %d (%.1f%%) for a retention conversation",
                len(output), n_flagged, 100 * output["flagged"].mean())
    logger.info("Note: risk scores rank employees by risk; they are not calibrated probabilities.")
    logger.info("Wrote %d output rows to %s", len(output), args.output)
    logger.info("Batch scoring finished in %.2f s", time.perf_counter() - start)

    # Shown on screen for the demo only, deliberately not logged (employee-level results)
    print("\nTop 5 by risk score:")
    print(output.head(5).to_string(index=False))
