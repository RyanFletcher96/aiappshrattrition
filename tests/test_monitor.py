"""
test_monitor.py, Unit tests for the HR attrition drift monitoring checks.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

# Allow the test runner to find src/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import NOMINAL_COLUMNS, NUMERIC_COLUMNS, ORDINAL_COLUMNS, PROTECTED_COLUMNS
from monitor import drift_report, psi_categorical, psi_numeric


def make_employees(n, seed, overtime_share=0.3):
    """Create a small made-up employee table with every monitored column."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.normal(50, 10, n) for col in NUMERIC_COLUMNS})
    for col in ORDINAL_COLUMNS:
        df[col] = rng.integers(1, 5, n)
    for col in NOMINAL_COLUMNS:
        df[col] = rng.choice(["A", "B", "C"], n)
    df["OverTime"] = np.where(rng.random(n) < overtime_share, "Yes", "No")
    df["Gender"] = rng.choice(["Female", "Male"], n)
    df["MaritalStatus"] = rng.choice(["Single", "Married", "Divorced"], n)
    return df


def test_identical_numeric_distributions_give_psi_near_zero():
    """Comparing a feature with itself should show no drift."""
    values = np.random.default_rng(0).normal(5000, 1000, 1000)

    psi = psi_numeric(values, values)

    assert psi == pytest.approx(0.0, abs=1e-9), f"Expected 0, got {psi}"


def test_shifted_numeric_distribution_gives_high_psi():
    """Moving the whole distribution up by one standard deviation is a big change."""
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 1000)
    current = rng.normal(1, 1, 1000)

    psi = psi_numeric(reference, current)

    assert psi > 0.25, f"Expected PSI above 0.25 for a clear shift, got {psi:.3f}"


def test_identical_categories_give_psi_near_zero():
    """The same category proportions should show no drift."""
    reference = ["Yes"] * 30 + ["No"] * 70
    current = ["Yes"] * 3 + ["No"] * 7

    psi = psi_categorical(reference, current)

    assert psi == pytest.approx(0.0, abs=1e-9), f"Expected 0, got {psi}"


def test_big_category_change_gives_high_psi():
    """Overtime going from 30% to 70% of employees should be flagged."""
    reference = ["Yes"] * 30 + ["No"] * 70
    current = ["Yes"] * 70 + ["No"] * 30

    psi = psi_categorical(reference, current)

    assert psi > 0.25, f"Expected PSI above 0.25 for a big change, got {psi:.3f}"


def test_new_category_is_handled():
    """A category missing from the reference should not cause an error or infinite PSI."""
    reference = ["A"] * 50 + ["B"] * 50
    current = ["A"] * 40 + ["B"] * 40 + ["C"] * 20

    psi = psi_categorical(reference, current)

    assert np.isfinite(psi), "PSI should stay finite when a new category appears"
    assert psi > 0.1, f"A new 20% category should be noticed, got {psi:.3f}"


def test_drift_report_has_one_row_per_model_input():
    """The report should cover every model input once, and no protected attributes."""
    reference = make_employees(400, seed=1)
    current = make_employees(200, seed=2)

    report = drift_report(reference, current)

    expected_features = NUMERIC_COLUMNS + ORDINAL_COLUMNS + NOMINAL_COLUMNS
    assert sorted(report["feature"]) == sorted(expected_features)
    assert list(report.columns) == ["feature", "type", "psi", "test", "p_value", "status"]
    assert not set(PROTECTED_COLUMNS) & set(report["feature"]), "Protected columns should not be monitored as model inputs"


def test_drift_report_flags_overtime_change():
    """A big rise in overtime should be marked for investigation."""
    reference = make_employees(400, seed=1, overtime_share=0.3)
    current = make_employees(200, seed=2, overtime_share=0.7)

    report = drift_report(reference, current).set_index("feature")

    assert report.loc["OverTime", "status"] == "significant - investigate"
    assert report.loc["OverTime", "test"] == "chi2"
    assert report.loc["MonthlyIncome", "test"] == "ks"
