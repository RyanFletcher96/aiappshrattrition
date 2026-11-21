"""
test_fairness.py, Unit tests for the HR attrition fairness checks.
"""

import pandas as pd
import pytest
import sys
import os

# Allow the test runner to find src/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fairness import fairness_gaps, sensitive_features


def test_fairness_gaps_measures_difference_between_groups():
    """Recall gap should be the difference between the best and worst group."""
    # Group A: 2 leavers, both flagged (recall 1.0)
    # Group B: 2 leavers, one flagged (recall 0.5)
    y_true = pd.Series([1, 1, 0, 1, 1, 0])
    y_pred = pd.Series([1, 1, 0, 1, 0, 0])
    groups = pd.Series(["A", "A", "A", "B", "B", "B"])

    by_group, gaps = fairness_gaps(y_true, y_pred, groups)

    assert by_group.loc["A", "recall"] == pytest.approx(1.0)
    assert by_group.loc["B", "recall"] == pytest.approx(0.5)
    assert gaps["recall_gap"] == pytest.approx(0.5), f"Expected 0.5, got {gaps['recall_gap']}"


def test_equal_treatment_gives_zero_gap():
    """If both groups are treated identically, both gaps should be zero."""
    y_true = pd.Series([1, 0, 1, 0])
    y_pred = pd.Series([1, 0, 1, 0])
    groups = pd.Series(["A", "A", "B", "B"])

    _, gaps = fairness_gaps(y_true, y_pred, groups)

    assert gaps["recall_gap"] == pytest.approx(0.0)
    assert gaps["fpr_gap"] == pytest.approx(0.0)


def test_sensitive_features_builds_age_bands():
    """Ages should be placed into the configured bands, with no one left out."""
    X = pd.DataFrame({
        "Age": [18, 29, 30, 45, 46, 60],
        "Gender": ["Female", "Male", "Female", "Male", "Female", "Male"],
        "MaritalStatus": ["Single", "Married", "Divorced", "Single", "Married", "Divorced"],
    })

    groups = sensitive_features(X, bins=[18, 30, 46, 65], labels=["<30", "30-45", ">45"])

    assert list(groups["AgeBand"].astype(str)) == ["<30", "<30", "30-45", "30-45", ">45", ">45"]
    assert groups["AgeBand"].isna().sum() == 0, "Some ages were not placed in a band"
