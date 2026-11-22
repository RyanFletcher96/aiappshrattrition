"""
test_score_batch.py, Unit tests for the batch scoring checks.
"""

import pandas as pd
import pytest
import sys
import os

# Allow the test runner to find src/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from score_batch import ID_COLUMN, MODEL_INPUT_COLUMNS, model_version, validate_for_scoring


def make_mini_df():
    """Create a small DataFrame with every column the model needs for scoring."""
    data = {col: [1, 2, 3] for col in MODEL_INPUT_COLUMNS}
    df = pd.DataFrame(data)
    # Make Age and the IDs realistic so the valid data passes validation
    df["Age"] = [25, 40, 55]
    df[ID_COLUMN] = [101, 102, 103]
    return df


def test_valid_data_passes():
    """Valid data should pass validation without raising an error."""
    validate_for_scoring(make_mini_df())


def test_missing_column_raises():
    """A missing model input column (OverTime) should raise a ValueError."""
    df = make_mini_df().drop(columns=["OverTime"])
    with pytest.raises(ValueError, match="Missing columns needed for scoring"):
        validate_for_scoring(df)


def test_duplicate_employee_number_raises():
    """The same EmployeeNumber twice should raise, so nobody is scored twice."""
    df = make_mini_df()
    df[ID_COLUMN] = [101, 101, 103]
    with pytest.raises(ValueError, match="Duplicate EmployeeNumber"):
        validate_for_scoring(df)


def test_model_version_falls_back_when_no_metadata():
    """Models saved without metadata should report "unversioned"."""
    assert model_version({}) == "unversioned"
    assert model_version({"metadata": {"version": "1.2.0"}}) == "1.2.0"
