"""
test_preprocess.py, Unit tests for the HR attrition preprocessing pipeline.
"""

import pandas as pd
import pytest
import sys
import os

# Allow the test runner to find src/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import validate_columns, EXPECTED_COLUMNS


def make_mini_df():
    """Create a small DataFrame with the expected schema for testing."""
    data = {col: [1, 2, 3] for col in EXPECTED_COLUMNS}
    df = pd.DataFrame(data)
    # Make Age and Attrition realistic so the valid data passes validation
    df["Age"] = [25, 40, 55]
    df["Attrition"] = ["Yes", "No", "No"]
    return df


def test_valid_data_passes():
    """Valid data should pass validation without raising an error."""
    df = make_mini_df()
    validate_columns(df)


def test_missing_column_raises():
    """A missing expected column (Age) should raise a ValueError."""
    df = make_mini_df().drop(columns=["Age"])
    with pytest.raises(ValueError, match="Missing expected columns"):
        validate_columns(df)


def test_null_monthly_income_raises():
    """A null MonthlyIncome should raise a ValueError."""
    df = make_mini_df()
    df["MonthlyIncome"] = [5000, None, 7000]
    with pytest.raises(ValueError, match="Nulls in required columns"):
        validate_columns(df)


def test_invalid_attrition_value_raises():
    """An Attrition value other than Yes/No should raise a ValueError."""
    df = make_mini_df()
    df["Attrition"] = ["Yes", "Maybe", "No"]
    with pytest.raises(ValueError, match="Invalid Attrition values"):
        validate_columns(df)


def test_out_of_range_value_raises():
    """A value outside VALUE_RANGES (JobSatisfaction of 7) should raise a ValueError."""
    df = make_mini_df()
    df["JobSatisfaction"] = [1, 7, 3]
    with pytest.raises(ValueError, match="JobSatisfaction above maximum"):
        validate_columns(df)