"""
test_features.py, Unit tests for the HR attrition feature preparation.
"""

import numpy as np
import pandas as pd
import sys
import os

# Allow the test runner to find src/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import (
    add_engineered_features,
    build_preprocessor,
    ENGINEERED_COLUMNS,
    NUMERIC_COLUMNS,
    ORDINAL_COLUMNS,
    NOMINAL_COLUMNS,
    PROTECTED_COLUMNS,
)


def make_mini_df():
    """Create a small DataFrame with the model input columns for testing."""
    data = {col: [1, 2, 3] for col in NUMERIC_COLUMNS + ORDINAL_COLUMNS}
    df = pd.DataFrame(data)
    df["BusinessTravel"] = ["Travel_Rarely", "Non-Travel", "Travel_Frequently"]
    df["Department"] = ["Sales", "Research & Development", "Sales"]
    df["EducationField"] = ["Life Sciences", "Medical", "Marketing"]
    df["JobRole"] = ["Sales Executive", "Research Scientist", "Manager"]
    df["OverTime"] = ["Yes", "No", "No"]
    df["Gender"] = ["Female", "Male", "Male"]
    df["MaritalStatus"] = ["Single", "Married", "Divorced"]
    return df


def test_engineered_features_handle_zero_years():
    """Zero working years or zero years at the company must not give NaN or inf."""
    df = make_mini_df()
    df["TotalWorkingYears"] = [0, 0, 5]
    df["YearsAtCompany"] = [0, 3, 0]

    result = add_engineered_features(df)

    for col in ENGINEERED_COLUMNS:
        assert col in result.columns, f"Engineered column '{col}' was not added"
        assert np.isfinite(result[col]).all(), f"{col} contains NaN or inf"


def test_engineered_features_do_not_change_input():
    """add_engineered_features must return a new DataFrame, not modify the original."""
    df = make_mini_df()
    original_columns = list(df.columns)

    add_engineered_features(df)

    assert list(df.columns) == original_columns, "The input DataFrame was modified"


def test_protected_columns_excluded_by_default():
    """Gender and MaritalStatus must not reach the model unless explicitly included."""
    df = add_engineered_features(make_mini_df())

    preprocessor = build_preprocessor().fit(df)
    feature_names = preprocessor.get_feature_names_out()

    for col in PROTECTED_COLUMNS:
        assert not any(col in name for name in feature_names), (
            f"Protected column '{col}' was passed to the model"
        )


def test_protected_columns_included_when_requested():
    """include_protected=True should add Gender and MaritalStatus for the comparison run."""
    df = add_engineered_features(make_mini_df())

    preprocessor = build_preprocessor(include_protected=True).fit(df)
    feature_names = preprocessor.get_feature_names_out()

    for col in PROTECTED_COLUMNS:
        assert any(col in name for name in feature_names), (
            f"Protected column '{col}' is missing when include_protected=True"
        )
