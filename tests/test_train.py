"""
test_train.py, Unit tests for the HR attrition training helpers.
"""

import numpy as np
import pytest
import sys
import os

# Allow the test runner to find src/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train import choose_threshold, get_models


def make_scores():
    """Create small hand-made labels and predicted probabilities for testing."""
    y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    proba = np.array([0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.40, 0.55, 0.70, 0.90])
    return y_true, proba


def test_threshold_meets_minimum_recall():
    """The chosen threshold must catch at least min_recall of the leavers."""
    y_true, proba = make_scores()

    threshold, precision, recall = choose_threshold(y_true, proba, min_recall=0.75)

    predicted = proba >= threshold
    actual_recall = predicted[y_true == 1].mean()
    assert actual_recall >= 0.75, f"Recall {actual_recall} is below 0.75"
    assert recall == pytest.approx(actual_recall), "Returned recall does not match the threshold"


def test_threshold_picks_highest_precision():
    """Among thresholds that meet recall, the one with the best precision should be chosen."""
    y_true, proba = make_scores()

    # Recall >= 0.75 is reached at 0.55 (precision 0.75) and at lower thresholds
    # with worse precision, so 0.55 is the best choice
    threshold, precision, recall = choose_threshold(y_true, proba, min_recall=0.75)

    assert threshold == pytest.approx(0.55), f"Expected 0.55, got {threshold}"
    assert precision == pytest.approx(0.75), f"Expected precision 0.75, got {precision}"


def test_unreachable_recall_raises():
    """Asking for more recall than any threshold gives should raise a ValueError."""
    y_true, proba = make_scores()
    with pytest.raises(ValueError, match="No threshold reaches recall"):
        choose_threshold(y_true, proba, min_recall=1.1)


def test_dummy_baseline_is_included():
    """The model comparison must always include the dummy baseline."""
    models = get_models(seed=42)
    assert "dummy" in models, "Dummy baseline missing from the model comparison"
    assert len(models) >= 3, "Expected the baseline plus at least two real models"
