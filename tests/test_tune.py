"""
test_tune.py, Unit tests for the logistic regression tuning helpers.
"""

import pandas as pd
import sys
import os
import tempfile
from pathlib import Path

# Allow the test runner to find src/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tune import C_GRID, CURRENT_C, get_model, save_tuning_plot


def test_grid_includes_current_c():
    """The grid must include the current C, so the tuned values can be compared with it."""
    assert CURRENT_C in C_GRID, f"Current C={CURRENT_C} missing from the grid {C_GRID}"
    assert C_GRID == sorted(C_GRID), "Grid should be in increasing order for the plot"


def test_model_only_changes_c():
    """Tuned models must keep the same settings as the chosen model in train.py."""
    model = get_model(0.1)
    params = model.get_params()
    assert params["C"] == 0.1, f"Expected C=0.1, got {params['C']}"
    assert params["class_weight"] == "balanced", "class_weight should stay balanced"
    assert params["max_iter"] == 1000, "max_iter should stay 1000"


def test_plot_is_saved():
    """The plot should be written to disk from a small hand-made results table."""
    results = pd.DataFrame({
        "C": C_GRID,
        "pr_auc_mean": [0.60] * len(C_GRID),
        "pr_auc_std": [0.04, 0.06, 0.04, 0.06, 0.04, 0.06, 0.04],
    })
    results["meets_std_target"] = results["pr_auc_std"] <= 0.05

    with tempfile.TemporaryDirectory() as folder:
        output_path = Path(folder) / "tuning_c.png"
        save_tuning_plot(results, 0.05, output_path)
        assert output_path.exists(), "Tuning plot was not saved"
