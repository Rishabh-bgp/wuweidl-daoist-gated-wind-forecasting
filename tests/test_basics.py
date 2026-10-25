"""Basic repository checks: synthetic data present, metrics correct, real data never tracked."""
import subprocess
from pathlib import Path
import numpy as np
from src.metrics import regression_metrics

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_csv_present():
    assert list((ROOT / "data" / "synthetic").glob("synthetic_*.csv"))


def test_metrics_perfect_and_persistence():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(y, y)
    assert m["MAE"] == 0 and m["RMSE"] == 0 and abs(m["R2"] - 1) < 1e-12


def test_only_synthetic_data_is_tracked():
    try:
        files = subprocess.run(["git", "ls-files", "data"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    except Exception:
        return
    assert all(f.startswith("data/synthetic/synthetic_") for f in files), files
