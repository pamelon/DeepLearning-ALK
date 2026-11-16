"""Export California housing to CSV + MLTable folders for the AutoML job.

Uses exactly the same data as `keras.datasets.california_housing.load_data()` (same file,
same test split) and the same train/validation split as the Class 3 challenge notebook
(SPLIT_SEED=2026, 15% validation), so students can compare AutoML with their own model.
Only numpy is needed - no TensorFlow.

Run once before submitting automl-housing.yml:
    python prepare_automl_data.py
"""
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/california_housing.npz"
COLUMNS = ["longitude", "latitude", "housing_median_age", "total_rooms",
           "total_bedrooms", "population", "households", "median_income"]
TARGET = "median_house_value"
SPLIT_SEED, VAL_FRACTION = 2026, 0.15

MLTABLE = """paths:
  - file: ./data.csv
transformations:
  - read_delimited:
      delimiter: ','
      encoding: 'utf8'
      header: all_files_same_headers
"""


def load_like_keras(test_split=0.2, seed=113):
    """Same result as keras.datasets.california_housing.load_data(version="large")."""
    cache = Path.home() / ".keras" / "datasets" / "california_housing.npz"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, cache)
    with np.load(cache) as f:
        x, y = f["x"], f["y"]
    indices = np.arange(len(x))
    np.random.RandomState(seed).shuffle(indices)
    x, y = x[indices], y[indices]
    n_train = int(len(x) * (1 - test_split))
    return (x[:n_train], y[:n_train]), (x[n_train:], y[n_train:])


def write(folder, x, y):
    folder.mkdir(parents=True, exist_ok=True)
    np.savetxt(folder / "data.csv", np.column_stack([x, y]), delimiter=",",
               header=",".join(COLUMNS + [TARGET]), comments="", fmt="%.6g")
    (folder / "MLTable").write_text(MLTABLE)
    print(f"{folder}: {len(y)} rows")


(x_full, y_full), (x_test, y_test) = load_like_keras()
idx = np.random.default_rng(SPLIT_SEED).permutation(len(x_full))
n_val = int(len(idx) * VAL_FRACTION)

root = Path(__file__).parent / "data" / "california"
write(root / "train", x_full[idx[n_val:]], y_full[idx[n_val:]])
write(root / "validation", x_full[idx[:n_val]], y_full[idx[:n_val]])
write(root / "test", x_test, y_test)
