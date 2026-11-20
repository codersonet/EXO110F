import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from sklearn.model_selection import StratifiedShuffleSplit

from exoplanet.data import load_split

ROOT = Path(__file__).resolve().parents[2]

CONFIG = {
    "version": "v1",
    "detrend_median_window": 101,   # time steps
    "upper_clip_sigma": 4.0,
    "val_fraction": 0.2,
    "seed": 42,
}


def preprocess(X, cfg=CONFIG):
    """X: (n_stars, n_points) raw flux. Returns normalized float32 array, same shape."""
    if np.isnan(X).any():
        raise ValueError("NaNs found: decide how to handle them explicitly.")
    trend = median_filter(X, size=(1, cfg["detrend_median_window"]), mode="nearest")
    R = X - trend
    med = np.median(R, axis=1, keepdims=True)
    sigma = 1.4826 * np.median(np.abs(R - med), axis=1, keepdims=True)  # robust std
    sigma[sigma == 0] = 1.0
    R = (R - med) / sigma
    return np.minimum(R, cfg["upper_clip_sigma"]).astype("float32")


def main():
    X_tr_raw, y_tr = load_split("Train")
    X_te_raw, y_te = load_split("Test")

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=CONFIG["val_fraction"], random_state=CONFIG["seed"]
    )
    tr_idx, va_idx = next(sss.split(X_tr_raw, y_tr))

    X_tr = preprocess(X_tr_raw[tr_idx])
    X_va = preprocess(X_tr_raw[va_idx])
    X_te = preprocess(X_te_raw)

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "dataset_v1.npz",
        X_train=X_tr, y_train=y_tr[tr_idx],
        X_val=X_va, y_val=y_tr[va_idx],
        X_test=X_te, y_test=y_te,
        train_idx=tr_idx, val_idx=va_idx,
    )
    (ROOT / "configs").mkdir(exist_ok=True)
    (ROOT / "configs" / "preprocess_v1.json").write_text(json.dumps(CONFIG, indent=2))

    for name, y in [("train", y_tr[tr_idx]), ("val", y_tr[va_idx]), ("test", y_te)]:
        print(f"{name}: n={len(y)}, planets={int(y.sum())}, non-planets={int((y == 0).sum())}")


if __name__ == "__main__":
    main()