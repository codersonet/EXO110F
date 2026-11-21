"""Small automatic checks. Run from the project root:  PYTHONPATH=src pytest -q"""
import numpy as np

from exoplanet.preprocess import ROOT, preprocess


def _load():
    return np.load(ROOT / "data" / "processed" / "dataset_v1.npz")


def test_no_star_in_both_train_and_validation():
    d = _load()
    assert len(np.intersect1d(d["train_idx"], d["val_idx"])) == 0


def test_shapes_and_labels():
    d = _load()
    for s in ("train", "val", "test"):
        assert d[f"X_{s}"].shape[0] == d[f"y_{s}"].shape[0]
        assert set(np.unique(d[f"y_{s}"])) <= {0, 1}
    assert d["X_train"].shape[1] == d["X_val"].shape[1] == d["X_test"].shape[1]


def test_every_split_has_planets():
    d = _load()
    for s in ("train", "val", "test"):
        assert d[f"y_{s}"].sum() > 0


def test_no_nans_after_preprocessing():
    d = _load()
    for s in ("train", "val", "test"):
        assert not np.isnan(d[f"X_{s}"]).any()


def test_transit_dip_survives_preprocessing():
    rng = np.random.default_rng(0)
    x = 1000 + rng.normal(0, 1, (1, 3000))
    x[0, 1500:1510] -= 15                      # a short, deep dip
    out = preprocess(x.astype("float32"))
    assert out.min() < -5


def test_upward_spike_is_clipped_but_dips_are_not():
    rng = np.random.default_rng(1)
    x = 1000 + rng.normal(0, 1, (1, 3000))
    x[0, 100] += 500                           # cosmic-ray style spike
    out = preprocess(x.astype("float32"))
    assert out.max() <= 4.0 + 1e-6


def test_preprocessing_is_per_star():
    rng = np.random.default_rng(2)
    a = 1000 + rng.normal(0, 1, (1, 3000)).astype("float32")
    b = 50 + rng.normal(0, 5, (1, 3000)).astype("float32")
    together = preprocess(np.vstack([a, b]))
    alone = preprocess(a)
    assert np.allclose(together[0], alone[0], atol=1e-5)