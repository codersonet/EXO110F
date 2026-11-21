"""Final evaluation on the locked Kaggle test file. RUN ONCE.
Decide MODEL from the cross-validation results BEFORE running this file.
The threshold comes from cross-validation (results/cv_results.json), 
never from the test data."""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from exoplanet.evaluate import report
from exoplanet.features import features
from exoplanet.models import augment_positives, build_cnn
from exoplanet.preprocess import ROOT

MODEL = "cnn"      # "cnn" or "baseline", chosen from cross-validation, not from the test data. The threshold comes from cross-validation, never from the test data.
SEED = 42

OUT = ROOT / "results" / "final_test.json"
if OUT.exists():
    sys.exit(f"{OUT.name} already exists: the test set has been used. Re-running means choosing again "
             "after seeing test results. Delete the file only if you are starting a new, documented experiment.")

cv = json.loads((ROOT / "results" / "cv_results.json").read_text())
thr = cv[MODEL]["threshold_max_f1"]

d = np.load(ROOT / "data" / "processed" / "dataset_v1.npz")
X = np.concatenate([d["X_train"], d["X_val"]])      # train on ALL labeled stars
y = np.concatenate([d["y_train"], d["y_val"]])
X_te, y_te = d["X_test"], d["y_test"]
print(f"training stars={len(y)} (planets={int(y.sum())}), test stars={len(y_te)} (planets={int(y_te.sum())})")

if MODEL == "baseline":
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(class_weight="balanced", max_iter=2000))
    model.fit(features(X), y)
    p = model.predict_proba(features(X_te))[:, 1]
else:
    keras.utils.set_random_seed(SEED)
    X_tr, y_tr = X, y
    if cv["augment_copies"]:
        X_tr, y_tr = augment_positives(X, y, cv["augment_copies"], np.random.default_rng(SEED))
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    net = build_cnn(X.shape[1])
    net.fit(X_tr[..., None], y_tr, epochs=cv["epochs"], batch_size=32,
            class_weight={0: cw[0], 1: cw[1]}, verbose=2)
    p = net.predict(X_te[..., None], verbose=0).ravel()

report(y_te, p, thr=thr, name=f"{MODEL} FINAL TEST")


# ---------------------------------------------------------------- ranked candidate lists
def ranked_table(scores, labels, csv_rows, source):
    """Stars sorted from most to least planet-like. 'known_label' is the Kaggle label, shown for checking."""
    order = np.argsort(-scores)
    return pd.DataFrame({
        "rank": np.arange(1, len(scores) + 1),
        "source": source,
        "csv_row": csv_rows[order],                 # 0-based data row in that CSV (header not counted)
        "planet_score": np.round(scores[order], 4),
        "flagged_as_candidate": scores[order] >= thr,
        "known_label": np.where(labels[order] == 1, "planet", "non-planet"),
    })


res_dir = ROOT / "results"
res_dir.mkdir(exist_ok=True)

# 1) Held-out list: test stars scored by the final model. This is the list that is a fair exam.
test_list = ranked_table(p, y_te, np.arange(len(y_te)), "exoTest.csv (held-out test)")
test_list.to_csv(res_dir / "ranked_candidates_test.csv", index=False)

# 2) Labeled-star list: out-of-fold scores from cross-validation (each star was scored by a model
#    that never trained on it). Only written if cv.py saved results/cv_oof.npz.
oof_path = res_dir / "cv_oof.npz"
if oof_path.exists():
    lab_rows = np.concatenate([d["train_idx"], d["val_idx"]])
    lab_list = ranked_table(np.load(oof_path)[MODEL], y, lab_rows, "exoTrain.csv (cross-validated)")
    lab_list.to_csv(res_dir / "ranked_candidates_labeled_cv.csv", index=False)
else:
    print("note: results/cv_oof.npz not found, so the labeled-star ranked list was skipped")

order_te = np.argsort(-p)
topk = {int(k): int(y_te[order_te[:k]].sum()) for k in (5, 10, 20, 50)}
print("\nTop of the held-out ranked list:")
print(test_list.head(15).to_string(index=False))
print("known planets inside the top k of the test list:", topk, f"(of {int(y_te.sum())} planets)")

pred = (p >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
result = {
    "model": MODEL, "threshold_from_cv": thr, "seed": SEED,
    "epochs": cv.get("epochs"), "augment_copies": cv.get("augment_copies"),
    "test_stars": int(len(y_te)), "test_planets": int(y_te.sum()),
    "pr_auc": float(average_precision_score(y_te, p)),
    "roc_auc": float(roc_auc_score(y_te, p)),
    "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    "planets_in_top_k_of_test_list": topk,
    "note": "Few test planets: treat every number as rough. Predictions are candidates, not confirmations.",
}
OUT.write_text(json.dumps(result, indent=2))
print(f"\nplanets found: {tp} of {int(y_te.sum())}; false alarms: {fp}")
print("saved", OUT)