import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from exoplanet.features import features
from exoplanet.models import augment_positives, build_cnn
from exoplanet.preprocess import ROOT

SEED = 42
N_SPLITS = 5
N_REPEATS = 1        # start with 1 to test; use 3 for the final numbers
EPOCHS = 30          # set from where val_pr_auc peaked in your cnn.py run
AUGMENT_COPIES = 0   # try 10 in a second run

d = np.load(ROOT / "data" / "processed" / "dataset_v1.npz")
X = np.concatenate([d["X_train"], d["X_val"]])     # test file stays locked
y = np.concatenate([d["y_train"], d["y_val"]])
F = features(X)                                     # per-star features, so no leakage across folds
print(f"stars={len(y)}, planets={int(y.sum())}")

cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
oof = {m: np.zeros((N_REPEATS, len(y))) for m in ("baseline", "cnn")}
fold_pr = {m: [] for m in oof}
fold_roc = {m: [] for m in oof}

for k, (tr, te) in enumerate(cv.split(X, y)):
    rep = k // N_SPLITS

    base = make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=2000))
    base.fit(F[tr], y[tr])
    preds = {"baseline": base.predict_proba(F[te])[:, 1]}

    keras.utils.set_random_seed(SEED + k)
    Xtr, ytr = X[tr], y[tr]
    if AUGMENT_COPIES:
        Xtr, ytr = augment_positives(Xtr, ytr, AUGMENT_COPIES, np.random.default_rng(SEED + k))
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=ytr)
    model = build_cnn(X.shape[1])
    model.fit(Xtr[..., None], ytr, epochs=EPOCHS, batch_size=32,
              class_weight={0: cw[0], 1: cw[1]}, verbose=0)
    preds["cnn"] = model.predict(X[te][..., None], verbose=0).ravel()
    keras.backend.clear_session()

    for m, p in preds.items():
        oof[m][rep, te] = p
        fold_pr[m].append(average_precision_score(y[te], p))
        fold_roc[m].append(roc_auc_score(y[te], p))
    print(f"fold {k + 1}/{N_SPLITS * N_REPEATS}: "
          f"baseline PR-AUC={fold_pr['baseline'][-1]:.3f}, cnn PR-AUC={fold_pr['cnn'][-1]:.3f}")


def best_threshold(y_true, p):
    prec, rec, thr = precision_recall_curve(y_true, p)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    i = int(np.argmax(f1))
    return float(thr[i]), float(prec[i]), float(rec[i]), float(f1[i])


summary = {"seed": SEED, "n_splits": N_SPLITS, "n_repeats": N_REPEATS,
           "epochs": EPOCHS, "augment_copies": AUGMENT_COPIES}
y_all = np.tile(y, N_REPEATS)
for m in oof:
    thr, prec, rec, f1 = best_threshold(y_all, oof[m].reshape(-1))
    summary[m] = {
        "pr_auc_mean": float(np.mean(fold_pr[m])), "pr_auc_std": float(np.std(fold_pr[m])),
        "roc_auc_mean": float(np.mean(fold_roc[m])), "roc_auc_std": float(np.std(fold_roc[m])),
        "threshold_max_f1": thr, "precision_at_thr": prec, "recall_at_thr": rec, "f1_at_thr": f1,
    }
    print(f"\n[{m}] PR-AUC {np.mean(fold_pr[m]):.3f} +/- {np.std(fold_pr[m]):.3f} | "
          f"ROC-AUC {np.mean(fold_roc[m]):.3f} +/- {np.std(fold_roc[m]):.3f}")
    print(f"    best-F1 threshold {thr:.3f}: precision {prec:.3f}, recall {rec:.3f}, F1 {f1:.3f}")

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "cv_results.json").write_text(json.dumps(summary, indent=2))
print("\nsaved results/cv_results.json")