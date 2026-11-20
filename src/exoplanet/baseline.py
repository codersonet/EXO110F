import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from exoplanet.preprocess import ROOT


def features(X):
    f = [X.min(1), X.max(1), X.std(1), skew(X, axis=1), kurtosis(X, axis=1)]
    f += [np.percentile(X, p, axis=1) for p in (1, 5, 25, 50, 75, 95, 99)]
    spec = np.abs(np.fft.rfft(X, axis=1))[:, 1:101]          # 100 lowest frequencies
    spec = spec / (spec.sum(1, keepdims=True) + 1e-9)
    return np.column_stack(f + [spec])


d = np.load(ROOT / "data" / "processed" / "dataset_v1.npz")
F_tr, F_va = features(d["X_train"]), features(d["X_val"])
y_tr, y_va = d["y_train"], d["y_val"]

model = make_pipeline(StandardScaler(),
                      LogisticRegression(class_weight="balanced", max_iter=2000))
model.fit(F_tr, y_tr)

p = model.predict_proba(F_va)[:, 1]
pred = (p >= 0.5).astype(int)
print(f"val planets: {int(y_va.sum())} of {len(y_va)}")
print(f"PR-AUC : {average_precision_score(y_va, p):.3f}  (random is about {y_va.mean():.3f})")
print(f"ROC-AUC: {roc_auc_score(y_va, p):.3f}")
print(f"precision: {precision_score(y_va, pred, zero_division=0):.3f}, "
      f"recall: {recall_score(y_va, pred):.3f}")
print("confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_va, pred))