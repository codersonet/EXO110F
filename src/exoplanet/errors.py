"""Error analysis on out-of-fold (cross-validation) predictions.

Uses only train+validation stars. The locked test file is never touched here.
Run:  PYTHONPATH=src python -m exoplanet.errors
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from exoplanet.preprocess import ROOT

MODEL = "cnn"      # "cnn" or "baseline"
TOP = 6            # how many mistakes of each kind to plot

d = np.load(ROOT / "data" / "processed" / "dataset_v1.npz")
X = np.concatenate([d["X_train"], d["X_val"]])
y = np.concatenate([d["y_train"], d["y_val"]])
csv_row = np.concatenate([d["train_idx"], d["val_idx"]])   # row number in exoTrain.csv

oof = np.load(ROOT / "results" / "cv_oof.npz")
p = oof[MODEL]
thr = json.loads((ROOT / "results" / "cv_results.json").read_text())[MODEL]["threshold_max_f1"]
pred = (p >= thr).astype(int)

tp = int(((pred == 1) & (y == 1)).sum())
fp_all = np.where((pred == 1) & (y == 0))[0]
fn_all = np.where((pred == 0) & (y == 1))[0]
print(f"model={MODEL}, threshold={thr:.3f}")
print(f"planets found (TP)={tp}, missed planets (FN)={len(fn_all)}, false alarms (FP)={len(fp_all)}")

# most confident false alarms = highest score among non-planets
fp = fp_all[np.argsort(-p[fp_all])][:TOP]
# most confident misses = lowest score among planets
fn = fn_all[np.argsort(p[fn_all])][:TOP]

fig, axes = plt.subplots(2, TOP, figsize=(3 * TOP, 6), squeeze=False)
for row, (idx, label) in enumerate([(fp, "FALSE ALARM"), (fn, "MISSED PLANET")]):
    for j in range(TOP):
        ax = axes[row, j]
        if j < len(idx):
            i = idx[j]
            ax.plot(X[i], lw=0.5)
            ax.set_title(f"{label}\nCSV row {csv_row[i]}, score {p[i]:.2f}", fontsize=8)
        else:
            ax.axis("off")
plt.tight_layout()
out = ROOT / "results" / f"error_analysis_{MODEL}.png"
plt.savefig(out, dpi=120)
print("saved", out)

print("\nCSV rows of the plotted false alarms:", [int(csv_row[i]) for i in fp])
print("CSV rows of the plotted missed planets:", [int(csv_row[i]) for i in fn])
print("""
What to look for in each plot:
  false alarms -> a second dip of different depth (eclipsing binary), big smooth wiggles
                  (stellar variability), a single huge spike, a jump or step in the curve.
  missed planets -> a dip so shallow it sits in the noise, or a dip removed by the median filter.
  A pattern across many mistakes usually points to a preprocessing or labeling issue.
""")