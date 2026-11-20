import numpy as np
import matplotlib.pyplot as plt
from exoplanet.data import load_split
from exoplanet.preprocess import ROOT, preprocess

d = np.load(ROOT / "data" / "processed" / "dataset_v1.npz")
X_raw, _ = load_split("Train")
idx = d["train_idx"][d["y_train"] == 1][:3]   # three planet stars

fig, axes = plt.subplots(2, 3, figsize=(14, 6))
for j, i in enumerate(idx):
    axes[0, j].plot(X_raw[i], lw=0.5); axes[0, j].set_title(f"raw, star {i}")
    axes[1, j].plot(preprocess(X_raw[i:i + 1])[0], lw=0.5); axes[1, j].set_title("processed")
plt.tight_layout()
(ROOT / "results").mkdir(exist_ok=True)
plt.savefig(ROOT / "results" / "preprocess_check.png", dpi=120)
print("saved results/preprocess_check.png")