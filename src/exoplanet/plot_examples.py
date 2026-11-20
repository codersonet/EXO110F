import matplotlib.pyplot as plt
from data import load_split

X, y = load_split("Train")
planet_idx = y.nonzero()[0][:3]
normal_idx = (y == 0).nonzero()[0][:3]

fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
for ax, i in zip(axes[0], planet_idx):
    ax.plot(X[i], lw=0.5); ax.set_title(f"Planet star #{i}")
for ax, i in zip(axes[1], normal_idx):
    ax.plot(X[i], lw=0.5); ax.set_title(f"Non-planet star #{i}")
fig.supxlabel("Time index"); fig.supylabel("Flux")
plt.tight_layout()
plt.savefig("results/example_light_curves.png", dpi=120)
print("saved results/example_light_curves.png")