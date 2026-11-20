import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight

from exoplanet.evaluate import report
from exoplanet.preprocess import ROOT

SEED = 42
AUGMENT_COPIES = 0      # try 10 on a second run
keras.utils.set_random_seed(SEED)

d = np.load(ROOT / "data" / "processed" / "dataset_v1.npz")
X_tr, y_tr = d["X_train"], d["y_train"]
X_va, y_va = d["X_val"], d["y_val"]


def augment_positives(X, y, copies, rng):
    """Add shifted/flipped/noisy copies of TRAINING planets only (after the split, so no leakage)."""
    pos = X[y == 1]
    new = []
    for _ in range(copies):
        for x in pos:
            x2 = np.roll(x, rng.integers(0, len(x)))
            if rng.random() < 0.5:
                x2 = x2[::-1]
            new.append(x2 + rng.normal(0, 0.05, size=x2.shape))
    return (np.concatenate([X, np.array(new, dtype="float32")]),
            np.concatenate([y, np.ones(len(new), dtype=y.dtype)]))


if AUGMENT_COPIES:
    X_tr, y_tr = augment_positives(X_tr, y_tr, AUGMENT_COPIES, np.random.default_rng(SEED))

X_tr, X_va = X_tr[..., None], X_va[..., None]       # (stars, time, 1 channel)

cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
class_weight = {0: cw[0], 1: cw[1]}
print("train positives:", int(y_tr.sum()), "of", len(y_tr), "| class weights:", class_weight)

model = keras.Sequential([
    layers.Input(shape=(X_tr.shape[1], 1)),
    layers.Conv1D(16, 11, activation="relu"),
    layers.MaxPooling1D(4),
    layers.Conv1D(32, 7, activation="relu"),
    layers.MaxPooling1D(4),
    layers.Conv1D(32, 5, activation="relu"),
    layers.GlobalMaxPooling1D(),        # "is there a dip anywhere?"
    layers.Dropout(0.5),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")])
model.summary()

early = keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max",
                                      patience=15, restore_best_weights=True)
model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=100, batch_size=32,
          class_weight=class_weight, callbacks=[early], verbose=2)

p = model.predict(X_va, verbose=0).ravel()
report(y_va, p, name="CNN v1")
(ROOT / "results").mkdir(exist_ok=True)
model.save(ROOT / "results" / "cnn_v1.keras")