import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def build_cnn(n_points):
    model = keras.Sequential([
        layers.Input(shape=(n_points, 1)),
        layers.Conv1D(16, 11, activation="relu"),
        layers.MaxPooling1D(4),
        layers.Conv1D(32, 7, activation="relu"),
        layers.MaxPooling1D(4),
        layers.Conv1D(32, 5, activation="relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dropout(0.5),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")])
    return model

def augment_positives(X, y, copies, rng):
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