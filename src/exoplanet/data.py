from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw")

def load_split(name):
    """name is 'Train' or 'Test'. Returns flux array X and binary labels y (1 = planet)."""
    df = pd.read_csv(RAW / f"exo{name}.csv")
    y = (df["LABEL"] == 2).astype(int).to_numpy()   # Kaggle: 1 = no planet, 2 = planet
    X = df.drop(columns="LABEL").to_numpy(dtype="float32")
    return X, y

if __name__ == "__main__":
    for name in ["Train", "Test"]:
        X, y = load_split(name)
        print(f"{name}: stars={X.shape[0]}, points per star={X.shape[1]}, "
              f"planets={y.sum()}, non-planets={(y == 0).sum()}, NaNs={int(np.isnan(X).sum())}")