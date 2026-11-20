import numpy as np
from scipy.stats import skew, kurtosis

def features(X):
    f = [X.min(1), X.max(1), X.std(1), skew(X, axis=1), kurtosis(X, axis=1)]
    f += [np.percentile(X, p, axis=1) for p in (1, 5, 25, 50, 75, 95, 99)]
    spec = np.abs(np.fft.rfft(X, axis=1))[:, 1:101]
    spec = spec / (spec.sum(1, keepdims=True) + 1e-9)
    return np.column_stack(f + [spec])