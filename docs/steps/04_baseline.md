# Step 4: Baseline

**Goal:** a simple model that gives the neural network a score to beat.

## Files
`src/exoplanet/baseline.py` (later `features.py` holds the same feature code for reuse).

## Run
```bash
PYTHONPATH=src python -m exoplanet.baseline
```

## How it works
1. Describe each star with about 112 numbers: lowest and highest point, spread, percentiles, and how much of the
   signal is at each of the 100 lowest frequencies (an FFT).
2. Rescale the numbers (`StandardScaler`).
3. Fit a logistic regression with `class_weight="balanced"`, so each rare planet counts for much more than a
   non-planet.

## What the output means
`PR-AUC`, `ROC-AUC`, precision, recall and a confusion matrix, as explained in `03_outputs_explained.md`.
The honest, fair number comes from cross-validation (Step 6).

| Baseline, 5-fold cross-validation | Value |
|---|---|
| PR-AUC | 0.054 ± 0.026 |
| ROC-AUC | 0.715 ± 0.074 |
| Planets found at its threshold | about 13% |

Pure guessing would give a PR-AUC of about 0.007, so the baseline learned a little, but not much.

## Why we did it this way
A baseline tells you whether the neural network is worth its complexity. A simple model is also hard to overfit.
