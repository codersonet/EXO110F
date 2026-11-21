# Step 6: Cross-validation

**Goal:** compare the baseline and the CNN fairly. A single validation set holds only about 7 planets, so one star
can swing the score.

## Files
`src/exoplanet/cv.py` (uses `features.py` and `models.py`). Outputs: `results/cv_results.json`, `results/cv_oof.npz`.

## Run
```bash
PYTHONPATH=src python -m exoplanet.cv
```

## How it works
The 5,087 labeled stars are cut into 5 groups with the same planet share in each. Five times, 4 groups teach the
model and 1 group is the exam, so every star is examined exactly once. The test file is not used. "Out-of-fold"
scores are scores from a model that never saw that star.

## Our results (PR-AUC per fold)
| Fold | Baseline | CNN |
|---|---|---|
| 1 | 0.092 | 0.122 |
| 2 | 0.039 | 0.370 |
| 3 | 0.071 | 0.183 |
| 4 | 0.018 | 0.257 |
| 5 | 0.050 | 0.037 |

| Summary | Baseline | CNN |
|---|---|---|
| PR-AUC (mean ± spread) | 0.054 ± 0.026 | 0.193 ± 0.114 |
| ROC-AUC | 0.715 ± 0.074 | 0.873 ± 0.093 |
| Precision at chosen threshold | 0.114 | 0.239 |
| Recall at chosen threshold | 0.135 | 0.432 |
| Threshold (best F1) | 0.998 | 0.698 |

## How to read it
The CNN won 4 of 5 folds and is about 3.5 times better than the baseline on PR-AUC, but the scores swing a lot between
folds because there are so few planets. Conclusion: the CNN is clearly better, and still a modest model.

## Why we did it this way
- **Fixed 30 epochs, no early stopping:** early stopping watches a validation set, and watching the exam fold would
  leak its answers.
- **Threshold chosen from out-of-fold scores:** never from the test file.
- **Seeds fixed:** the run is repeatable.
