# Step 3: Preprocessing

**Goal:** make every star comparable, so the model learns the shape of a dip and not how bright the star is.

## Files
`src/exoplanet/preprocess.py`, `check_preprocess.py`, `configs/preprocess_v1.json`, output `data/processed/dataset_v1.npz`.

## Run
```bash
PYTHONPATH=src python -m exoplanet.preprocess
PYTHONPATH=src python -m exoplanet.check_preprocess
```

## What happens to each star (one star at a time)
1. **Detrend:** a median filter (window 101) finds the slow drift, which is subtracted. Short dips survive.
2. **Scale:** subtract the typical level, divide by a robust noise estimate, so every star has the same scale.
3. **Clip:** cut only huge upward spikes (above 4 noise units). Dips go downward, so they are untouched.
4. **Split:** the labeled stars are split 80/20 into train and validation, keeping the planet share equal. Seed 42.

## What the output means
- `train / val / test` lines show how many planets and non-planets are in each part.
- `results/preprocess_check.png` shows raw curves above, cleaned curves below. The drift should be gone and the dips
  should still be there.

## Why we did it this way
| Choice | Reason |
|---|---|
| Median instead of average | An average is pulled down by a dip and erases it. A median ignores it. |
| Robust noise estimate | Outliers and dips cannot distort it. |
| Upward-only clipping | Spikes from glitches go up. Planet dips go down. |
| One star at a time | Nothing is shared between train, validation and test, so there is no leakage. |
| Settings saved in JSON | A result can always be traced back to the exact settings. |

## Limit to remember
A window of 101 steps can flatten very long dips.
