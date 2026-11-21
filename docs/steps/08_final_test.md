# Step 8: Final test

**Goal:** one honest exam on stars the model never influenced.

## Files
`src/exoplanet/final_test.py`. Output: `results/final_test.json`.

## Run (once)
```bash
PYTHONPATH=src python -m exoplanet.final_test
```

## How it works
1. Reads the threshold (0.698) from the cross-validation results. It is never tuned on test data.
2. Trains the CNN on all 5,087 labeled stars (30 epochs, seed 42).
3. Scores the 570 test stars once and saves `final_test.json`.
4. If `final_test.json` exists, the script refuses to run again, to protect the "use the test once" rule.

## Our results (570 stars, 5 planets)
| Item | Value |
|---|---|
| Planets found / missed | 2 / 3 |
| False alarms | 1 |
| Precision / Recall | 0.667 / 0.400 |
| PR-AUC / ROC-AUC | 0.639 / 0.993 |
| Planets in the top 5 / 10 / 20 of the ranked list | 3 / 4 / 5 |

## How to read it
At the cut-off it found 2 of 5 planets, but the ranking is stronger: all 5 planets sit in the top 20 of 570 stars.
With only 5 test planets these numbers are rough. The cross-validation scores (Step 6) are the more reliable
estimate. The test was run twice with identical settings, the second time to add the ranked list.

## Why we did it this way
A test set only measures something if it is used once, after every choice is made.
