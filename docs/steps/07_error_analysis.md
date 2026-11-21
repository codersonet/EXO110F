# Step 7: Error analysis

**Goal:** look at the model's biggest mistakes, because a single score can hide problems.

## Files
`src/exoplanet/errors.py`. Output: `results/error_analysis_cnn.png`.

## Run
```bash
PYTHONPATH=src python -m exoplanet.errors
```

## What it printed
At threshold 0.698: **16 planets found, 21 missed, 51 false alarms** (labeled stars, out-of-fold).

## What the picture showed
- **False alarms (top row):** all six had very deep, repeating dips, some reaching about -1,500 on the scaled axis,
  far deeper than the real planets in the data. Two alternated between deep and shallow dips, which is a typical
  pattern for two stars eclipsing each other (we cannot confirm that from a picture alone). The CNN seems to have
  learned "a big repeating dip means planet" and cannot tell a planet-sized dip from a star-sized one.
- **Missed planets (bottom row):** four of six are extremely noisy, so any dip is buried. One (row 12) shows three clean
  dips and still scored low, and looks similar to a "non-planet" in the top row. That points to hard data and imperfect
  labels, not only a weak model.

## What we decided
Nothing was changed after looking. The findings go into the README as an honest "what the model gets wrong" note.
A possible future improvement is making the model aware of dip depth.

## Why we did it this way
Only labeled, out-of-fold stars were used. The locked test file must never influence a decision.
