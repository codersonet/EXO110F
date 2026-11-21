# Step 5: The CNN

**Goal:** train a small neural network that scans a star's brightness record for dips.

## Files
`src/exoplanet/evaluate.py` (shared scoring), `cnn.py` (first run), `models.py` (reusable `build_cnn`).

## Run
```bash
PYTHONPATH=src python -m exoplanet.cnn
```

## The network in plain words
```text
brightness record
 -> 3 convolution layers   (slide small dip-detectors along time; pooling shrinks the record)
 -> global max pooling     ("what is the strongest dip anywhere?")
 -> dropout                (randomly switch off half the units so it cannot memorize)
 -> small dense layer -> one number between 0 and 1 (the planet score)
```

## Settings we used
Class weights (planets count much more), Adam optimizer, up to 100 epochs with early stopping in the single-split run,
and **30 fixed epochs** in cross-validation. No augmentation (`augment_copies = 0`).

## What the training lines mean
`loss` and `pr_auc` are scores on training stars. `val_loss` and `val_pr_auc` are scores on stars it never trains on,
which is the honest one. If training improves while validation does not, the network is memorizing.

## Why we did it this way
| Choice | Reason |
|---|---|
| 1D convolutions | Our data is a 1D signal. The same dip is found wherever it happens. |
| Small network, dropout | Only 37 planets exist, so a big network would memorize them. |
| Global max pooling | The question is "is there a dip anywhere?". |
| Class weights | Otherwise the model learns to say "no planet" for everything. |

## Result
The CNN beat the baseline in cross-validation (next step).
