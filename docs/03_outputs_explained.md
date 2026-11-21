# 03. What each run prints, and what it means

Written for anyone, including people with no machine-learning background.
Every section shows the command, an example of the output (sections 1 to 6 use illustrative numbers; sections 7 to 9
use the real results of this project), and a plain
reading of it. Numbers in sections 1 to 6 are illustrations only; replace them with your own output if you
want. Sections 7 to 9 show the real results.

---

## 1. Loading the data

```bash
python src/exoplanet/data.py
```

```text
Train: stars=5087, points per star=3197, planets=37, non-planets=5050, NaNs=0
Test:  stars=570,  points per star=3197, planets=5,  non-planets=565,  NaNs=0
```

**In simple words**

- `stars`: how many stars (rows) are in the file.
- `points per star`: how many brightness measurements each star has. Think of it as the length of each star's diary.
- `planets` vs `non-planets`: the correct answers. Planets are extremely rare, which is the main difficulty
  of the project.
- `NaNs`: how many measurements are missing. It should be `0`. If not, preprocessing stops instead of guessing.

**Good sign:** NaNs = 0 and the planet count is small compared with non-planets.

---

## 2. Raw light-curve picture

`results/example_light_curves.png`

Top row: three stars with planets. Bottom row: three stars without. The horizontal axis is time, the vertical
axis is brightness.

**In simple words:** the raw curves are messy. They drift up and down and are noisy, so a planet's tiny dip is
hard or impossible to see by eye. That is the reason we clean the data before training.

---

## 3. Preprocessing and splitting

```bash
PYTHONPATH=src python -m exoplanet.preprocess
```

```text
train: n=4069, planets=30, non-planets=4039
val:   n=1018, planets=7,  non-planets=1011
test:  n=570,  planets=5,  non-planets=565
```

**In simple words:** the training stars were split into two piles: a bigger one to learn from (train) and a
smaller one to check progress (val). The test pile is the locked final exam. Each pile has roughly the same share
of planets.

**Things to notice**

- `val` has only a handful of planets. One extra planet found or missed will move the scores a lot, so do not
  over-trust any single validation number.
- `test` has even fewer. It will be used exactly once, at the end.

**Files created:** `data/processed/dataset_v1.npz` (the cleaned data) and `configs/preprocess_v1.json` (the
settings used).

---

## 4. Before-and-after picture

`results/preprocess_check.png`

Top row: raw curves of three planet stars. Bottom row: the same stars after cleaning.

**In simple words:** after cleaning, the curve should sit flat around zero with the slow drift gone, while short
dips still show up as downward spikes. If the dips disappeared, the cleaning was too aggressive.

---

## 5. Baseline model

```bash
PYTHONPATH=src python -m exoplanet.baseline
```

```text
val planets: 7 of 1018
PR-AUC : 0.150  (random is about 0.007)
ROC-AUC: 0.880
precision: 0.050, recall: 0.710
confusion matrix [[TN FP] [FN TP]]:
 [[ 880 131]
  [   2   5]]
```

How to read each line:

| Line | Plain meaning |
|---|---|
| `val planets: 7 of 1018` | The exam had 1018 stars and 7 were real planets. |
| `PR-AUC` | The main score, from 0 to 1. The number in brackets is what pure guessing would score. Higher than that bracket means the model learned something. |
| `ROC-AUC` | How well the model puts planets above non-planets in its ranking. 0.5 is a coin flip, 1.0 is perfect. Looks generous on rare-class problems. |
| `precision` | Of the stars the model flagged, what fraction were real planets. Low means many false alarms. |
| `recall` | Of the real planets, what fraction the model found. Low means it misses planets. |
| confusion matrix | A 2x2 count table. Top-left: correctly ignored. Top-right: false alarms. Bottom-left: missed planets. Bottom-right: planets found. |

**Why not "accuracy"?** If the model said "no planet" for every star, it would be right more than 99% of the time
and find nothing. Accuracy would look great and mean nothing.

**Reading the example above:** it found 5 of 7 planets (good coverage) but raised 131 false alarms (poor trust).
That is typical for a first baseline. The aim is not perfection, just a reference line for the CNN.

---

## 6. CNN training and result

```bash
PYTHONPATH=src python -m exoplanet.cnn
```

During training you will see lines like:

```text
Epoch 12/100
128/128 - 3s - loss: 0.41 - pr_auc: 0.22 - val_loss: 0.55 - val_pr_auc: 0.18
```

| Item | Plain meaning |
|---|---|
| `Epoch 12/100` | The 12th full pass over the training stars, out of at most 100. |
| `loss` | How wrong the model is on training stars. Should go down. |
| `pr_auc` | The main score on training stars. |
| `val_loss`, `val_pr_auc` | The same two numbers on stars the model never trains on. This is the honest one. |

**What to watch for**

- Training score rising while validation score stalls or falls means **overfitting**: the model is memorizing
  instead of learning. Early stopping halts training and keeps the best version.
- Training usually stops early, well before epoch 100.

At the end, the same block of metrics as the baseline is printed, labeled `[CNN v1]`, and a model file is saved as
`results/cnn_v1.keras`.

**How to compare with the baseline:** look at PR-AUC and recall first. Do not declare a winner from one run,
because validation holds only a few planets. Cross-validation (section 7) gives a fairer comparison.

---

## 7. Cross-validation

```bash
PYTHONPATH=src python -m exoplanet.cv
```

```text
stars=5087, planets=37
fold 1/5: baseline PR-AUC=0.092, cnn PR-AUC=0.122
fold 2/5: baseline PR-AUC=0.039, cnn PR-AUC=0.370
fold 3/5: baseline PR-AUC=0.071, cnn PR-AUC=0.183
fold 4/5: baseline PR-AUC=0.018, cnn PR-AUC=0.257
fold 5/5: baseline PR-AUC=0.050, cnn PR-AUC=0.037

[baseline] PR-AUC 0.054 +/- 0.026 | ROC-AUC 0.715 +/- 0.074
    best-F1 threshold 0.998: precision 0.114, recall 0.135, F1 0.123

[cnn] PR-AUC 0.193 +/- 0.114 | ROC-AUC 0.873 +/- 0.093
    best-F1 threshold 0.698: precision 0.239, recall 0.432, F1 0.308

saved results/cv_results.json
```

(Real output from this project.)

**Our result in plain words:** the CNN scored about 3.5 times higher than the baseline (0.193 against 0.054) and won
4 of the 5 exams. Its score still swings a lot from exam to exam (0.04 to 0.37) because there are only 37 planets, so
the honest summary is "clearly better than the baseline, but still a modest model".

**In simple words:** the labeled stars are cut into 5 groups. The model is trained on 4 and tested on the 5th, five
times, so every planet is tested once while hidden from the model. It is like five different exams.

| Item | Plain meaning |
|---|---|
| `fold k/5` lines | The score on each exam. |
| `PR-AUC x +/- y` | The average score and how much it varied between exams. If `y` is large compared with the gap between the two models, you cannot say one is better. |
| `best-F1 threshold` | The cut-off probability that gives the best balance of precision and recall on the pooled honest scores. This is the number carried into the final test. |
| `precision`, `recall` at that threshold | What you get if you use that cut-off. |

**Files created:** `results/cv_results.json` (summary and threshold) and `results/cv_oof.npz` (every star's honest
score, used by error analysis).

---

## 8. Error analysis

```bash
PYTHONPATH=src python -m exoplanet.errors
```

```text
model=cnn, threshold=0.698
planets found (TP)=16, missed planets (FN)=21, false alarms (FP)=51
saved results/error_analysis_cnn.png
```

(Real output from this project.)

**Our result in plain words:** of the 37 known planets the CNN found 16 and missed 21, and it wrongly flagged 51 other
stars. In the gallery, the false alarms all had very deep repeating dips (some possibly stars eclipsing each other),
and most missed planets were buried in noise.

**In simple words:** it counts the hits, misses and false alarms, then draws a gallery. The top row shows the
non-planets the model was most sure were planets. The bottom row shows the real planets it was most sure were not.
Each title gives the star's row number in the CSV and the model's score.

**How to read the gallery:** look for what the false alarms have in common (two dips of different depths, slow
wiggles, a single giant spike, a sudden jump). Look at whether missed planets have dips that are too small to see.
If many mistakes share one cause, that points to a preprocessing or labeling issue worth fixing or at least
documenting in the README.

---

## 9. Final test

```bash
PYTHONPATH=src python -m exoplanet.final_test
```

Real results from this project (570 test stars, 5 of them planets):

| Item | Value |
|---|---|
| Planets found / missed | 2 / 3 |
| False alarms | 1 |
| Precision / recall | 0.667 / 0.400 |
| PR-AUC / ROC-AUC | 0.639 / 0.993 |
| Planets inside the top 5 / 10 / 20 of the ranked list | 3 / 4 / 5 |

Confusion matrix: TN=564, FP=1, FN=3, TP=2. The numbers are saved in `results/final_test.json`.

**In simple words:** the model is trained once on all labeled stars and examined once on the locked test stars,
using the cut-off picked during cross-validation. With only about 5 planets in the test file, each planet moves
recall by 20 percentage points, so report these numbers with that warning.

**The lock:** if `results/final_test.json` exists, the script refuses to run again. Re-running after seeing the
score and tweaking the model would turn the test set into just another validation set.

---

## 10. Tests

```bash
python -m pytest -q
```

```text
.......                                                  [100%]
7 passed
```

**In simple words:** each dot is one automatic check passing, for example "no star appears in both train and
validation" or "a deep dip survives preprocessing". A letter `F` means something we rely on is broken.

---

## Quick troubleshooting guide

| What you see | Likely meaning | What to do |
|---|---|---|
| Nothing printed, no error | the script file is empty or unsaved | save the file and rerun |
| `FileNotFoundError ... exoTrain.csv` | the CSV is missing or misnamed | check `data/raw/` |
| `ModuleNotFoundError: exoplanet` | run from the wrong folder or missing `PYTHONPATH=src` | run from the project root with `PYTHONPATH=src` |
| `ValueError: NaNs found` | the data has missing values | decide how to handle them on purpose |
| PR-AUC close to the "random" number | the model has learned almost nothing | check preprocessing and class weights |
| Training score high, validation low | overfitting | smaller model, more dropout, augmentation |
