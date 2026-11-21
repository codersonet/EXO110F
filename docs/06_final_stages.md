# 06. The final stages

Guides 01 to 05 mark cross-validation, error analysis, the final test and the ranked list as "planned".
This guide covers them now that they exist. For each stage: what it is for, how to run it, what the output means in
plain words, which functions it uses, and why.

```text
dataset_v1.npz -> cv.py -> cv_results.json + cv_oof.npz -> choose model -> errors.py
                                                                      \-> final_test.py -> final_test.json
                                                                                        -> ranked lists (CSV)
```

---

## 1. Cross-validation: `cv.py` (with `features.py` and `models.py`)

**Purpose.** Compare the baseline and the CNN fairly. One small validation set contains only a handful of planets,
so one star can swing the score. Cross-validation uses every labeled star as an exam question exactly once.

**Run.** `PYTHONPATH=src python -m exoplanet.cv`

**Output and how to read it**

```text
fold 1/5: baseline PR-AUC=..., cnn PR-AUC=...
[cnn] PR-AUC 0.xxx +/- 0.xxx | ROC-AUC 0.xxx +/- 0.xxx
    best-F1 threshold 0.xxx: precision 0.xxx, recall 0.xxx, F1 0.xxx
saved results/cv_results.json
```

| Item | Plain meaning |
|---|---|
| `fold k/n` | The labeled stars are cut into 5 groups. Each round, 4 groups teach the model and 1 group is the exam. |
| `PR-AUC mean +/- std` | The average score over the exams, and how much it wobbled between them. A big wobble means the result is shaky. |
| `best-F1 threshold` | The score cut-off above which a star is called a candidate. It is chosen to balance false alarms and misses, using only held-out predictions. |
| `precision` / `recall` at that threshold | Trust in the alarms / share of real planets found. |

**Files written.** `results/cv_results.json` (the numbers, plus settings such as seed, epochs and augmentation) and
`results/cv_oof.npz` (every star's out-of-fold score). "Out-of-fold" means the star was scored by a model that never
saw it in training.

**Functions and why**

| Function | Why |
|---|---|
| `RepeatedStratifiedKFold` | Splits into folds that keep the planet share equal; repeating it with different shuffles gives more stable averages. |
| `features(X)` in `features.py` | The baseline's per-star features, moved to its own module so `cv.py` can import them without running `baseline.py`. Features are computed per star, so computing them once for all stars leaks nothing. |
| `build_cnn(n_points)` and `augment_positives(...)` in `models.py` | The CNN and augmentation, importable and reusable. |
| Fixed epochs instead of early stopping | Early stopping needs a validation set to watch. Watching the exam fold would leak its answers. |
| `keras.backend.clear_session()` | Frees memory between folds. |
| `keras.utils.set_random_seed(SEED + k)` | A different but repeatable seed for each fold. |
| `precision_recall_curve` + F1 | Finds the threshold with the best balance on the pooled out-of-fold scores. Pick a different rule if you care more about recall. |

**Patch needed in `cv.py`.** Add this right after the line `(ROOT / "results").mkdir(exist_ok=True)`:

```python
np.savez(ROOT / "results" / "cv_oof.npz",
         baseline=oof["baseline"].mean(axis=0), cnn=oof["cnn"].mean(axis=0))
```

`errors.py` and the ranked list read this file. Averaging over repeats gives one score per star.

**Choosing the model.** Open `results/cv_results.json` and compare `pr_auc_mean` for the two models, keeping an eye on
`pr_auc_std`. If the CNN is not clearly better than the baseline, the honest conclusion is that the simpler model
is enough, and that is a valid result. Whatever you pick, write the reason in the README, then set the same
`MODEL = "..."` in `errors.py` and `final_test.py`.

---

## 2. Error analysis: `errors.py`

**Purpose.** A single score hides bugs. Looking at the biggest mistakes shows what fools the model.

**Run.** `PYTHONPATH=src python -m exoplanet.errors`

**Output**

```text
model=cnn, threshold=0.xxx
planets found (TP)=..., missed planets (FN)=..., false alarms (FP)=...
saved results/error_analysis_cnn.png
CSV rows of the plotted false alarms: [...]
CSV rows of the plotted missed planets: [...]
```

**How to read the picture.** Top row: stars the model was most sure were planets but are not (false alarms).
Bottom row: real planets the model gave the lowest scores (misses). Titles show the CSV row and the score.

| What you see | What it usually means |
|---|---|
| A second, different-depth dip | an eclipsing binary (two stars), a classic planet impostor |
| Big slow wiggles | the star's own variability |
| One huge spike, or a sudden step | an instrument glitch or outlier |
| Missed planet with a dip lost in noise | the signal is too weak for this data |
| Missed planet whose dip looks removed | the median filter (window 101) may have flattened it |

If many mistakes share one pattern, look for a preprocessing or labeling issue before changing the model.

**Why only labeled out-of-fold data.** The locked test file must not influence any decision.

---

## 3. Final test and ranked list: `final_test.py`

**Purpose.** One last exam on `exoTest.csv`, with the model and threshold already decided, plus the final ranked
list of candidates.

**Run (once).** `PYTHONPATH=src python -m exoplanet.final_test`

**What it does**

1. Reads the threshold from `cv_results.json`. It is never tuned on test data.
2. Trains the chosen model on all labeled stars (train plus validation).
3. Scores the test stars once, prints the metrics, and saves `results/final_test.json`.
4. If `final_test.json` already exists, the script refuses to run. This protects the "use the test once" rule.
5. Writes two ranked lists (below).

**Output and how to read it**

| Item | Plain meaning |
|---|---|
| PR-AUC, ROC-AUC, precision, recall, confusion matrix | same meaning as in guide 03, now on stars the model never influenced |
| `planets found: a of b; false alarms: c` | the headline in plain words |
| `known planets inside the top k` | if you only looked at the k highest-ranked stars, how many real planets would you have found |

With only a few planets in the test file, every number is rough. One planet more or less changes recall a lot. Say
so whenever you quote it.

**The ranked lists**

| File | What it contains |
|---|---|
| `results/ranked_candidates_test.csv` | held-out test stars, sorted from most to least planet-like |
| `results/ranked_candidates_labeled_cv.csv` | the labeled stars, each scored out-of-fold, sorted the same way |

| Column | Meaning |
|---|---|
| `rank` | 1 is the most planet-like star in that list |
| `source` | which file and which kind of score |
| `csv_row` | 0-based data row in the original CSV (the header is not counted, so spreadsheet line = `csv_row` + 2) |
| `planet_score` | the model's 0 to 1 score. It is a ranking score, not a guaranteed probability. |
| `flagged_as_candidate` | whether the score reached the CV-chosen threshold |
| `known_label` | the Kaggle label, shown so you can check the ranking |

**Read the lists honestly.** Every star in this dataset already has a label, so this is not a hunt for new planets.
The list shows how high the model ranks the known planets. A good ranking puts planets near the top. Items flagged
as candidates would still need follow-up observations by astronomers.

---

## 4. Automatic checks: `tests/test_data.py`

Run with `python -m pytest -q`. The `pytest.ini` file points pytest at the `tests/` folder (without it, pytest
tries to run `final_test.py` as a test), and `pytest` must be listed in `requirements.txt`.

| Test | What it protects against |
|---|---|
| no star in both train and validation | data leakage |
| shapes and labels are consistent, labels are only 0 or 1 | broken arrays |
| every split has planets | a split that cannot be scored |
| no NaNs after preprocessing | silent bad data |
| a deep, short dip survives preprocessing | cleaning that erases transits |
| an upward spike is clipped but dips are not | clipping the wrong side |
| preprocessing one star gives the same result alone or with others | leakage between stars |

---

## 5. Final checklist before publishing

1. `results/cv_results.json` and `results/final_test.json` exist, and their numbers are pasted into the README.
2. The README placeholders (`<fill>`) are all replaced, including mission and flux details from the Kaggle data card.
3. Python version, package versions (`pip freeze > requirements-lock.txt`), seed, hardware and git commit are recorded.
4. `.gitignore` still excludes `data/raw/`, `data/processed/`, `.venv/` and `*.keras`.
5. A license is chosen and a `LICENSE` file added. The data's own terms are respected and the data is not committed.
6. Wording check: every output is called a "candidate", never a "detection" or "confirmed planet".
7. Check that the example numbers in sections 1 to 6 of `docs/03_outputs_explained.md` are labeled as examples;
   sections 7 to 9 already hold the real results.
