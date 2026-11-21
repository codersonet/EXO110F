# 01. Project flow

The big picture of the whole project, from raw data to the final write-up.
All stages are now built. Each has a file, an output, and an entry in the other four guides.

## The goal in one paragraph

Telescopes like Kepler record how bright a star is over time. When a planet passes in front of its star, the
brightness dips slightly. We train a small neural network (a 1D CNN) to look at a star's brightness record and say
how likely it is to contain a planet-like dip. The output is a ranked list of *candidates*. It is **not** a
confirmation of any planet, which would need follow-up observations by astronomers.

## The pipeline

```text
 raw CSV files
     |
     v
 [1] Load data ........................ data.py
     |
     v
 [2] Look at it by eye ................ plot_examples.py
     |
     v
 [3] Clean and normalize .............. preprocess.py   (+ check_preprocess.py)
     |
     v
 [4] Split into train / validation .... preprocess.py
     |
     v
 [5] Baseline model ................... baseline.py     (simple, to beat)
     |
     v
 [6] CNN model ........................ cnn.py          (the main model)
     |
     v
 [7] Fair comparison .................. cv.py (cross-validation)
     |
     v
 [8] Study the mistakes ............... errors.py
     |
     v
 [9] One final exam on the test file .. final_test.py
     |
     v
 [10] Write-up ........................ README.md, docs/, tests/
```

## Stage by stage

| # | Stage | Question it answers | Files | Output | Status |
|---|---|---|---|---|---|
| 0 | Setup | Can anyone rebuild this project? | `requirements.txt`, `.gitignore`, folders, GitHub repo | a working environment | done |
| 1 | Load | What does the raw data look like? | `src/exoplanet/data.py` | star count, class counts, NaN count | done |
| 2 | Inspect | Can I see a planet in the raw data? | `plot_examples.py` | `results/example_light_curves.png` | done |
| 3 | Preprocess | How do I make every star comparable? | `preprocess.py` | `data/processed/dataset_v1.npz`, `configs/preprocess_v1.json` | done |
| 3b | Verify | Did cleaning keep the dips? | `check_preprocess.py` | `results/preprocess_check.png` | done |
| 4 | Baseline | What score can a simple model reach? | `baseline.py` | PR-AUC, recall, confusion matrix | done |
| 5 | CNN | Can a neural network do better? | `evaluate.py`, `cnn.py` | same metrics, saved model | done |
| 6 | Cross-validation | Is the comparison fair and stable? | `features.py`, `models.py`, `cv.py` | metrics averaged over folds, chosen threshold, `results/cv_results.json`, `results/cv_oof.npz` | done |
| 7 | Error analysis | Where does the model go wrong? | `errors.py` | `results/error_analysis_<model>.png` | done |
| 8 | Final test | How well does it do on stars it never saw? | `final_test.py` | `results/final_test.json` (written once, then locked) | done |
| 9 | Documentation and tests | Can others understand and reproduce it? | `README.md`, these guides, `tests/test_data.py` | published repo, passing tests | done |

## Rules that hold across the whole flow

1. **Test data stays locked.** `exoTest.csv` is only used once, at the very end.
2. **Split first, then anything that copies or learns.** Augmentation and scaling statistics only ever come from
   the training part.
3. **Every setting is written down.** Preprocessing settings are saved to a JSON file and the random seed is fixed.
4. **Accuracy is never the headline.** Planets are so rare that a model saying "no planet" every time would score
   over 99% accuracy.
5. **Look at mistakes, not just scores.** A single number can hide bugs.
6. **Words matter.** A prediction is a "candidate", never a "detection" or "confirmed planet".

## What each stage hands to the next

```text
exoTrain.csv, exoTest.csv
   -> (X, y) arrays                                  [data.py]
   -> cleaned arrays + train/val/test split          [preprocess.py -> dataset_v1.npz]
   -> trained model + validation scores              [baseline.py, cnn.py]
   -> averaged scores + threshold                    [cross-validation]
   -> list of mistakes and what they teach           [error analysis]
   -> final test scores                              [final test]
   -> README with results, limits and how to repeat  [documentation]
```

## How to run the finished parts, in order

```bash
cd ~/Desktop/EXO110F
source .venv/bin/activate

python src/exoplanet/data.py                          # 1  load and count
PYTHONPATH=src/exoplanet python src/exoplanet/plot_examples.py   # 2  raw plots
PYTHONPATH=src python -m exoplanet.preprocess         # 3  clean and split
PYTHONPATH=src python -m exoplanet.check_preprocess   # 3b verify
PYTHONPATH=src python -m exoplanet.baseline           # 4  baseline
PYTHONPATH=src python -m exoplanet.cnn                # 5  CNN
PYTHONPATH=src python -m exoplanet.cv                 # 6  cross-validation
PYTHONPATH=src python -m exoplanet.errors             # 7  error analysis
PYTHONPATH=src python -m exoplanet.final_test         # 8  final test, ONCE
python -m pytest -q                                   # checks
```
