# Step 9: Ranked list and summary

**Goal:** produce the final list of candidates and a one-page summary for readers.

## Files
Made by `final_test.py`: `results/ranked_candidates_test.csv` and `results/ranked_candidates_labeled_cv.csv`.
Made by `src/exoplanet/summary.py`: `results/final_summary.txt`.

## Run
```bash
PYTHONPATH=src python -m exoplanet.summary
```

## The two lists
| File | Contains |
|---|---|
| `ranked_candidates_test.csv` | the 570 held-out test stars, most to least planet-like |
| `ranked_candidates_labeled_cv.csv` | the 5,087 labeled stars, each scored by a model that never trained on it |

Columns: `rank`, `source`, `csv_row` (0-based data row, header not counted), `planet_score`,
`flagged_as_candidate` (score at or above 0.698) and `known_label`.

## How to read them
Every star in this dataset already has a label, so this is not a search for new planets. The list shows how high the
model ranks the known planets. The Kaggle files have no star names (such as KIC numbers), so rows are the identifiers.
A flagged star is a **candidate**. Confirming a planet needs follow-up observations by astronomers.

## The summary
`summary.py` prints the run settings, split counts, threshold, test metrics, confusion matrix and the top ranked
stars. It trains nothing and reads only files that earlier steps made.
