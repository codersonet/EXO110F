# Step 2: Data

**Goal:** load the Kaggle light-curve files and count what is in them.

## Files
- `data/raw/exoTrain.csv` and `data/raw/exoTest.csv` (downloaded from Kaggle, never committed to git).
- `src/exoplanet/data.py`: the loader.
- `data/README.md`: where the data came from.

## Run
```bash
python src/exoplanet/data.py
```

## What the output means
Each row of the CSV is one star. Each column is one brightness measurement in time: 3,197 per star. The loader
also turns Kaggle's labels (1 = no planet, 2 = planet) into 0 and 1.

| Our data | Stars | Planets |
|---|---|---|
| Labeled stars (training file) | 5,087 | 37 |
| Test file | 570 | 5 |

So only about 0.7% of the stars are planets. That is the main difficulty of the whole project.

## Why we did it this way
- **Labels as 0/1:** the standard form that machine-learning tools expect.
- **float32 numbers:** half the memory of the default.
- **Stop on missing values:** we never guess a missing brightness value.
- **Test file kept locked:** it is only used once, at the very end.

## Problems we hit
| Problem | Fix |
|---|---|
| Downloaded a notebook (`.ipynb`) instead of the data | Use the dataset page, not the Code tab. |
| `FileNotFoundError: data/raw/exoTrain.csv` | The path depended on the folder the command was run from. We now build it from the file's own location. |
