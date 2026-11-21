# 04. Code reference

For every code file: what it does, and a table of the functions it uses and the purpose of each.
Library functions are written as `module.function`. Functions we wrote ourselves are marked **(ours)**.

All files are documented below, including the final stages: features, models, cross-validation, error analysis,
final test, and tests.

**Contents:** `data.py`, `plot_examples.py`, `preprocess.py`, `check_preprocess.py`, `baseline.py`, `evaluate.py`, `cnn.py`,
`features.py` and `models.py`, `cv.py`, `errors.py`, `final_test.py` (including the ranked lists), `tests/test_data.py`,
`summary.py`, and `README.md`.

---

## `src/exoplanet/data.py`

Loads a CSV and returns arrays.

| Function | Purpose |
|---|---|
| `Path(__file__).resolve().parents[2]` | Builds the project root from this file's own location, so the data path works from any folder. |
| `Path / "data" / "raw"` | Joins folder names into a path safely. |
| `pd.read_csv` | Reads the CSV table into a DataFrame. |
| `df["LABEL"] == 2` | Marks which rows are planets (Kaggle label 2). |
| `.astype(int)` | Turns True/False into 1/0. |
| `df.drop(columns="LABEL")` | Removes the label column, leaving only the flux measurements. |
| `.to_numpy(dtype="float32")` | Converts to a numpy array using less memory than the default 64-bit. |
| `load_split(name)` **(ours)** | Returns `(X, y)` for `"Train"` or `"Test"`. |
| `np.isnan(X).sum()` | Counts missing values for the printed report. |
| `if __name__ == "__main__"` | Runs the report only when the file is executed directly, not when imported. |

---

## `src/exoplanet/plot_examples.py`

Draws raw light curves for a visual check.

| Function | Purpose |
|---|---|
| `y.nonzero()[0][:3]` | Positions of the first three planet stars. |
| `(y == 0).nonzero()[0][:3]` | Positions of the first three non-planet stars. |
| `plt.subplots(2, 3, ...)` | Creates a 2x3 grid of plots. |
| `ax.plot`, `ax.set_title` | Draws one curve and titles it. |
| `fig.supxlabel`, `fig.supylabel` | Shared axis labels for the whole figure. |
| `plt.tight_layout`, `plt.savefig` | Tidies spacing, then writes the image to `results/`. |

---

## `src/exoplanet/preprocess.py`

Cleans every star, splits the data, and saves the result.

| Function | Purpose |
|---|---|
| `CONFIG` (dict) | One place holding all settings: window size, clip level, split fraction, seed. |
| `preprocess(X, cfg)` **(ours)** | Cleans an array of stars, one row at a time. Steps below. |
| `np.isnan(X).any()` + `raise ValueError` | Stops if any value is missing, so we never invent data silently. |
| `median_filter(X, size=(1, 101), mode="nearest")` | Estimates each star's slow trend. `(1, 101)` means smooth along time only, never across stars. `mode="nearest"` handles the ends. |
| `X - trend` | Removes the trend, leaving short-term changes (including dips). |
| `np.median(R, axis=1, keepdims=True)` | The typical level of each star, kept as a column so it can be subtracted. |
| `np.abs`, `np.median` and `1.4826 *` | Computes the robust noise level (MAD scaled to match a standard deviation). |
| `sigma[sigma == 0] = 1.0` | Avoids dividing by zero on a perfectly flat star. |
| `(R - med) / sigma` | Scales each star to a common noise level. |
| `np.minimum(R, 4.0)` | Clips only large upward spikes; dips (negative values) are left alone. |
| `.astype("float32")` | Saves memory. |
| `StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` | Makes one train/validation split that keeps the planet share equal and is repeatable. |
| `.split(X, y)` | Produces the index lists for the two piles. |
| `np.savez_compressed` | Saves all arrays and the split indices into one compressed file. |
| `json.dumps(CONFIG, indent=2)` + `write_text` | Writes the settings to `configs/preprocess_v1.json`. |
| `Path.mkdir(parents=True, exist_ok=True)` | Creates output folders if missing. |
| `main()` **(ours)** | Runs the full load, split, preprocess, save, and print. |

---

## `src/exoplanet/check_preprocess.py`

Compares raw and cleaned curves.

| Function | Purpose |
|---|---|
| `np.load(...npz)` | Opens the saved dataset. |
| `d["train_idx"][d["y_train"] == 1][:3]` | Finds the original CSV row numbers of three training planets. |
| `X_raw[i:i+1]` | Keeps the 2D shape (one row) so `preprocess` accepts it. |
| `preprocess(...)[0]` **(ours, reused)** | Cleans one star for the "after" plot. |
| `plt.subplots`, `plt.savefig` | Draws raw (top) and cleaned (bottom) curves. |

---

## `src/exoplanet/baseline.py`

A simple model for the CNN to beat.

| Function | Purpose |
|---|---|
| `features(X)` **(ours)** | Turns each star into a short list of numbers. |
| `X.min(1)`, `X.max(1)`, `X.std(1)` | Lowest point, highest point, spread. |
| `scipy.stats.skew`, `kurtosis` | Whether the values lean to one side, and how heavy the tails are. |
| `np.percentile(X, p, axis=1)` | Value levels at 1%, 5%, 25%, 50%, 75%, 95%, 99%. |
| `np.fft.rfft(X, axis=1)` | Frequency content of the signal. |
| `np.abs(...)[:, 1:101]` | Strength of the 100 lowest frequencies, skipping the zero-frequency average. |
| `spec / (spec.sum(1, keepdims=True) + 1e-9)` | Normalizes the spectrum so only its shape matters; `1e-9` prevents division by zero. |
| `np.column_stack` | Joins all features into one table. |
| `make_pipeline` | Chains the scaler and the model so they are fit and applied together. |
| `StandardScaler` | Rescales each feature to mean 0 and spread 1. |
| `LogisticRegression(class_weight="balanced", max_iter=2000)` | The classifier; planets get a larger weight; allows enough iterations to converge. |
| `.fit`, `.predict_proba(...)[:, 1]` | Trains, then returns each star's planet probability. |
| `(p >= 0.5).astype(int)` | Turns probabilities into yes/no using the placeholder threshold. |
| `average_precision_score`, `roc_auc_score`, `precision_score`, `recall_score`, `confusion_matrix` | The evaluation metrics. |

---

## `src/exoplanet/evaluate.py`

Shared scoring so every model is judged the same way.

| Function | Purpose |
|---|---|
| `report(y, p, thr, name)` **(ours)** | Prints positives count, PR-AUC, ROC-AUC, precision, recall and the confusion matrix. |
| `zero_division=0` | Prevents an error when the model flags nothing, so precision is 0 instead of undefined. |

---

## `src/exoplanet/cnn.py`

The neural network.

| Function | Purpose |
|---|---|
| `keras.utils.set_random_seed(42)` | Fixes Python, numpy and TensorFlow randomness together. |
| `augment_positives(X, y, copies, rng)` **(ours)** | Makes extra training planets by shifting, flipping and adding small noise. Only touches training data. |
| `np.roll` | Circular time shift of a light curve. |
| `x[::-1]` | Reverses time. |
| `rng.normal(0, 0.05, size)` | Adds small random noise. |
| `np.random.default_rng(seed)` | A seeded random generator. |
| `X[..., None]` | Adds the channel axis the Conv1D layer expects: `(stars, time, 1)`. |
| `compute_class_weight("balanced", ...)` | Computes how much heavier a planet should count. |
| `keras.Sequential([...])` | Builds the network as a simple stack of layers. |
| `layers.Input(shape=...)` | Declares the input size. |
| `layers.Conv1D(filters, kernel)` | Sliding pattern detectors along time. Kernel sizes 11, 7, 5 look at progressively smaller windows. |
| `layers.MaxPooling1D(4)` | Shrinks the sequence by 4x, keeping the strongest response. |
| `layers.GlobalMaxPooling1D()` | Keeps the strongest response anywhere, giving one number per detector. |
| `layers.Dropout(0.5)` | Randomly turns off half the units in training to reduce memorizing. |
| `layers.Dense(16, "relu")` | A small decision layer. |
| `layers.Dense(1, "sigmoid")` | Final planet probability between 0 and 1. |
| `model.compile(...)` | Chooses optimizer, loss and metric. |
| `keras.optimizers.Adam(1e-3)` | The learning method with a small learning rate. |
| `loss="binary_crossentropy"` | The standard loss for yes/no problems. |
| `keras.metrics.AUC(curve="PR")` | PR-AUC tracked during training. |
| `keras.callbacks.EarlyStopping(...)` | Stops when validation PR-AUC stops improving; keeps the best weights. |
| `model.fit(...)` | Trains the model. |
| `model.predict(...).ravel()` | Gets probabilities for validation stars as a flat list. |
| `model.save(...)` | Stores the trained model in `results/`. |

---

## `src/exoplanet/features.py` and `src/exoplanet/models.py`

Reusable building blocks, so `cv.py` and `final_test.py` can import them without running a training job.

| Function | Purpose |
|---|---|
| `features(X)` **(ours)** | Same feature table as in `baseline.py` (summary stats, percentiles, 100 lowest FFT frequencies). |
| `build_cnn(n_points)` **(ours)** | Builds and compiles the same CNN as in `cnn.py` and returns it. |
| `augment_positives(X, y, copies, rng)` **(ours)** | Makes shifted, flipped, noisy copies of training planets. |

---

## `src/exoplanet/cv.py`

Fair comparison of the baseline and the CNN, plus threshold selection.

| Function | Purpose |
|---|---|
| `np.concatenate([X_train, X_val])` | Pools all labeled stars so cross-validation uses every planet. The test file stays locked. |
| `RepeatedStratifiedKFold(n_splits, n_repeats, random_state)` | Makes the folds, each with a similar planet share, optionally repeated with new shuffles. |
| `cv.split(X, y)` | Yields train and held-out index lists for each fold. |
| `make_pipeline(StandardScaler(), LogisticRegression(...))` | Fresh baseline model in every fold. |
| `keras.utils.set_random_seed(SEED + k)` | Different but repeatable randomness per fold. |
| `compute_class_weight` | Class weights computed from that fold's training part only. |
| `build_cnn(...)`, `model.fit(..., epochs=EPOCHS)` | Fresh CNN in every fold, trained for a fixed number of epochs. |
| `keras.backend.clear_session()` | Frees memory between folds. |
| `oof[m][rep, te] = p` | Stores each star's out-of-fold score. |
| `average_precision_score`, `roc_auc_score` | Per-fold PR-AUC and ROC-AUC. |
| `np.mean`, `np.std` | The `mean +/- std` across folds. |
| `best_threshold(y, p)` **(ours)** | Uses `precision_recall_curve` and picks the threshold with the highest F1. |
| `np.tile(y, N_REPEATS)` | Repeats labels to match pooled predictions across repeats. |
| `np.savez(... "cv_oof.npz")` | Saves each star's out-of-fold score for error analysis. |
| `json.dumps(summary)` | Writes `results/cv_results.json`. |

---

## `src/exoplanet/errors.py`

| Function | Purpose |
|---|---|
| `np.load("cv_oof.npz")` | Loads honest scores for every labeled star. |
| `json.loads(...)["threshold_max_f1"]` | Loads the threshold chosen in `cv.py`. |
| `np.where((pred == 1) & (y == 0))` | Finds false alarms (FP). |
| `np.where((pred == 0) & (y == 1))` | Finds missed planets (FN). |
| `np.argsort(-p[idx])` | Sorts false alarms by confidence, highest first. |
| `np.argsort(p[idx])` | Sorts misses by confidence, lowest first. |
| `csv_row = concat(train_idx, val_idx)` | Maps each star back to its row in `exoTrain.csv`. |
| `plt.subplots(2, TOP, squeeze=False)`, `ax.axis("off")` | Draws the gallery and hides unused panels. |

---

## `src/exoplanet/final_test.py`

| Function | Purpose |
|---|---|
| `OUT.exists()` + `sys.exit(...)` | The lock: refuses to run twice. |
| `json.loads(cv_results.json)` | Reads the chosen threshold, epochs, and augmentation setting. |
| `np.concatenate([train, val])` | Trains on all labeled stars. |
| `net.fit(...)` or `model.fit(...)` | Trains the chosen model once. |
| `report(y_te, p, thr, name)` | Prints the standard metrics block. |
| `confusion_matrix(..., labels=[0, 1]).ravel()` | Gets TN, FP, FN, TP as four numbers. |
| `ranked_table(scores, labels, csv_rows, source)` **(ours)** | Builds a table of stars sorted from most to least planet-like, with rank, CSV row, score, flag and known label. |
| `np.argsort(-scores)` | The order from the highest score to the lowest. |
| `pd.DataFrame({...})`, `.to_csv(...)` | Builds and saves `ranked_candidates_test.csv` and `ranked_candidates_labeled_cv.csv`. |
| `np.where(labels == 1, "planet", "non-planet")` | Turns 0/1 labels into readable text. |
| `scores >= thr` | Marks which stars count as flagged candidates. |
| `np.load("cv_oof.npz")[MODEL]` | The out-of-fold scores used for the labeled-star ranked list. |
| `y_te[order_te[:k]].sum()` | Counts the known planets inside the top k of the test list. |
| `OUT.write_text(json.dumps(result))` | Saves the final result and creates the lock. |

---

## `tests/test_data.py`

| Test | Purpose |
|---|---|
| `test_no_star_in_both_train_and_validation` | Uses `np.intersect1d` to confirm the split has no overlap. |
| `test_shapes_and_labels` | Row counts match, labels are 0 or 1, and the time length is the same everywhere. |
| `test_every_split_has_planets` | No split ended up with zero planets. |
| `test_no_nans_after_preprocessing` | Cleaning produced no missing values. |
| `test_transit_dip_survives_preprocessing` | A synthetic deep dip is still clearly negative after cleaning. |
| `test_upward_spike_is_clipped_but_dips_are_not` | Upward spikes are capped at 4. |
| `test_preprocessing_is_per_star` | A star's result does not depend on other stars in the batch. |

Run with `python -m pytest -q`.

---

## `src/exoplanet/summary.py`

Prints and saves the one-page summary of the finished run. It trains nothing and reads only files made by earlier steps.

| Function | Purpose |
|---|---|
| `json.loads(...read_text())` | Reads `final_test.json` and the preprocessing config. |
| `np.load(...npz)` | Reads the stored split, to count planets and non-planets in each part. |
| `pd.read_csv` | Reads the ranked test list. |
| `counts(split)` **(ours)** | Returns the number of planets and non-planets in one split. |
| `tp / (tp + fp)`, `tp / (tp + fn)` | Computes precision and recall from the confusion-matrix counts, guarding against dividing by zero. |
| `ranked.head(TOP).iterrows()` | Walks through the top-ranked stars to print them. |
| f-strings with `:>7`, `:.3f` | Lines up the table columns and rounds the numbers. |
| `Path.write_text` | Saves the same text to `results/final_summary.txt`. |

---

## `README.md`

The project front page: what it is and is not, the data it uses, how it works, how to run it, real results, an honest
assessment of the model, why Lightkurve is not used, reproducibility, limitations and learnings.
