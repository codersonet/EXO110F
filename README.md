### EXO110F
EXO110f sees when a planet passes in front of its star, the star dims by a tiny fraction. A 1D CNN reads NASA Kepler light curves and learns to tell a real planet's dip from look-alike, like two stars eclipsing each other. This approach to find Kepler-90i, the eighth planet in its system.

---

A small, beginner-friendly, Keras-based exoplanet **candidate classifier**, inspired by Google's AstroNet
([Shallue & Vanderburg, arXiv:1712.05044](https://arxiv.org/abs/1712.05044)).

> **Important:** this is a learning and reproduction project. It does **not** detect new planets and does **not**
> confirm anything. Its output is a **ranked list of candidates**: signals that *look* planet-like and may deserve a closer look.
> Confirmation needs extra observations and expert review.

---

## 1. What the project does

Given a star's archived brightness measurements (a *light curve*) and a repeating dip that has already been detected
(with a known period and epoch), the model estimates how likely that dip is a **planet candidate** rather than a **false positive**
(eclipsing binary, instrument glitch, stellar variability, etc.).

It works on **saved archive data**. It does not watch stars live, and it does not search for dips itself.

**One-line summary:** light curve + known dip → fold → two views → small CNN → score between 0 and 1.

---

## 2. Input

Each example (one star / one signal) needs:

| Field | Meaning |
|---|---|
| `source_id` | Star identifier (for example a KIC or KOI ID) |
| `mission` | Kepler or TESS |
| `period` | Orbital period of the repeating dip (days) |
| `epoch` | Time of one dip (same time system as the light curve) |
| `cadence` | Time between measurements |
| `label` | confirmed / candidate / false positive (from the NASA Exoplanet Archive) |

Light curves come from NASA's archive through **Lightkurve**. Periods, epochs, and labels come from the
**NASA Exoplanet Archive** (KOI cumulative table). Missing periods, epochs, or labels are **never invented**; those rows are skipped and logged.

---

## 3. How it works (backend)

```text
light curve -> clean/normalize -> fold on period -> make views
            -> label -> split by star -> train CNN -> evaluate candidates
```

1. **Load:** read the metadata table and download or load each star's light curve.
2. **Clean:** remove quality-flagged points, NaNs, and outliers.
3. **Normalize / detrend:** remove slow brightness drift so the network sees dip *shape*, not stellar brightness.
4. **Fold:** stack all orbits on top of each other using `period` and `epoch`.
5. **Make two views** (fixed length):
   - **Global view:** the whole orbit (shows secondary eclipses and other binary signs).
   - **Local view:** zoomed on the dip (shows its exact shape).
6. **Split by star:** train / validation / test. All views from one star stay in one split, so there is no leakage.
7. **Train:** a compact two-branch 1D CNN (one branch per view), merged into dense layers, ending in a sigmoid score.
   Class weights handle the imbalance between planets and non-planets.
8. **Evaluate:** on held-out stars only. The threshold is chosen on validation data, and the test set is run once.
9. **Report:** print a terminal summary and save ranked results and plots.

---

## 4. How to run

> Replace these commands with the real ones once the code exists.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.load --config configs/preprocess_v1.json      # download + build metadata
python -m src.preprocess --config configs/preprocess_v1.json # clean, fold, make views
python -m src.train --seed 42                                # split by star + train
python -m src.evaluate --run run_YYYY-MM-DD_seed42           # metrics, ranking, plots
```

---

## 5. Output

### In the terminal

A printed summary after each run, for example:

```text
Run: seed=42, preprocess_v1
Split (by star):   train  val  test
  positives         ...   ...  ...
  negatives         ...   ...  ...
Threshold (chosen on validation): ...

Test results:
  Precision ...  Recall ...  PR-AUC ...  ROC-AUC ...
  Confusion matrix: TN=...  FP=...  FN=...  TP=...

Top ranked candidates:
  1. KIC ...   score ...   NASA label: ...
```

### Is each star processed once, or can it be reused?

- **Every star is processed independently.** The run produces output for **all** stars in the dataset, one row each. It does not stop after one star.
- **Processed stars are cached.** After a star's views are built, they are saved as arrays in `data/processed/`. Re-running skips the heavy
  download and folding steps for stars already processed, unless the preprocessing config or version changes.
- **Each run is saved separately** under `outputs/`, so you can compare runs without overwriting earlier ones.

---

## 6. How data is stored

```text
EXO110F/
  data/                      # gitignored: never commit raw or large data
    raw/
      koi_table.csv          # NASA Exoplanet Archive table (periods, epochs, labels)
    processed/
      views_<source_id>.npz  # global + local view arrays for one star
      metadata.csv           # one row per example (see Section 2) + split + preprocess_version
  configs/
    preprocess_v1.json       # cleaning/folding/binning settings
  outputs/
    run_YYYY-MM-DD_seed42/
      ranked_candidates.csv
      metrics.json
      config_used.json
      model.keras
      plots/
```

- **Kepler / NASA light curves:** downloaded by Lightkurve, which keeps its own local cache (check Lightkurve's documentation for the cache location on your system).
  Our code reads from that cache and writes only the *processed* arrays into `data/processed/`.
- **NASA Exoplanet Archive table:** downloaded once as CSV into `data/raw/`. Record the download date in the README, because the archive is updated over time.
- **Metadata table:** the single source of truth for each example (ID, target, period, epoch, cadence, label, split, preprocessing version).

---

## 7. How ranked results are stored

After a successful run, `outputs/run_<date>_seed<N>/` contains:

| File | Contents |
|---|---|
| `ranked_candidates.csv` | `source_id`, `score`, `NASA label`, `split`, sorted by score (highest first) |
| `metrics.json` | precision, recall, PR-AUC, ROC-AUC, threshold, class counts per split |
| `config_used.json` | preprocessing settings, seed, code version |
| `model.keras` | trained model, saved next to the config used to train it |
| `plots/top_candidates.png` | global + local views of the highest-ranked signals |
| `plots/false_positives.png` | confident wrong guesses |
| `plots/false_negatives.png` | planets the model missed |

A high score means "looks planet-like, worth a follow-up". It is **not** the probability that the object is a confirmed exoplanet.

---

## 8. Reproducibility checklist

Fill these in for every reported result:

- Flux product and cadence: `...`
- Quality flags, detrending, clipping: `...`
- How periods and transit centers were obtained: `...`
- Split made by star **before** creating views: yes / no
- Threshold and the validation metric used to choose it: `...`
- Data download date, code commit, random seed, hardware: `...`

---

## 9. What is used

**Language and libraries**
- Python
- [Lightkurve](https://docs.lightkurve.org/): download and work with Kepler/TESS light curves
- NumPy, pandas: arrays and tables
- scikit-learn: group-aware splitting and metrics
- TensorFlow / Keras: the CNN
- Matplotlib: saved plots
- Jupyter: first exploration of a known planet

**Data sources**
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/): periods, epochs, and dispositions (confirmed / candidate / false positive)
- [Kepler / MAST archive](https://archive.stsci.edu/kepler/): Kepler light curves (accessed through Lightkurve)
- *(Optional)* [Kaggle Kepler Labelled Time Series](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series):
  5,087 stars, 37 confirmed planets; useful for learning about class imbalance

**Reference material**
- Shallue & Vanderburg, *Identifying Exoplanets with Deep Learning* (arXiv:1712.05044)
- [Google Research exoplanet-ml](https://github.com/google-research/exoplanet-ml) (original AstroNet; not redistributed here)
- [TensorFlow exoplanet CNN example](https://blog.tensorflow.org/2018/11/deep-learning-for-identifying-exoplanets.html)

---

## 10. Data licensing and credit

NASA and MAST data are publicly available, but **check each source's own data policy and acknowledgement requirements before publishing or redistributing**.
Do the same for the Kaggle dataset (see its license on the dataset page) and the exoplanet-ml repository. Do not commit large datasets or third-party data packages to this repository.

When you publish results, cite the archive and the paper, and credit Lightkurve according to its documentation.

---

## 11. Limitations

- Only 37 confirmed planets in the Kaggle set means a single example can swing a metric. Repeat splits or use cross-validation.
- Labels from the archive are not a complete catalogue of astrophysical truth.
- Kepler and TESS data are not interchangeable without care (cadence, noise, systematics).
- This model ranks signals. It does not detect dips and does not confirm planets.

---

## 12. What I learned

> What I've worked through so far:

- **Detecting vs classifying:** finding a repeating dip is a different job from judging whether that dip looks like a planet.
- **Candidate vs confirmed:** a high model score is a ranking, not a confirmation. Confirmation needs follow-up observations and expert review.
- **Why fold a light curve:** stacking orbits makes a faint, repeating dip clear enough to see and learn from.
- **Why split by star:** if views of the same star appear in both training and testing, the score looks better than it really is.
- **Why not accuracy alone:** with so few planets, a model can be highly "accurate" while missing nearly all of them.
- **Errors teach the most:** plotting confident false positives and false negatives reveals preprocessing bugs and label problems.
- **What I learned in Electronics & Communication:**

| Field | Meaning |
|---|---|
| Removing bad points and outliers | Noise removal, outlier clipping |
| Flattening slow brightness drift | Detrending, which works like high-pass filtering |
| Normalizing the flux | Signal normalization |
| Folding on the period | Averaging repeated cycles to lift a weak signal above noise |
| Binning into global and local views | Resampling and binning to a fixed length |
| `Conv1D` layers in the CNN | Sliding-window filtering (convolution) with learned coefficients |
| Faint dip versus noise | SNR, which is the core challenge |
