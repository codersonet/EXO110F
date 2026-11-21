# 05. Why we chose what we chose

Doc 4 says *what* each function does. This guide says *why* it was picked over the alternatives and
*how it helps* the project. Each entry is: choice, reason, what would go wrong otherwise.

---

## 1. Project structure and tools

| Choice | Why | Without it |
|---|---|---|
| Plain `.py` scripts for the pipeline, not notebooks | Scripts rerun the same way every time, are easy to test and read well on GitHub. Notebooks are kept for optional exploring. | Hidden state and out-of-order cells make results hard to reproduce. |
| One folder per kind of work (`data`, `src`, `configs`, `results`, `docs`, `tests`) | Anyone can find things quickly, and generated files stay separate from code. | A messy repo where data, code and plots are mixed. |
| Virtual environment `.venv` | Pins this project's packages in one private place. | Version clashes with other projects. |
| Python 3.12 | TensorFlow supports it, and python.org still ships a Mac installer for 3.12.10. Python 3.15 is newer but TensorFlow support for it could not be confirmed. | Failed TensorFlow installs. |
| `requirements.txt` | Others can rebuild the environment with one command. | "Works on my machine" problems. |
| `.gitignore` for data and models | GitHub rejects files over 100 MB, and large files bloat the repo. Data has its own download instructions. | Failed pushes and licensing risk. |
| `Path(__file__).resolve().parents[2]` for paths | Paths no longer depend on which folder you ran the command from. | The `FileNotFoundError` we hit earlier. |
| `PYTHONPATH=src` and a package with `__init__.py` | Lets files import each other cleanly (`from exoplanet.data import ...`). | Messy or broken imports. |
| Colab was considered, local VS Code chosen | The dataset and model are small, so a laptop CPU is enough. TPUs are overkill here. | Extra setup with no benefit. |

## 2. Data and labels

| Choice | Why | Without it |
|---|---|---|
| Convert labels to 0 and 1 | Standard form for binary classifiers and metrics. | Metric functions would misread label 1 vs 2. |
| `float32` | Half the memory of float64 and what TensorFlow expects. | Slower runs, more memory. |
| Raise an error on NaNs | The research guide says never silently invent values. | A hidden fix might distort a real signal. |
| Keep the Kaggle test file locked | A test set only measures anything if the model never influenced choices through it. | Over-optimistic final scores. |

## 3. Preprocessing

| Choice | Why | Without it |
|---|---|---|
| Median filter for detrending | A median ignores short dips, so it follows the slow drift but not the transit. A plain average would be pulled down by a dip and partly erase it. | Drift dominates the signal, or the dips get removed along with it. |
| Window of 101 steps | Wide enough that a short transit is not mistaken for trend. | A smaller window flattens the dips. A much larger window leaves drift in. |
| Subtract the median, divide by robust noise (MAD x 1.4826) | The usual standard deviation is inflated by outliers and dips; MAD is not. Every star then has the same noise scale. | The model would learn brightness and noise level instead of dip shape. |
| Clip only the upper side (4 sigma) | Flares and cosmic rays spike upward, while transits go downward. We protect the signal we want. | Symmetric clipping would cut the dips themselves. |
| Process each star independently | No statistic is shared across stars, so nothing leaks between train, validation and test. | Subtle leakage inflates the scores. |
| `StratifiedShuffleSplit` with a fixed seed | Keeps the planet fraction equal in each pile and makes the split repeatable. | A random split could leave the validation set with zero planets. |
| Save to `.npz` and keep split indices | Fast to reload, and any prediction can be traced back to its CSV row. | Reprocessing on every run, and no way to trace errors. |
| Config saved as JSON with a version name | Results can be tied to the exact settings used. | Impossible to compare runs fairly later. |

## 4. Baseline model

| Choice | Why | Without it |
|---|---|---|
| Build a baseline first | It tells us whether a CNN is actually worth its complexity. If the CNN cannot beat it, that is a real finding. | No way to judge the CNN's number. |
| Summary stats and percentiles | Cheap descriptions of how deep and how skewed a star's brightness goes. A planet adds a small low tail. | The model would have nothing to work with. |
| FFT of the lowest 100 frequencies | Periodic dimming appears as energy at certain frequencies. Normalizing it keeps the shape and discards the scale. | Misses repeating patterns. |
| `StandardScaler` in a pipeline | Features have very different scales, which hurts logistic regression. The pipeline guarantees the scaler is fit only on training data. | Slow or poor training, and possible leakage. |
| Logistic regression | Simple, fast, hard to overfit, easy to explain. | A complex baseline would defeat its purpose. |
| `class_weight="balanced"` | Makes each rare planet count more in the loss. | The model would learn to say "no planet" for everything. |

## 5. Evaluation

| Choice | Why | Without it |
|---|---|---|
| PR-AUC as the main metric | It focuses on the rare positive class and punishes false alarms. | ROC-AUC alone can look excellent while the model is poor. |
| Report precision, recall and the confusion matrix | They show the trade-off between false alarms and misses, which accuracy hides. | A model that finds nothing could look perfect. |
| Do not report accuracy | Over 99% is reached by predicting "no planet" every time. | Misleading headline numbers. |
| One shared `report()` function | Baseline and CNN are scored identically. | Unfair comparisons from slightly different code. |
| `zero_division=0` | Avoids a crash, or a confusing result, when a model flags no stars. | An error exactly when the model is failing. |
| Placeholder threshold 0.5, chosen properly later | The right threshold depends on how many false alarms we tolerate, and should be picked on validation data, never on the test file. | A bad threshold picked by accident. |
| Cross-validation | With a handful of validation planets one star can move the scores a lot. Averaging over splits shows the real size of the uncertainty. | Mistaking luck for a result. |

## 6. CNN design

| Choice | Why | Without it |
|---|---|---|
| 1D convolutions | Our data is a 1D signal; convolutions find the same dip shape wherever it occurs. AstroNet uses the same idea. | A 2D CNN would need images, losing information and adding complexity. |
| Small network (three conv layers) | With only a few dozen planets, a bigger network would memorize them. | Overfitting. |
| Kernel sizes 11, 7, 5 | Start by looking at wider windows, then finer features. | Less variety of patterns. |
| Max pooling | Reduces length and keeps the strongest response, which makes the model less sensitive to exact dip timing. | A long, slow model. |
| Global max pooling | Asks "is there a dip anywhere?", which is exactly the question. | Dependence on where in the curve the dip sits. |
| ReLU | Lets the network model curved, non-linear patterns. | A purely linear model. |
| Dropout 0.5 | Forces the network not to rely on any single unit, reducing memorizing. | Faster overfitting. |
| Sigmoid output | A probability from 0 to 1, which fits binary cross-entropy and the thresholding step. | Scores that cannot be read as probabilities. |
| Adam, learning rate 1e-3 | A common default that works without much tuning. | A lot of manual tuning for little gain. |
| Class weights | The same imbalance fix as in the baseline, so the comparison is fair. | The network learns to ignore planets. |
| Early stopping on validation PR-AUC | Stops when generalization stops improving, and restores the best weights. | Training too long and overfitting. |
| Augmentation only on training planets, after the split | Adds variety (shifts, flips, small noise) without letting copies of one star reach validation. | Leakage: copies of validation stars in training. |
| Fixed random seeds | Same code gives the same result. | Differences you cannot tell from real changes. |
| Saving the model (git-ignored) | Keeps the trained network without bloating the repo. | Retraining every time. |

## 7. Cross-validation, threshold, error analysis, final test

| Choice | Why | Without it |
|---|---|---|
| Combine train and validation for cross-validation | Every one of the few dozen planets gets tested exactly once. | A single small validation set decides everything. |
| 5 stratified folds | Each fold keeps the planet share, giving roughly 7 planets per fold. | Folds with zero planets. |
| Fixed number of epochs inside the folds | Early stopping on the held-out fold would let that fold influence training and inflate the score. | Optimistic cross-validation numbers. |
| Fresh model and fresh class weights per fold | Nothing learned in one fold leaks into another. | Leakage between folds. |
| Report mean and standard deviation | Shows how shaky the result is. | A single lucky number looks like a result. |
| Pick the threshold on pooled out-of-fold scores with F1 | Uses honest scores, never test data, and balances false alarms against misses. F1 is one reasonable rule; a recall target is another. | A threshold tuned on the test set. |
| Error analysis on out-of-fold predictions | Uses honest mistakes without touching the test file. | Either dishonest plots or a contaminated test set. |
| Sort mistakes by confidence | The most confident errors are the most informative about bugs and blind spots. | Looking at random, uninformative mistakes. |
| Final script refuses to run twice | Keeps the test set a true one-time exam. | Quietly tuning on the test file. |
| Train the final model on all labeled stars | Uses all available planets for the model we report. | Wasting about 20% of rare data. |
| Unit tests | Cheap protection against silent mistakes such as overlapping splits or a broken cleaning step. | Bugs found only after strange results. |
| README with placeholders and a limitations section | Others can see exactly what was and was not shown. | Over-claiming. |

Note: the threshold comes from models trained on 80% of the data, and the final model is trained on 100%, so its
scores may be calibrated slightly differently. This is a known small mismatch to mention in the write-up.

## 8. Honest limits

- **Few planets** (about 37 in training) means big uncertainty. We report it instead of hiding it.
- **No period or epoch** in the data means no folded global and local views as in AstroNet.
- **Labels** are not complete astrophysical truth. A high score means "looks like the labeled planets", not
  "is a planet".
- **A candidate is not a confirmation.** That wording is deliberate throughout the project.
