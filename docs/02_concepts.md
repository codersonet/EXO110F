# 02. Concepts used in this project

Every idea used in the project, explained in plain language and grouped by topic.
For each one: what it is, an everyday picture of it, and where it shows up.

---

## A. Astronomy ideas

| Concept | In simple words | Where it appears |
|---|---|---|
| **Light curve** | A graph of a star's brightness over time. Each row of our data is one star's light curve. | all data files |
| **Transit** | A tiny, brief dimming when a planet passes in front of its star, like a moth flying past a lamp. | what the model looks for |
| **Exoplanet** | A planet orbiting a star other than our Sun. | the whole project |
| **Planet candidate** | A signal that looks like a planet but is not yet proven. Our model only ever produces candidates. | wording in all docs |
| **False positive** | A signal that looks like a planet but is something else, such as two stars eclipsing each other, a star's own flickering, or an instrument glitch. | `errors.py` |
| **Eclipsing binary** | Two stars orbiting each other and blocking each other's light. A common planet impostor. | `errors.py` |
| **Kepler** | The NASA space telescope that watched thousands of stars for dips. | data source |
| **Period and epoch** | How often a dip repeats, and when the first one happened. Needed to "fold" a light curve. Our Kaggle data does not include them. | limitations |
| **Phase folding** | Stacking all repeats of a dip on top of each other so the dip becomes clear. AstroNet does this; our simpler project cannot. | research guide |

## B. Data ideas

| Concept | In simple words | Where it appears |
|---|---|---|
| **CSV file** | A table saved as text, one row per star. | `exoTrain.csv`, `exoTest.csv` |
| **Label** | The known answer for each star: planet or not. Kaggle uses 1 and 2; we convert to 0 and 1. | `data.py` |
| **NaN** | "Not a number": a missing value. We stop with an error instead of guessing a replacement. | `preprocess.py` |
| **Array / tensor shape** | A grid of numbers. `(5000, 3000)` means 5000 stars with 3000 measurements each. The CNN needs an extra `1` at the end: `(stars, time, 1)`. | all code |
| **Class imbalance** | Far more non-planets than planets. Like finding 37 needles in 5000 haystacks. | the central difficulty |
| **Normalization** | Putting every star on the same scale so brightness differences between stars do not matter. | `preprocess.py` |
| **Detrending** | Removing slow drifts, like subtracting a slowly sliding floor so that short bumps stand out. | `preprocess.py` |
| **Outlier** | A single freak point, such as a cosmic-ray hit. | clipping |
| **Train / validation / test** | Three separate piles of stars. Train: the model learns from them. Validation: used to tune and compare. Test: a final exam used once. | `preprocess.py` |
| **Stratified split** | Splitting so each pile has the same planet fraction, so planets are not accidentally all in one pile. | `preprocess.py` |
| **Data leakage** | Accidentally letting the model peek at exam answers, for example copies of the same star in both training and validation. It makes scores falsely good. | design rule |
| **Data augmentation** | Making extra training examples by slightly changing existing planets (shifting, flipping, adding a little noise). | `cnn.py` |

## C. Machine learning ideas

| Concept | In simple words | Where it appears |
|---|---|---|
| **Feature** | A number that summarizes something about a star, such as its lowest point. | `baseline.py` |
| **FFT (Fourier transform)** | Splits a signal into how much of each repeating rhythm it contains, like hearing the individual notes inside a chord. | `baseline.py` |
| **Logistic regression** | A simple model that combines features into a probability between 0 and 1. | `baseline.py` |
| **Baseline** | The simple model you must beat. If your fancy model cannot, the fanciness is not helping. | `baseline.py` |
| **Neural network** | A stack of layers that each learn to detect patterns, from simple to complex. | `cnn.py` |
| **1D CNN (convolutional network)** | A network that slides small pattern detectors along a sequence, for example "is there a dip shaped like this here?" | `cnn.py` |
| **Conv1D layer** | The sliding pattern detector. | `cnn.py` |
| **Pooling** | Shrinking the sequence by keeping the strongest responses. Global max pooling keeps the single strongest response anywhere, which answers "is there a dip somewhere?" | `cnn.py` |
| **ReLU** | A simple on/off style rule that lets the network learn curved, non-linear patterns. | `cnn.py` |
| **Sigmoid** | Squashes the final number into 0 to 1, read as "probability of planet". | `cnn.py` |
| **Dropout** | Randomly switching off parts of the network during training so it cannot just memorize. | `cnn.py` |
| **Overfitting** | Memorizing the training examples instead of learning the real pattern. A big risk with few planets. | design rule |
| **Loss function** | The score of how wrong the model is. Training tries to lower it. We use binary cross-entropy. | `cnn.py` |
| **Optimizer (Adam)** | The method that adjusts the network to lower the loss. | `cnn.py` |
| **Epoch / batch** | One pass over all training data / the small group of stars processed at once. | `cnn.py` |
| **Class weights** | Making a missed planet cost much more than a false alarm, to counter imbalance. | `baseline.py`, `cnn.py` |
| **Early stopping** | Stop training when the validation score stops improving and keep the best version. | `cnn.py` |
| **Random seed** | A fixed starting number so random steps repeat exactly. | all code |
| **Cross-validation** | Repeating the experiment on several different splits and averaging, because one small split is unreliable. | `cv.py` |
| **Threshold** | The probability above which we call a star a candidate. 0.5 is the placeholder in `baseline.py` and `cnn.py`; the real one is chosen in `cv.py`. | `evaluate.py`, `cv.py` |

## D. Evaluation ideas

| Concept | In simple words |
|---|---|
| **True positive (TP)** | A real planet that we flagged. |
| **False positive (FP)** | A non-planet we flagged: a false alarm. |
| **False negative (FN)** | A real planet we missed. |
| **True negative (TN)** | A non-planet we correctly ignored. |
| **Confusion matrix** | The 2x2 table counting TN, FP, FN, TP. |
| **Precision** | Of everything we flagged, how much was real? (Trust in the alarms.) |
| **Recall** | Of all real planets, how many did we find? (Coverage.) |
| **ROC-AUC** | How well the model ranks planets above non-planets. 0.5 is a coin flip, 1.0 is perfect. Can look too rosy on very imbalanced data. |
| **PR-AUC** | Summarizes the balance between precision and recall across all thresholds. Our headline metric. A random model scores about the planet fraction (under 1%). |
| **Accuracy** | Share of correct answers. Misleading here, so not used. |

## E. Engineering ideas

| Concept | In simple words |
|---|---|
| **Virtual environment (`.venv`)** | A private box of packages for this project, so it does not clash with other projects. |
| **`requirements.txt`** | The shopping list of packages, so others can recreate the box. |
| **Python package (`__init__.py`)** | A folder Python recognizes as importable code. |
| **`PYTHONPATH=src`** | Tells Python where the project's code folder is. |
| **Git, commit, push** | Saving snapshots of the project and uploading them to GitHub. |
| **`.gitignore`** | Files git must not upload (huge data, the `.venv`, model files). |
| **Config file (JSON)** | Settings saved to disk so a result can be reproduced. |
| **`.npz` file** | A compressed bundle of several arrays in one file. |
| **Reproducibility** | Anyone, including future you, can rerun the project and get the same result. |

## F. Extra ideas from the final stages

| Concept | In simple words | Where it appears |
|---|---|---|
| **K-fold cross-validation** | Cut the labeled stars into K equal groups. Train on K-1 groups, test on the one left out, and rotate so every star is tested exactly once. | `cv.py` |
| **Out-of-fold prediction** | A star's score from a model that never trained on that star. These scores are honest. | `cv.py`, `errors.py` |
| **Mean and standard deviation across folds** | The average score, and how much it wobbles between folds. A big wobble means a shaky result. | `cv.py` |
| **F1 score** | One number that balances precision and recall. Used here to pick a threshold. | `cv.py` |
| **Precision-recall curve** | Shows precision and recall at every possible threshold. | `cv.py` |
| **Fixed number of epochs** | In cross-validation we train for a set number of rounds instead of using early stopping on the held-out fold, which would leak information. | `cv.py` |
| **Test set lock** | The final script refuses to run twice, so test results cannot quietly influence later choices. | `final_test.py` |
| **Unit test** | A tiny automatic check that something we rely on is true, such as "no star is in two splits". | `tests/test_data.py` |
| **Ranked list / ranking score** | Stars sorted from most to least planet-like by the model's score. The score ranks stars; it is not a probability. | `final_test.py` |
| **Top-k hit count** | How many known planets sit inside the k highest-ranked stars. A good ranking puts them near the top. | `final_test.py` |
| **Hyperparameter** | A setting we choose ourselves, such as the median window (101), the number of epochs (30) or dropout (0.5). | `preprocess.py`, `models.py`, `cv.py` |
| **Seed variance** | Different random seeds can give different scores, so one run is not proof. | `cv.py` |
| **Lightkurve** | A Python tool for downloading, cleaning and folding NASA light curves. Not used in this project (see README section 9). | README |
| **Campaign (K2)** | An observing period of the K2 mission. The dataset is described as derived mainly from Campaign 3 observations. | README |
