# Research guide: a small AstroNet-style exoplanet classifier

This project is a deliberately smaller, Keras-style implementation of the
idea behind AstroNet. It should be treated as a learning and reproduction
project, not as a pipeline for confirming new planets. A transit-like dip is
only evidence for a candidate; confirmation requires additional observations
and domain expertise.

## 1. The central paper

**Primary reference:** [Shallue & Vanderburg, *Identifying Exoplanets with
Deep Learning* (arXiv:1712.05044)](https://arxiv.org/abs/1712.05044).

The paper describes a convolutional neural network that classifies Kepler
signals as planet candidates or false positives. Its best-known demonstration
was recovering the previously missed **eighth planet in the Kepler-90
system, Kepler-90i**. The model was not looking at an image: it learned
features from one-dimensional brightness measurements after the light curve
had been folded on a candidate period and centered on transit.

Important ideas to carry into this project:

* use a global view of the phase-folded light curve and local views of the
  transit;
* normalize consistently so the network learns transit shape rather than
  stellar brightness;
* compare against astrophysically plausible false positives, not only random
  noise;
* keep training, validation, and test stars separate; otherwise repeated
  views of one star can make the score misleading.

The paper's data preparation, labels, and architecture are more sophisticated
than this project needs. We should reproduce the learning pattern, not claim
to reproduce the paper's result.

## 2. AstroNet and the build-system distinction

The reference implementation is the
[Google Research exoplanet-ml repository](https://github.com/google-research/exoplanet-ml).
It includes a pre-computed training-data package of roughly 106 MB, which is
useful for understanding the expected tensors and labels without rebuilding
the entire Kepler preprocessing pipeline. Check the repository's own license
and data instructions before redistributing that package; do not commit it to
this repository.

AstroNet is built with **Bazel** (not “bezel”) and uses TensorFlow-oriented
components. That build system is a reference for the original project, not a
requirement here. Our implementation can use ordinary Python/Keras data
loading, model construction, and training while preserving the scientific
steps:

```text
light curve -> clean/normalize -> fold on period -> make views
            -> label -> split by star -> train CNN -> evaluate candidates
```

## 3. Start with Lightkurve

The [Lightkurve tutorials](https://docs.lightkurve.org/tutorials/index.html)
begin with searching for and working with Kepler and TESS data. The first
practical milestone should be the **Recover a known planet** tutorial (found
under the tutorials' “Searching for a target” section), or the equivalent
tutorial in the version installed for this project.

Before training a CNN, work through these steps on one known target:

1. search the archive and download a light curve;
2. remove quality-flagged points and inspect gaps;
3. normalize the flux and, where appropriate, remove long-term trends;
4. identify or use a published period and epoch;
5. fold the light curve and inspect the transit by eye;
6. convert the result into the same fixed-size arrays used by the model.

This manual check catches unit, cadence, time-system, and normalization
mistakes earlier than model metrics do. TESS and Kepler data are not
interchangeable by default: cadence, noise, sectors/quarters, and systematics
must be recorded with each example.

## 4. A small, imbalanced training set

The Kaggle [Kepler Labelled Time Series
dataset](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series)
contains 5,087 stars and only 37 confirmed planets. It is therefore a useful
small dataset for learning about class imbalance, but it is much simpler than
the full AstroNet data and its labels should not be treated as a complete
catalogue of astrophysical truth.

Recommended first experiments:

* start with a stratified split, while ensuring all views from one star stay
  in one split;
* report the number of positive and negative examples in every split;
* use class weights or carefully chosen sampling rather than duplicating
  validation data;
* report precision, recall, PR-AUC, ROC-AUC, and a confusion matrix—not
  accuracy alone;
* keep a final test set untouched until the model and threshold are chosen.

With only 37 positives, one example can change a metric substantially. Include
the random seed and, when practical, repeat the split or use cross-validation.

## 5. False positives and the TensorFlow example

The [TensorFlow exoplanet CNN example](https://blog.tensorflow.org/2018/11/deep-learning-for-identifying-exoplanets.html)
is a useful conceptual companion. It frames the task as
separating planet candidates from eclipsing binaries and “junk” signals, which
is close to the false-positive problem this project is trying to make
understandable.

The model must learn more than “a dip means a planet.” Useful counterexamples
include:

* an eclipsing binary with a secondary eclipse;
* a blended or diluted binary;
* stellar variability and instrumental artifacts;
* a single isolated outlier or a discontinuity near a transit;
* a period alias that produces an apparently convincing folded signal.

Inspect false positives and false negatives as part of every experiment.
Plotting the highest-confidence mistakes often reveals preprocessing bugs or
label leakage that a single score hides.

## 6. Scope for the simpler Keras version

The smallest useful implementation should make each scientific assumption
visible:

1. load a documented CSV/array format;
2. clean and normalize each light curve;
3. fold using a supplied period and epoch;
4. resample global and local views to fixed lengths;
5. train a compact one-dimensional CNN in Keras;
6. evaluate on held-out stars and save the preprocessing configuration;
7. plot predictions and representative errors.

Do not silently invent periods, labels, or missing values. Store the source
identifier, target, period, epoch, cadence, split, and preprocessing version
alongside each example. A prediction from this project is a ranked candidate,
not a confirmed detection.

## 7. Further reading and reproducibility checklist

The paper and tutorials leave several implementation decisions that should be
written down before comparing runs:

* Which flux product and cadence were used?
* Which quality flags, detrending method, and clipping rules were applied?
* How were periods and transit centers obtained?
* Were examples split by star before or after creating phase-folded views?
* What threshold was selected, and using which validation metric?
* Which data, code version, random seed, and hardware produced the result?

For background on the mission data, use the
[NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) and the
[Kepler mission archive documentation](https://archive.stsci.edu/kepler/).
These sources help distinguish a confirmed planet, a candidate, and a false
positive—categories that must not be collapsed into one binary label without
documentation.
