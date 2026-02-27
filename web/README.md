# ECG Heartbeat Classification

A small neural network that sorts single ECG heartbeats into five beat types, with an
honest evaluation and a demo that runs entirely in the browser.

**Live demo:** pick a sample beat or paste your own signal; the model runs in your
browser via ONNX Runtime Web. Nothing is uploaded anywhere. (Deployed from `site/`.)

## The problem

Given one heartbeat (187 ECG samples), classify it as normal (N), supraventricular
ectopic (S), ventricular ectopic (V), fusion (F), or unknown (Q). This is the standard
MIT-BIH heartbeat classification task.

## Data

The MIT-BIH Arrhythmia Database (Moody and Mark 2001, via PhysioNet): 48 half-hour
ECG recordings from 47 subjects, annotated beat by beat by cardiologists. I use the
heartbeat-level version: each beat is a 187-sample window at 125 Hz starting at the R
peak, min-max normalized to [0, 1]. The source project ships `mitbih_test.csv`
(21,892 beats); for the grouped evaluation I reconstructed all 109,406 beats from the
48 records myself with the same recipe (see `reconstruct.py`).

## Model

Feed-forward network: 187 inputs, dense layers of 256, 128 and 64 ReLU units, 5-way
softmax. The demo uses the source repo's trained weights (`ecg_ann_best.h5`, 1.1 MB),
converted to ONNX (360 KB). I chose this model over the 80 MB CNN variants because it
fits comfortably in a browser.

## Results

### Fixed model on the shipped test split (n = 21,892)

This reproduces the original project's evaluation. Note it is optimistic: the original
project trained on 70% of the concatenated train+test data, so the model has seen most
of these beats.

| Metric | Value | 95% CI |
|---|---|---|
| Accuracy | 0.988 | [0.987, 0.990] |

Per-class (precision / recall / F1, with 95% bootstrap CIs on F1):

| Class | n | Precision | Recall | F1 | F1 95% CI |
|---|---|---|---|---|---|
| Normal (N) | 18,118 | 0.993 | 0.996 | 0.994 | [0.993, 0.995] |
| Supraventricular (S) | 556 | 0.902 | 0.849 | 0.875 | [0.853, 0.895] |
| Ventricular (V) | 1,448 | 0.968 | 0.970 | 0.969 | [0.963, 0.975] |
| Fusion (F) | 162 | 0.881 | 0.864 | 0.872 | [0.831, 0.909] |
| Unknown (Q) | 1,608 | 0.997 | 0.985 | 0.991 | [0.988, 0.994] |

CIs are percentile bootstrap, 2,000 resamples, stratified by class.

### Calibration

The model's confidence scores were already reasonable (expected calibration error
0.007 on held-out beats). Temperature scaling (T = 1.74, Guo et al. 2017) roughly
halved that to 0.003 without changing any prediction. The demo shows the calibrated
confidence.

### Honest evaluation: grouped by patient (5-fold GroupKFold over the 48 records)

Beats from the same patient share electrode placement and heart geometry, so a random
split lets the model memorize patients instead of learning beat types. Grouping by
record keeps every patient wholly in train or test. I retrained the same architecture
from scratch inside each fold (no class weights; see REPORT.md for why).

| | Accuracy |
|---|---|
| Random split (leaky) | 0.988 |
| Grouped by record, mean of 5 folds | 0.869 (std 0.073) |
| Grouped by record, pooled | 0.870 |

Per-class pooled F1 under grouped evaluation: N 0.933, S 0.001, V 0.577, F 0.000,
Q 0.677. The rare classes (S, F) are effectively not learned without rebalancing;
the model is honest about what it can and cannot do for new patients.

Per-fold and per-class numbers are in `grouped_cv_results.json`; see the notebook for
the full breakdown.

The gap is expected and well documented in the literature: intra-patient accuracy near
99% typically drops to around 90% under the inter-patient protocol. The grouped number
is the one to quote for new patients.

## How the web demo works

`site/` is a static page (no build step). It loads `ecg_ann.onnx` and runs it with
ONNX Runtime Web from a CDN. Preprocessing mirrors training exactly: the input must be
187 samples, min-max normalized to [0, 1], windowed from the R peak. Temperature
scaling (T = 1.74) is applied in JavaScript so the displayed confidence is calibrated.
Five sample beats with precomputed outputs live in `site/samples/`; visitors can also
paste or upload their own 187-sample beat.

This is a research demo, not a medical device.

## Reproduce

```
pip install tensorflow-cpu tf2onnx onnx wfdb scikit-learn matplotlib
python reconstruct.py     # needs the 48 MIT-BIH records in mitdb/ (see script)
python grouped_cv.py 5 15 # 5-fold grouped CV, the honest evaluation
python calibration.py     # ECE + temperature scaling, saves figures/calibration_curve.png
python bootstrap_cis.py   # bootstrap CIs for the fixed model
python build_notebook.py  # assemble analysis.ipynb
jupyter nbconvert --to notebook --execute notebooks/analysis.ipynb \
  --output notebooks/analysis.ipynb --allow-errors
```

## Files

- `analysis.ipynb` (in `notebooks/`, executed) - the full analysis with outputs
- `figures/` - confusion matrix, calibration curve, grouped-CV plots, sample beats
- `site/` - the static web demo
- `REPORT.md` - method justifications and literature citations
- `reconstruct.py`, `grouped_cv.py`, `calibration.py`, `bootstrap_cis.py` - analysis scripts
- `ecg_ann_best.h5` lives in the source repo, not duplicated here

## Sources

MIT-BIH Arrhythmia Database (Moody & Mark 2001; Goldberger et al. 2000, PhysioNet,
DOI 10.13026/C2F305); inter-patient protocol and benchmarks (de Chazal et al. 2004);
temperature scaling and ECE (Guo et al. 2017); percentile bootstrap (Efron 1979); AAMI
EC57 beat classes. Full citations in REPORT.md.
