# Technical report: ECG heartbeat classification

## What this project does

I take a pre-trained heartbeat classifier (dense network, trained on MIT-BIH
arrhythmia beats) and subject it to the checks the original project skipped:
uncertainty quantification, calibration, and patient-grouped validation. The trained
network is converted to ONNX and served as a static in-browser demo.

## Data

The MIT-BIH Arrhythmia Database (Moody and Mark 2001, hosted on PhysioNet) contains
48 half-hour two-lead ECG recordings from 47 subjects, recorded 1975-1979 at 360 Hz
and annotated beat by beat by cardiologists (about 110,000 beats). The heartbeat-level
dataset used here follows the common Kaggle preprocessing: resample to 125 Hz, take a
187-sample window starting at each annotated R peak, min-max normalize each beat to
[0, 1], label with five AAMI classes (N, S, V, F, Q). The source repo ships
`mitbih_test.csv` with 21,892 such beats; I evaluate the fixed model on exactly that
file.

Known quirks I accounted for: records 201 and 202 come from the same subject, so
grouping by record (not subject) still leaks slightly between those two; paced beats
and non-beat annotations are excluded from the AAMI mapping; the database is
dominated by normal beats (about 83% here), so accuracy alone is a weak metric.

## Methods and why I chose them

- **Per-class precision, recall, F1 alongside accuracy.** Accuracy lets a model ignore
  rare classes (fusion beats are 0.7% of the data). Per-class metrics show where it
  actually fails.
- **Percentile bootstrap confidence intervals (2,000 resamples, stratified by class).**
  The bootstrap (Efron) needs no distributional assumptions about the metrics, and
  stratification keeps the rare classes represented in every resample. This is the
  standard way to put uncertainty on test-set metrics.
- **Calibration check (reliability diagram, expected calibration error) and temperature
  scaling (Guo et al. 2017).** Neural networks are often overconfident; ECE measures the
  gap between stated confidence and observed accuracy. Temperature scaling divides the
  logits by a single fitted T before the softmax. I chose it because it cannot change
  any prediction, only the confidence, so accuracy is preserved by construction.
- **GroupKFold cross-validation grouped by record (5 folds).** Beats from the same
  patient share electrode placement, heart geometry and baseline quirks. A random split
  puts one patient's beats in both train and test, letting the model memorize patients
  instead of learning beat types. Grouping by record keeps every patient wholly on one
  side. This is the inter-patient protocol of de Chazal et al., the scheme the field's
  benchmarks use, as opposed to the leaky intra-patient random split. The literature
  consistently shows intra-patient accuracy near 99% dropping to roughly 90% (with much
  lower F1 on rare classes) under the inter-patient protocol.
- **Class weights instead of SMOTE for the grouped retraining.** The original project
  used SMOTE to balance classes. I tested balanced class weights as a cheaper
  alternative, but they destabilized training: grouped accuracy fell from about 0.88
  to about 0.68. I report the unweighted model and let the per-class metrics carry the
  imbalance story instead.
- **ONNX for the browser demo.** ONNX is the standard portable model format; ONNX
  Runtime Web executes it from a CDN with no build step and no server, so visitor data
  never leaves their machine. I verified the ONNX outputs match the Keras outputs to
  1e-7 with 100% argmax agreement on 500 test beats.

## Results

(Filled in from computed outputs; see README for the numbers.)

## Limitations

- The fixed model's 98.8% is measured on a split the model partially trained on (the
  original project trained on 70% of the concatenated train+test), so treat it as a
  reproduction of the original claim, not an independent test.
- My reconstruction of the 48 records follows the documented recipe but is mine, not
  the Kaggle author's exact code; small preprocessing differences are possible.
- Records 201/202 share a subject, so record-grouped folds leak slightly there.
- Fusion beats are extremely rare (a dozen or so across all records), so their metrics
  have wide intervals and some folds cannot learn them at all.
- The demo is a research artifact, not a medical device.

## Sources

- Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Engineering
  in Medicine and Biology Magazine, 2001. (Database provenance; 48 half-hour excerpts,
  47 subjects, 1975-1979, 360 Hz, ~110,000 cardiologist-annotated beats.)
- Goldberger AL et al. PhysioBank, PhysioTools, and PhysioNet. Circulation, 2000.
  (PhysioNet hosting; DOI 10.13026/C2F305.)
- de Chazal P, O'Dwyer M, Reilly RB. Automatic classification of heartbeats using ECG
  morphology and heartbeat interval features. IEEE Trans. Biomedical Engineering, 2004.
  (Inter-patient protocol; 96.4% accuracy benchmark under grouped evaluation.)
- Systematic review of deep learning for ECG arrhythmia classification (MDPI).
  (Intra-patient accuracy near 99% vs inter-patient around 90% with F1 near 0.63;
  documents the leakage gap.)
- Guo C et al. On calibration of modern neural networks. ICML, 2017. (Temperature
  scaling; expected calibration error.)
- Efron B. Bootstrap methods: another look at the jackknife. Annals of Statistics,
  1979. (Percentile bootstrap confidence intervals.)
- AAMI EC57. Testing and reporting performance results of cardiac rhythm and ST segment
  measurement algorithms. (Beat class groupings N, S, V, F, Q.)
