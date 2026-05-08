# Dashboard data: ECG heartbeat classification

Small JSON exports of the key results from this repo's notebooks, for building
interactive dashboards. Every number comes from the notebook outputs; nothing
is invented. See `source_notebook` in each file for provenance.

Label map used everywhere: 0 = N (normal), 1 = S (supraventricular ectopic),
2 = V (ventricular ectopic), 3 = F (fusion), 4 = Q (unknown).

## ann_knn_results.json
From `ecg-classification.ipynb`.

- `dataset`: source, beat/feature counts, train/test sizes, split description, label map, class counts, and the leakage caveat.
- `ann`: architecture string, `test_accuracy`, `per_class` precision/recall/F1/support keyed by class, the 5x5 `confusion_matrix` (rows = true, columns = predicted), and the saved weights file.
- `knn`: k, `test_accuracy`, 5x5 `confusion_matrix`.

## cnn_results.json
From `ecg-cnn.ipynb`.

- `architecture`: the 1D CNN layout and training settings.
- `five_class`: `test_accuracy`, `per_class` metrics, 5x5 `confusion_matrix`.
- `binary`: label map, class counts, `test_accuracy`, `per_class` metrics, 2x2 `confusion_matrix`, `roc_auc`, saved weights file.

## honest_evaluation.json
From `web/notebooks/analysis.ipynb`: the fixed ANN evaluated properly.

- `model`: architecture and weights used.
- `test_set`: file, beat count, per-class counts.
- `fixed_model_accuracy`: accuracy on the shipped test split.
- `bootstrap`: method, resample count, overall accuracy + 95% CI, and `per_class` precision/recall/F1 with F1 95% CIs (`f1_ci95`) and n.
- `calibration`: ECE before/after temperature scaling, the temperature, fit/eval sizes, accuracy before/after.
- `grouped_cv`: method (5-fold GroupKFold over the 48 MIT-BIH records), per-fold accuracies, mean/std, pooled accuracy, pooled per-class metrics, and the note that this is the honest number.
- `onnx`: converted model file, size, max abs diff vs Keras, argmax agreement.
- `demo_sample_beats`: the five demo beats with true label, predicted label, and confidence.
