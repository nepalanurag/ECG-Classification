"""Grouped cross-validation for ECG heartbeat classification.

Why GroupKFold over records: beats from the same patient share electrode
placement, heart geometry and baseline quirks. A random split puts beats from
one patient in both train and test, so the model can memorize patient-specific
morphology instead of learning beat types. Grouping by record keeps every
patient wholly in train or wholly in test, which is the honest estimate of how
the model does on a new patient. This is the inter-patient protocol (de Chazal
et al.), as opposed to the leaky intra-patient random split.

Model: same ANN as the original repo (256-128-64-5, ReLU, softmax), trained
from scratch inside each fold, no class weights (balanced weights destabilized
training in testing: grouped accuracy fell from ~0.88 to ~0.68). Early stopping
on a within-fold validation split of the training records.
"""
import numpy as np
import os
import json
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight

CLASS_NAMES = ["N", "S", "V", "F", "Q"]


def build_model(seed):
    tf.keras.utils.set_random_seed(seed)
    m = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu", input_dim=187),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(5, activation="softmax"),
    ])
    m.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def run_cv(X, y, groups, n_splits=5, epochs=15, seed=7, verbose=0, class_weights=False):
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    all_true, all_pred, all_fold = [], [], []
    t0 = time.time()
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        Xtr, ytr = X[tr], y[tr]
        Xte, yte = X[te], y[te]
        cw = None
        if class_weights:
            # balanced class weights over classes present in this fold's training
            # records; a class absent from training gets weight 1 (it cannot be learned)
            counts = np.bincount(ytr, minlength=5)
            cw = {c: float(len(ytr) / (5 * counts[c])) if counts[c] > 0 else 1.0
                  for c in range(5)}
        model = build_model(seed + fold)
        cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                               restore_best_weights=True, verbose=verbose)]
        model.fit(Xtr, ytr, epochs=epochs, batch_size=256, validation_split=0.15,
                  class_weight=cw, callbacks=cb, verbose=verbose)
        p = model.predict(Xte, verbose=0, batch_size=1024).argmax(axis=1)
        acc = float((p == yte).mean())
        per_class = {}
        for c in range(5):
            tp = int(((p == c) & (yte == c)).sum())
            fp = int(((p == c) & (yte != c)).sum())
            fn = int(((p != c) & (yte == c)).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            per_class[CLASS_NAMES[c]] = {"P": prec, "R": rec, "F1": f1, "n": int((yte == c).sum())}
        fold_metrics.append({"fold": fold, "accuracy": acc, "per_class": per_class,
                             "n_test": int(len(yte)),
                             "test_records": sorted(set(groups[te].tolist()))})
        all_true.append(yte); all_pred.append(p); all_fold.append(np.full(len(yte), fold))
        print(f"fold {fold}: acc={acc:.4f} n_test={len(yte)} "
              f"F1={[round(per_class[k]['F1'],3) for k in CLASS_NAMES]}", flush=True)
    all_true = np.concatenate(all_true); all_pred = np.concatenate(all_pred)
    overall_acc = float((all_pred == all_true).mean())
    print(f"pooled accuracy: {overall_acc:.4f}  ({time.time()-t0:.0f}s total)")
    return {"folds": fold_metrics,
            "pooled_accuracy": overall_acc,
            "y_true": all_true.tolist(), "y_pred": all_pred.tolist()}


if __name__ == "__main__":
    import sys
    X = np.load("X_recon.npy"); y = np.load("y_recon.npy"); g = np.load("g_recon.npy")
    n_splits = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    use_weights = len(sys.argv) > 4 and sys.argv[4] == "weights"
    # optional: restrict to a subset of records for a quick sanity run
    if len(sys.argv) > 3 and sys.argv[3] != "-":
        keep = np.array(sys.argv[3].split(","))
        mask = np.isin(g, keep)
        X, y, g = X[mask], y[mask], g[mask]
        print(f"sanity subset: {X.shape}, records {sorted(set(g.tolist()))}")
    res = run_cv(X, y, g, n_splits=n_splits, epochs=epochs, class_weights=use_weights)
    yt = np.array(res.pop("y_true")); yp = np.array(res.pop("y_pred"))
    np.save("grouped_y_true.npy", yt); np.save("grouped_y_pred.npy", yp)
    json.dump(res, open("grouped_cv_results.json", "w"), indent=2)
    print("saved grouped_cv_results.json")
