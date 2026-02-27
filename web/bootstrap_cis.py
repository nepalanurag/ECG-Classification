"""Bootstrap 95% CIs for per-class precision/recall/F1 and overall accuracy.

Percentile bootstrap (Efron): resample the test set with replacement B times,
recompute each metric, take the 2.5th and 97.5th percentiles. B=2000.
Stratified resampling keeps class proportions stable so rare classes (F, n=162)
are represented in every resample.
"""
import numpy as np
import json

rng = np.random.default_rng(11)
p = np.load("proba_test.npy")
y = np.load("y_test.npy").astype(int)
pred = p.argmax(axis=1)
classes = [0, 1, 2, 3, 4]
names = ["N", "S", "V", "F", "Q"]
B = 2000

# stratified resample indices: resample within each class
class_idx = [np.where(y == c)[0] for c in classes]

boot = {c: {"P": [], "R": [], "F1": []} for c in classes}
boot_acc = []
n = len(y)
for b in range(B):
    idx = np.concatenate([rng.choice(ci, size=len(ci), replace=True) for ci in class_idx])
    yb, pb = y[idx], pred[idx]
    boot_acc.append((pb == yb).mean())
    for c in classes:
        tp = int(((pb == c) & (yb == c)).sum())
        fp = int(((pb == c) & (yb != c)).sum())
        fn = int(((pb != c) & (yb == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        boot[c]["P"].append(prec)
        boot[c]["R"].append(rec)
        boot[c]["F1"].append(f1)

out = {"accuracy": {}, "per_class": {}}
lo, hi = np.percentile(boot_acc, [2.5, 97.5])
out["accuracy"] = {"value": float((pred == y).mean()), "ci95": [float(lo), float(hi)], "n": n}
for c in classes:
    yb_c = (y == c)
    pb_c = (pred == c)
    tp = int((pb_c & yb_c).sum()); fp = int((pb_c & ~yb_c).sum()); fn = int((~pb_c & yb_c).sum())
    prec = tp / (tp + fp); rec = tp / (tp + fn); f1 = 2 * prec * rec / (prec + rec)
    d = {"n": int(yb_c.sum()), "precision": {}, "recall": {}, "f1": {}}
    for m, v in [("precision", prec), ("recall", rec), ("f1", f1)]:
        key = {"precision": "P", "recall": "R", "f1": "F1"}[m]
        lo, hi = np.percentile(boot[c][key], [2.5, 97.5])
        d[m] = {"value": float(v), "ci95": [float(lo), float(hi)]}
    out["per_class"][names[c]] = d

with open("metrics_fixed_model.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
