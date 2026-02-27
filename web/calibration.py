"""Calibration analysis for the fixed ecg_ann_best.h5 model.

- Reliability diagram + Expected Calibration Error (ECE), Guo et al. 2017.
- Temperature scaling fit on a calibration split (half of the test CSV),
  evaluated on the other half. Temperature scaling divides logits by T before
  softmax; it cannot change the argmax, so accuracy is untouched.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
p = np.load("proba_test.npy")      # (21892, 5) softmax probabilities
y = np.load("y_test.npy").astype(int)

# split into calibration-fit half and calibration-eval half (stratified)
from sklearn.model_selection import train_test_split
idx = np.arange(len(y))
i_fit, i_eval = train_test_split(idx, test_size=0.5, random_state=7, stratify=y)
p_fit, y_fit = p[i_fit], y[i_fit]
p_eval, y_eval = p[i_eval], y[i_eval]
print(f"fit n={len(y_fit)}, eval n={len(y_eval)}")


def ece_score(proba, y_true, n_bins=15):
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    acc = (pred == y_true)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    rows = []
    for m in range(n_bins):
        lo, hi = edges[m], edges[m + 1]
        mask = (conf > lo) & (conf <= hi) if m > 0 else (conf <= hi)
        n = mask.sum()
        if n == 0:
            rows.append((0.5 * (lo + hi), 0.0, 0.0, 0))
            continue
        a = acc[mask].mean()
        c = conf[mask].mean()
        ece += (n / len(y_true)) * abs(a - c)
        rows.append((c, a, n / len(y_true), n))
    return ece, rows


def fit_temperature(p_fit, y_fit):
    # logits from probabilities; grid search T on NLL
    eps = 1e-12
    pc = np.clip(p_fit, eps, 1.0)
    logits = np.log(pc)
    Ts = np.concatenate([np.linspace(0.2, 3.0, 281)])
    best_T, best_nll = 1.0, np.inf
    for T in Ts:
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        q = e / e.sum(axis=1, keepdims=True)
        nll = -np.log(np.clip(q[np.arange(len(y_fit)), y_fit], eps, 1.0)).mean()
        if nll < best_nll:
            best_T, best_nll = T, nll
    return best_T


def apply_temperature(p, T):
    eps = 1e-12
    logits = np.log(np.clip(p, eps, 1.0))
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


ece_before, rows_before = ece_score(p_eval, y_eval)
T = fit_temperature(p_fit, y_fit)
p_cal = apply_temperature(p_eval, T)
ece_after, rows_after = ece_score(p_cal, y_eval)
acc_before = (p_eval.argmax(1) == y_eval).mean()
acc_after = (p_cal.argmax(1) == y_eval).mean()
print(f"ECE before: {ece_before:.4f}, after T={T:.2f}: {ece_after:.4f}")
print(f"accuracy before: {acc_before:.4f}, after: {acc_after:.4f} (must be identical)")

# reliability diagram
fig, ax = plt.subplots(figsize=(6, 5))
for rows, lab, ls in [(rows_before, "before (raw softmax)", "-"),
                      (rows_after, f"after temperature scaling (T={T:.2f})", "--")]:
    xs = [r[0] for r in rows]; ys = [r[1] for r in rows]
    ax.plot(xs, ys, marker="o", ms=4, linestyle=ls, label=lab)
ax.plot([0, 1], [0, 1], color="gray", lw=1, label="perfect calibration")
ax.set_xlabel("mean predicted confidence")
ax.set_ylabel("observed accuracy")
ax.set_title("Reliability diagram (held-out half of test set)")
ax.legend(fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig("figures/calibration_curve.png", dpi=150)
print("saved figures/calibration_curve.png")

np.save("temperature.npy", np.array([T]))
with open("calibration_summary.txt", "w") as f:
    f.write(f"ece_before={ece_before:.4f}\n")
    f.write(f"ece_after={ece_after:.4f}\n")
    f.write(f"temperature={T:.4f}\n")
    f.write(f"acc_before={acc_before:.4f}\n")
    f.write(f"acc_after={acc_after:.4f}\n")
    f.write(f"n_fit={len(y_fit)}\n")
    f.write(f"n_eval={len(y_eval)}\n")
print("done")
