"""Verify the Kaggle mitbih CSV beat-extraction recipe against MIT-BIH records.

Hypothesis: each CSV row = one annotated beat, window [R, R+187) in a 125 Hz
resampling of the MLII lead, min-max normalized per beat to [0, 1].
"""
import numpy as np
import wfdb
from scipy.signal import resample

REC = "100"
print("downloading record", REC)
record = wfdb.rdrecord(REC, pn_dir="mitdb/1.0.0")
ann = wfdb.rdann(REC, "atr", pn_dir="mitdb/1.0.0")
sig_names = [s.upper() for s in record.sig_name]
mlii = sig_names.index("MLII") if "MLII" in sig_names else 0
sig360 = record.p_signal[:, mlii].astype(np.float64)
print("sig len:", len(sig360), "fs:", record.fs, "lead:", record.sig_name[mlii])

# resample whole signal 360 -> 125 Hz
n125 = int(round(len(sig360) * 125.0 / 360.0))
sig125 = resample(sig360, n125)
print("resampled len:", len(sig125))

SYM2AAMI = {"N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
            "A": 1, "a": 1, "J": 1, "S": 1,
            "V": 2, "E": 2, "F": 3,
            "/": 4, "f": 4, "Q": 4}

beats, labels = [], []
for samp, sym in zip(ann.sample, ann.symbol):
    if sym not in SYM2AAMI:
        continue
    r125 = int(round(samp * 125.0 / 360.0))
    lo, hi = r125, r125 + 187
    if lo < 0 or hi > len(sig125):
        continue
    w = sig125[lo:hi].copy()
    w = (w - w.min()) / (w.max() - w.min() + 1e-12)
    beats.append(w)
    labels.append(SYM2AAMI[sym])
beats = np.array(beats)
labels = np.array(labels)
print("extracted beats:", beats.shape, "label counts:", np.bincount(labels))

# load a chunk of the real test CSV
csv = np.loadtxt("/home/hatch/workspace/diagnostic-demos/sources/ECG-Classification/mitbih_test.csv",
                 delimiter=",", max_rows=21892)
Xc, yc = csv[:, :187], csv[:, 187].astype(int)
print("csv chunk:", Xc.shape, "label counts:", np.bincount(yc))

# for each reconstructed beat, find best correlation among csv rows of same label
rng = np.random.default_rng(0)
idx = rng.choice(len(beats), size=min(40, len(beats)), replace=False)
best = []
for i in idx:
    cand = Xc[yc == labels[i]]
    if len(cand) == 0:
        continue
    b = beats[i]
    # correlation with each candidate
    bc = b - b.mean()
    cc = cand - cand.mean(axis=1, keepdims=True)
    denom = np.sqrt((bc ** 2).sum() * (cc ** 2).sum(axis=1)) + 1e-12
    corr = (cc @ bc) / denom
    j = int(np.argmax(corr))
    best.append((labels[i], float(corr[j])))
best = np.array(best)
print("n matched:", len(best))
print("correlation quantiles (10/50/90):", np.quantile(best[:, 1], [0.1, 0.5, 0.9]))
print("frac with corr > 0.999:", np.mean(best[:, 1] > 0.999))
print("frac with corr > 0.99:", np.mean(best[:, 1] > 0.99))
# check exact equality for the best one
i0 = idx[0]
cand = Xc[yc == labels[i0]]
bc = beats[i0] - beats[i0].mean()
cc = cand - cand.mean(axis=1, keepdims=True)
corr = (cc @ bc) / (np.sqrt((bc ** 2).sum() * (cc ** 2).sum(axis=1)) + 1e-12)
j = int(np.argmax(corr))
print("best corr:", corr[j], "max abs diff:", np.abs(cand[j] - beats[i0]).max())
