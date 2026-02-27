"""Reconstruct heartbeat windows from MIT-BIH records (wfdb) with a documented recipe.

Recipe (mirrors the Kaggle mitbih CSV construction):
  - lead: MLII (channel named MLII, else channel 0)
  - resample whole record 360 Hz -> 125 Hz (Fourier method)
  - for each beat annotation mapped to an AAMI class, take window
    [R, R+187) samples at 125 Hz (187 samples = ~1.5 s starting at the R peak)
  - per-beat min-max normalization to [0, 1]
  - drop beats whose window would run past the signal end
  - record id stored per beat for grouped validation

AAMI mapping follows AAMI EC57 as used in the literature:
  N: N L R e j | S: A a J S | V: V E | F: F | Q: / f Q
(paced beats and non-beat annotations excluded)
"""
import numpy as np
import wfdb
from scipy.signal import resample

SYM2AAMI = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    "A": 1, "a": 1, "J": 1, "S": 1,
    "V": 2, "E": 2,
    "F": 3,
    "/": 4, "f": 4, "Q": 4,
}
RECORDS = ("100 101 102 103 104 105 106 107 108 109 111 112 113 114 115 116 "
           "117 118 119 121 122 123 124 200 201 202 203 205 207 208 209 210 "
           "212 213 214 215 217 219 220 221 222 223 228 230 231 232 233 234").split()


def load_record_beats(rec_name, data_dir="mitdb"):
    record = wfdb.rdrecord(f"{data_dir}/{rec_name}")
    ann = wfdb.rdann(f"{data_dir}/{rec_name}", "atr")
    sig_names = [s.upper() for s in (record.sig_name or [])]
    ch = sig_names.index("MLII") if "MLII" in sig_names else 0
    sig360 = record.p_signal[:, ch].astype(np.float64)
    n125 = int(round(len(sig360) * 125.0 / 360.0))
    sig125 = resample(sig360, n125)
    beats, labels = [], []
    for samp, sym in zip(ann.sample, ann.symbol):
        if sym not in SYM2AAMI:
            continue
        r = int(round(samp * 125.0 / 360.0))
        if r < 0 or r + 187 > len(sig125):
            continue
        w = sig125[r:r + 187].copy()
        denom = w.max() - w.min()
        if denom <= 0:
            continue
        w = (w - w.min()) / denom
        beats.append(w.astype(np.float32))
        labels.append(SYM2AAMI[sym])
    return np.array(beats), np.array(labels, dtype=np.int64)


def main():
    X_parts, y_parts, g_parts = [], [], []
    for rec in RECORDS:
        try:
            b, l = load_record_beats(rec)
        except Exception as e:
            print(f"record {rec}: FAILED ({e})")
            continue
        X_parts.append(b)
        y_parts.append(l)
        g_parts.append(np.full(len(l), rec))
        print(f"record {rec}: {len(l)} beats")
    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    g = np.concatenate(g_parts)
    print("total:", X.shape, "classes:", np.bincount(y), "records:", len(np.unique(g)))
    np.save("X_recon.npy", X)
    np.save("y_recon.npy", y)
    np.save("g_recon.npy", g)


if __name__ == "__main__":
    main()
