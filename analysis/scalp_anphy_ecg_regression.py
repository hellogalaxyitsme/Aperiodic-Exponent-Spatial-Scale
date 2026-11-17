"""ECG regression control in ANPHY-Sleep (Methods 2.5; Results 3.3; S4).

measure: re-extracts the analysed ANPHY epochs with ECG1/ECG2 (same seed and rejection rule), verifies that the EEG
  equals the analysed segments, regresses both ECG channels jointly out of every EEG channel (coefficients from 1-40 Hz
  band-passed segments pooled over states), and computes derivation units as scalp_derivations.py.
analyze: P-2 (full band and five sub-bands) and pair slope after regression, and regressed minus unregressed.
Usage: python analysis/scalp_anphy_ecg_regression.py measure --jobs 8
       python analysis/scalp_anphy_ecg_regression.py analyze --jobs 16
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt, resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
from apex import stats  # noqa: E402
from apex.paths import CACHE  # noqa: E402

SEG = Path(f"{CACHE}/anphy_segments")
SCALP = Path(f"{CACHE}/final_round/scalp")
RES = Path("results/revision")
STATES = ("W", "N2", "N3", "R")
FS = 250


def reextract(k, cached):
    """anphy_v2.extract with ECG1/ECG2 added to the picks; returns {state: (eeg, ecg)}."""
    import mne
    import scalp_anphy_segments_ica as A
    edf, sc = A.edf_for(k)
    raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
    P = A.positions()
    chans = [c for c in raw.ch_names if c.upper() in P]
    aux = [c for c in ("EOG1", "EOG2", "ChEMG1", "ChEMG2") if c in raw.ch_names]
    assert list(chans) == list(cached["chans"]), "channel list differs from cache"
    fs = raw.info["sfreq"]
    rng = np.random.default_rng(k)
    out = {}
    for st in STATES:
        ep = [(t, d) for lab, t, d in sc if lab == st]
        if len(ep) < 5:
            continue
        ep = [ep[i] for i in sorted(rng.choice(len(ep), size=min(27, len(ep)), replace=False))]
        E, G = [], []
        for t, d in ep:
            a0 = int(t * fs)
            X = raw.get_data(picks=chans + aux + ["ECG1", "ECG2"], start=a0, stop=a0 + int(d * fs))
            X = resample_poly(X, FS, int(fs), axis=-1)
            for s in range(0, X.shape[1] - 4 * FS + 1, 4 * FS):
                seg = X[:, s:s + 4 * FS]
                if np.ptp(seg[:len(chans)], axis=1).max() < 1e-3:
                    E.append(seg[:len(chans)].astype(np.float32))
                    G.append(seg[-2:].astype(np.float64))
        if len(E) >= 20:
            out[st] = (np.stack(E), np.stack(G))
    return out


def one(npz):
    import scalp_derivations as FS_
    stem = Path(npz).stem
    try:
        z = np.load(npz, allow_pickle=True)
        if len(z["chans"]) < 20:
            return [f"{stem} skipped (11 channels)"]
        k = int(stem.replace("EPCTL", ""))
        D = reextract(k, z)
        if set(D) != {s for s in STATES if f"{s}_eeg" in z}:
            return [f"{stem} EXCLUDED: state set differs"]
        for st, (E, _) in D.items():
            dev = float(np.abs(E.astype(float) - z[f"{st}_eeg"].astype(float)).max())
            if E.shape != z[f"{st}_eeg"].shape or dev >= 1e-9:
                return [f"{stem} EXCLUDED: {st} EEG differs from cache (shape {E.shape}, max dev {dev:.2e})"]
        b, a = butter(4, [1.0, 40.0], btype="band", fs=FS)
        Xf = np.concatenate([filtfilt(b, a, D[st][0].astype(float), axis=-1) for st in D], axis=0)
        Gf = np.concatenate([filtfilt(b, a, D[st][1], axis=-1) for st in D], axis=0)
        Xf2 = Xf.transpose(1, 0, 2).reshape(Xf.shape[1], -1)
        Gf2 = Gf.transpose(1, 0, 2).reshape(2, -1)
        B = np.linalg.lstsq(Gf2.T, Xf2.T, rcond=None)[0]                     # [2, ch]
        r2 = 1.0 - np.var(Xf2 - B.T @ Gf2, axis=1) / np.var(Xf2, axis=1)
        msgs = [f"{stem}: ECG R2 median {np.median(r2):.4f} max {r2.max():.4f}"]
        for st, (E, G) in D.items():
            G = G - G.mean(-1, keepdims=True)
            Xr = E.astype(float) - np.einsum("rc,srt->sct", B, G)
            msgs.append(FS_.compute_unit(f"anphyecg_{stem}_{st}", Xr, float(FS), z["pos"],
                                         extra=dict(ecg_r2_median=float(np.median(r2)), ecg_r2_max=float(r2.max()))))
        return msgs
    except Exception as e:
        return [f"{stem} ERROR {type(e).__name__}: {e}"]


def measure(jobs):
    items = [p for p in sorted(glob.glob(str(SEG / "EPCTL*.npz"))) if ".tmp" not in p]
    for m in Parallel(n_jobs=jobs)(delayed(one)(p) for p in items):
        print("\n".join(m))


def boot_ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    b = np.random.default_rng(0).choice(v, (10000, len(v))).mean(1)
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def analyze(jobs):
    import scalp_derivations as FS_
    reg = sorted(p for p in glob.glob(str(SCALP / "anphyecg_*.npz")) if ".tmp" not in p)
    base = [str(SCALP / Path(p).name.replace("anphyecg_", "anphy_")) for p in reg]
    R = pd.DataFrame(Parallel(n_jobs=jobs)(delayed(FS_.unit_stats_cached)(p) for p in base + reg))
    R["variant"] = np.where(R.dataset == "anphyecg", "ecg", "raw")
    r2 = {Path(p).stem: float(np.load(p)["ecg_r2_median"]) for p in reg}
    R.to_csv(RES / "ecg_units.csv", index=False)
    cols = ["irasa_all_P2"] + [f"irasa_b{k}_P2" for k in range(5)] + ["irasa_all_pair"]
    J = {"n_units": len(reg), "ecg_r2_median_of_medians": float(np.median(list(r2.values())))}
    for st, g in R.groupby("cond"):
        P = g.pivot(index="sub", columns="variant")
        J[st] = {"n": int(P.shape[0])}
        for c in cols:
            e, r = P[c]["ecg"], P[c]["raw"]
            J[st][c] = dict(ecg=dict(stats.signflip_test(e.dropna().to_numpy(), "greater"), ci95=boot_ci(e)),
                            raw=dict(mean=float(r.mean()), ci95=boot_ci(r)),
                            ecg_minus_raw=dict(stats.signflip_test((e - r).dropna().to_numpy(), "two-sided"), ci95=boot_ci(e - r)))
    json.dump(J, open(RES / "ecg_group.json", "w"), indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["measure", "analyze"])
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    (measure if a.step == "measure" else analyze)(a.jobs)
    print("done", a.step)


if __name__ == "__main__":
    main()
