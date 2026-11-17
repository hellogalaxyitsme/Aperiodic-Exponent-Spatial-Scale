"""Scalp data loading and preprocessing variants (MODMA, ds003478, ANPHY).

Provides the MODMA and ds003478 loaders used by scalp_derivations.py, and compares raw recordings with a
1-40 Hz / 100 Hz cleaning pipeline. Bad channels: median 4-s peak-to-peak > 5x or < 0.1x the across-channel median;
4-s segments dropped if any channel exceeds 5x its median peak-to-peak.
Usage: python analysis/scalp_preprocessing_variants.py --dataset modma --out results/arbitration --jobs 16
"""
import argparse
import glob
import os
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apex import hdeeg, stats  # noqa: E402
from apex.paths import CACHE, DATA  # noqa: E402

MODMA_RAW = Path(f"{DATA}/MDD Datasets/EEG_128channels_resting_lanzhou_2015")
MODMA_CLEAN = Path(f"{DATA}/MODMA FIF Format 128 channels cleaned")
DS3478 = Path(f"{DATA}/ds003478")
DS3478_CLEAN = Path(f"{DATA}/ds003478_preprocessed_clean_fif")
ANPHY_SEG = Path(f"{CACHE}/anphy_segments")
NONSCALP = {"HEOG", "VEOG", "CB1", "CB2", "EKG", "ECG", "EMG"}
METRICS = ("P2_gap_interior", "pair_slope", "P3_slope_n5", "chi_avg")


def segment(X, fs, clean_channels=True):
    L = int(4 * fs)
    S = np.stack([X[:, i * L:(i + 1) * L] for i in range(X.shape[1] // L)])
    ptp = np.ptp(S, axis=-1)                                   # [seg, ch]
    keep_ch = np.ones(S.shape[1], bool)
    if clean_channels:
        med = np.median(ptp, axis=0)
        keep_ch = (med < 5 * np.median(med)) & (med > 0.1 * np.median(med))
    S, ptp = S[:, keep_ch], ptp[:, keep_ch]
    S = S[np.all(ptp <= 5 * np.median(ptp, axis=0, keepdims=True), axis=1)]
    return S - S.mean(1, keepdims=True), keep_ch


def mimic(X, fs):
    import mne
    from scipy.signal import resample_poly
    Y = mne.filter.filter_data(X.astype(float), fs, 1.0, 40.0, verbose="ERROR")
    from fractions import Fraction
    fr = Fraction(100.0 / fs).limit_denominator(1000)
    return resample_poly(Y, fr.numerator, fr.denominator, axis=-1), 100.0


METHODS = os.environ.get("ARB_METHODS", "irasa,mt,mtpf").split(",")   # read in workers too; mtpf = multitaper, 7-16 Hz excluded from the fit (post hoc, E025)


def run_variants(tag, variants):
    rows = []
    for var, (S, fs, pos) in variants.items():
        for method in METHODS:
            try:
                m = hdeeg.metrics_v2(S, fs, pos, method=method)
                rows.append(dict(rec=tag, variant=var, method=method, n_ch=S.shape[1], n_seg=S.shape[0], fs=fs, **{k: m[k] for k in METRICS}))
            except Exception as e:
                rows.append(dict(rec=tag, variant=var, method=method, error=f"{type(e).__name__}: {e}"))
    return rows


def modma_one(mat):
    import mne
    from scipy.io import loadmat
    try:
        sid = Path(mat).name[:8]
        clean = sorted(MODMA_CLEAN.glob(f"{sid}*.fif"))[0]
        rc = mne.io.read_raw_fif(clean, preload=True, verbose="ERROR").pick("eeg")
        posc = np.array([c["loc"][:3] for c in rc.info["chs"]]) * 1000.0
        okc = np.linalg.norm(posc, axis=1) > 1
        Xc = rc.get_data()[okc]
        Sc, _ = segment(Xc - Xc.mean(0, keepdims=True), rc.info["sfreq"], clean_channels=False)
        M = loadmat(mat)
        key = [k for k in M if not k.startswith("__") and isinstance(M[k], np.ndarray) and M[k].ndim == 2 and M[k].shape[0] >= 128][0]
        fs = float(np.ravel(M["samplingRate"])[0])
        X = M[key][:len(posc)] * 1e-6                             # EGI export in microvolts
        X, pos = X[okc[:X.shape[0]]], posc[okc][:X.shape[0]]
        Xa = X - X.mean(0, keepdims=True)
        Sr, kc = segment(Xa, fs)
        Xm, fsm = mimic(Xa[kc], fs)
        Sm, _ = segment(Xm, fsm, clean_channels=False)
        return run_variants(sid, {"raw": (Sr, fs, pos[kc]), "mimic": (Sm, fsm, pos[kc]), "clean": (Sc, rc.info["sfreq"], posc[okc])})
    except Exception as e:
        return [dict(rec=Path(mat).name, error=f"{type(e).__name__}: {e}")]


def ds3478_one(setf):
    import mne
    try:
        tag = Path(setf).name.split("_eeg")[0]
        r = mne.io.read_raw_eeglab(setf, preload=True, verbose="ERROR")
        r.pick([c for c in r.ch_names if c.upper() not in NONSCALP])
        # .set positions are NaN; BIDS electrodes.tsv holds coordinates in mm
        E = pd.read_csv(setf.replace("_eeg.set", "_electrodes.tsv"), sep="\t")
        mont = {n: np.array([x, y, z], float) for n, x, y, z in E[["name", "x", "y", "z"]].itertuples(index=False)}
        names = [c for c in r.ch_names if c in mont and np.all(np.isfinite(mont[c]))]
        r.pick(names)
        pos = np.array([mont[c] for c in names])
        mont = {k: v / 1000.0 for k, v in mont.items()}                      # metres, as used for the cleaned file below
        r.resample(250.0, verbose="ERROR")
        X = r.get_data()
        Xa = X - X.mean(0, keepdims=True)
        Sr, kc = segment(Xa, 250.0)
        Xm, fsm = mimic(Xa[kc], 250.0)
        Sm, _ = segment(Xm, fsm, clean_channels=False)
        V = {"raw": (Sr, 250.0, pos[kc]), "mimic": (Sm, fsm, pos[kc])}
        if os.environ.get("ARB_ONLY_CLEAN"):                              # cleaned variant only
            V = {}
        cf = DS3478_CLEAN / tag.split("_")[0] / "eeg" / f"{tag}_eeg_cleaned_raw.fif"   # files sit in sub-*/eeg/
        if cf.exists():
            rc = mne.io.read_raw_fif(cf, preload=True, verbose="ERROR")
            rc.pick([c for c in names if c in rc.ch_names])
            Xc = rc.get_data()
            Sc, _ = segment(Xc - Xc.mean(0, keepdims=True), rc.info["sfreq"], clean_channels=False)
            V["clean"] = (Sc, rc.info["sfreq"], np.array([mont[c] for c in rc.ch_names]) * 1000.0)
        return run_variants(tag, V)
    except Exception as e:
        return [dict(rec=Path(setf).name, error=f"{type(e).__name__}: {e}")]


def anphy_one(npz):
    from scipy.signal import butter, resample_poly, sosfiltfilt
    import mne
    try:
        z = np.load(npz, allow_pickle=True)
        if "W_eeg" not in z:
            return []
        S = z["W_eeg"].astype(float)
        S = S - S.mean(1, keepdims=True)
        Y = mne.filter.filter_data(S.reshape(-1, S.shape[-1]), 250.0, None, 40.0, verbose="ERROR")
        Y = sosfiltfilt(butter(4, 1.0, "highpass", fs=250.0, output="sos"), Y, axis=-1)
        Y = resample_poly(Y, 2, 5, axis=-1).reshape(S.shape[0], S.shape[1], -1)
        return run_variants(Path(npz).stem, {"raw": (S, 250.0, z["pos"]), "mimic": (Y, 100.0, z["pos"])})
    except Exception as e:
        return [dict(rec=Path(npz).name, error=f"{type(e).__name__}: {e}")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["modma", "ds003478", "anphy"])
    ap.add_argument("--out", default="results/arbitration")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--methods", default="irasa,mt,mtpf")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.environ["ARB_METHODS"] = args.methods            # loky workers inherit the environment
    METHODS[:] = args.methods.split(",")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.dataset == "modma":
        items, fn = sorted(glob.glob(str(MODMA_RAW / "*.mat"))), modma_one
    elif args.dataset == "ds003478":
        items, fn = sorted(glob.glob(str(DS3478 / "sub-*/eeg/*task-Rest*_eeg.set"))), ds3478_one
    else:
        items, fn = [p for p in sorted(glob.glob(str(ANPHY_SEG / "EPCTL*.npz"))) if ".tmp" not in p], anphy_one
    if args.limit:
        items = items[:args.limit]
    R = pd.DataFrame([r for rr in Parallel(n_jobs=args.jobs)(delayed(fn)(i) for i in items) for r in rr])
    R.to_csv(out / f"{args.dataset}{args.tag}.csv", index=False)
    Q = R[R["error"].isna()] if "error" in R else R
    G = {"n_items": len(items), "errors": R.loc[R["error"].notna(), ["rec", "error"]].head(20).to_dict("records") if "error" in R else []}
    for (var, meth), g in Q.groupby(["variant", "method"]):
        G[f"{var}_{meth}"] = {"P2_gap_interior": stats.signflip_test(g["P2_gap_interior"].to_numpy(), "greater"),
                              "pair_slope": stats.signflip_test(g["pair_slope"].to_numpy(), "greater"),
                              "P3_slope_n5_negative": stats.signflip_test(-g["P3_slope_n5"].to_numpy(), "greater"),
                              "chi_avg_mean": float(g["chi_avg"].mean())}
    json.dump(G, open(out / f"{args.dataset}{args.tag}_group.json", "w"), indent=2, default=float)
    print(json.dumps(G, indent=1, default=float))


if __name__ == "__main__":
    main()
