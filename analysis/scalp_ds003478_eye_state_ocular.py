"""ds003478: eyes-open and eyes-closed segments and ocular controls (Methods 2.5; S4).

Participant-level 4-s segments pooled over rest runs (250 Hz, scalp channels). Variants from the same data and channels:
  base    all kept segments (bad-channel rule and 5x peak-to-peak segment rule; average reference)
  eogreg  HEOG and VEOG regressed jointly out of every scalp channel (coefficients estimated per run on 1-40 Hz
          band-passed kept segments; subtraction applied to the broadband data), then average reference
  nofront periocular channels FP1 FPZ FP2 AF3 AF4 F7 F8 removed, re-referenced to the remaining average
  EO, EC  segments labelled eyes open / eyes closed, equalised to the smaller count (random subset, seed 0)
A segment is labelled with condition c if every event marker in [t0 - 2 s, t0 + 4 s) belongs to c and the window holds
>= 6 'Every 500 ms' markers of c. Participants with < 20 labelled segments in either condition are dropped from EO/EC.
Usage: python analysis/scalp_ds003478_eye_state_ocular.py --out results/ds3478_followup --jobs 16
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apex import hdeeg, stats  # noqa: E402
from apex.paths import DATA  # noqa: E402

DS3478 = Path(f"{DATA}/ds003478")
NONSCALP = {"HEOG", "VEOG", "CB1", "CB2", "EKG", "ECG", "EMG"}
FRONT = {"FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F8"}
METRICS = ("P2_gap_interior", "pair_slope", "P3_slope_n5", "chi_avg")
METHODS = ("irasa", "mtpf")
FS, L = 250.0, 1000
MIN_SEG = 20


def load_run(setf):
    import mne
    r = mne.io.read_raw_eeglab(setf, preload=True, verbose="ERROR")
    r.resample(FS, verbose="ERROR")
    E = pd.read_csv(setf.replace("_eeg.set", "_electrodes.tsv"), sep="\t")
    mont = {n: np.array([x, y, z], float) for n, x, y, z in E[["name", "x", "y", "z"]].itertuples(index=False)}
    names = [c for c in r.ch_names if c.upper() not in NONSCALP and c in mont and np.all(np.isfinite(mont[c]))]
    X = r.get_data(picks=names)
    eog = r.get_data(picks=[c for c in ("HEOG", "VEOG") if c in r.ch_names])
    ev = pd.read_csv(setf.replace("_eeg.set", "_events.tsv"), sep="\t")
    return names, np.array([mont[c] for c in names]), X - X.mean(0, keepdims=True), eog, ev


def labels(ev, n_seg):
    t = ev.onset.to_numpy(float)
    tt = ev.trial_type.astype(str).to_numpy()
    cond = np.where(np.char.startswith(tt.astype(str), "Eyes Open"), "EO", np.where(np.char.startswith(tt.astype(str), "Eyes Closed"), "EC", ""))
    is500 = np.char.find(tt.astype(str), "500 ms") >= 0
    lab = np.array([""] * n_seg, dtype=object)
    for i in range(n_seg):
        t0 = i * L / FS
        w = (t >= t0 - 2) & (t < t0 + 4)
        c = set(cond[w])
        if len(c) == 1 and "" not in c:
            c = c.pop()
            if np.sum(w & is500 & (cond == c) & (t >= t0)) >= 6:
                lab[i] = c
    return lab


def segs(X, n):
    return np.stack([X[:, i * L:(i + 1) * L] for i in range(n)])


def eog_regress(Xa, eog, keep_seg, n):
    import mne
    Xf = mne.filter.filter_data(Xa, FS, 1.0, 40.0, verbose="ERROR")
    Ef = mne.filter.filter_data(eog, FS, 1.0, 40.0, verbose="ERROR")
    idx = np.concatenate([np.arange(i * L, (i + 1) * L) for i in np.nonzero(keep_seg)[0]])
    B, *_ = np.linalg.lstsq(Ef[:, idx].T, Xf[:, idx].T, rcond=None)          # [n_eog, n_ch]
    Eo = eog - eog.mean(1, keepdims=True)
    return Xa - B.T @ Eo, B


def run_metrics(sub, var, S, pos, rows, extra):
    S = S - S.mean(1, keepdims=True)
    for method in METHODS:
        try:
            m = hdeeg.metrics_v2(S, FS, pos, method=method)
            rows.append(dict(sub=sub, variant=var, method=method, n_ch=S.shape[1], n_seg=S.shape[0], **extra, **{k: m[k] for k in METRICS}))
        except Exception as e:
            rows.append(dict(sub=sub, variant=var, method=method, error=f"{type(e).__name__}: {e}"))


def subject_one(sub, files):
    rows = []
    try:
        runs = [load_run(f) for f in files]
        names = [c for c in runs[0][0] if all(c in r[0] for r in runs)]
        pos = runs[0][1][[runs[0][0].index(c) for c in names]]
        # bad channels: per-run median 4-s ptp rule; a channel is kept only if kept in every run
        keep_ch = np.ones(len(names), bool)
        for nm, _, Xa, _, _ in runs:
            S = segs(Xa[[nm.index(c) for c in names]], Xa.shape[1] // L)
            med = np.median(np.ptp(S, axis=-1), axis=0)
            keep_ch &= (med < 5 * np.median(med)) & (med > 0.1 * np.median(med))
        kn = [c for c, k in zip(names, keep_ch) if k]
        P = pos[keep_ch]
        base, reg, lab, nreg = [], [], [], 0
        for nm, _, Xa, eog, ev in runs:
            ix = [nm.index(c) for c in kn]
            n = Xa.shape[1] // L
            S = segs(Xa[ix], n)
            ptp = np.ptp(S, axis=-1)
            ks = np.all(ptp <= 5 * np.median(ptp, axis=0, keepdims=True), axis=1)
            base.append(S[ks])
            lab.append(labels(ev, n)[ks])
            if eog.shape[0] == 2 and np.all(np.std(eog, axis=1) > 0):
                Xr, _ = eog_regress(Xa[ix], eog, ks, n)
                reg.append(segs(Xr, n)[ks])
                nreg += 1
        base, lab = np.concatenate(base), np.concatenate(lab)
        extra = dict(n_runs=len(runs))
        run_metrics(sub, "base", base, P, rows, extra)
        if nreg == len(runs):
            run_metrics(sub, "eogreg", np.concatenate(reg), P, rows, extra)
        fr = np.array([c.upper() not in FRONT for c in kn])
        run_metrics(sub, "nofront", base[:, fr], P[fr], rows, extra)
        neo, nec = int((lab == "EO").sum()), int((lab == "EC").sum())
        k = min(neo, nec)
        if k >= MIN_SEG:
            rng = np.random.default_rng(0)
            for c in ("EO", "EC"):
                ii = np.sort(rng.choice(np.nonzero(lab == c)[0], k, replace=False))
                run_metrics(sub, c, base[ii], P, rows, dict(extra, n_EO=neo, n_EC=nec))
        else:
            rows.append(dict(sub=sub, variant="EO/EC", error=f"too few labelled segments EO={neo} EC={nec}"))
    except Exception as e:
        rows.append(dict(sub=sub, error=f"{type(e).__name__}: {e}"))
    return rows


def boot_mean_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    return stats.bootstrap_ci(rng.choice(x, (n, x.size)).mean(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/ds3478_followup")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    by = defaultdict(list)
    for f in sorted(glob.glob(str(DS3478 / "sub-*/eeg/*task-Rest*_eeg.set"))):
        by[Path(f).name.split("_")[0]].append(f)
    items = sorted(by.items())[: args.limit or None]
    R = pd.DataFrame([r for rr in Parallel(n_jobs=args.jobs)(delayed(subject_one)(s, f) for s, f in items) for r in rr])
    R.to_csv(out / "subjects.csv", index=False)
    Q = R[R["error"].isna()] if "error" in R else R
    G = {"n_subjects": len(items), "errors": R.loc[R["error"].notna(), ["sub", "variant", "error"]].to_dict("records") if "error" in R else []}
    for meth in METHODS:
        q = Q[Q.method == meth]
        W = {m: q.pivot(index="sub", columns="variant", values=m) for m in METRICS}
        g = {}
        for var in W["pair_slope"].columns:
            g[var] = {"P2_gap_interior": stats.signflip_test(W["P2_gap_interior"][var].dropna(), "greater"),
                      "pair_slope": stats.signflip_test(W["pair_slope"][var].dropna(), "greater"),
                      "P3_slope_n5_negative": stats.signflip_test(-W["P3_slope_n5"][var].dropna(), "greater"),
                      "chi_avg_mean": float(W["chi_avg"][var].mean())}
        for a, b in (("EC", "EO"), ("eogreg", "base"), ("nofront", "base")):
            if a in W["pair_slope"] and b in W["pair_slope"]:
                for m in ("P2_gap_interior", "pair_slope", "P3_slope_n5"):
                    d = (W[m][a] - W[m][b]).dropna()
                    g[f"{a}_minus_{b}"] = g.get(f"{a}_minus_{b}", {})
                    g[f"{a}_minus_{b}"][m] = dict(stats.signflip_test(d, "two-sided"), ci95=boot_mean_ci(d),
                                                  ratio_of_means=float(W[m][a][d.index].mean() / W[m][b][d.index].mean()))
        G[meth] = g
    json.dump(G, open(out / "group.json", "w"), indent=2, default=float)
    print(json.dumps(G, indent=1, default=float))


if __name__ == "__main__":
    main()
