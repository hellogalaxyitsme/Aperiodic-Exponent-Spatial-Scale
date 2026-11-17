"""ANPHY-Sleep: segment extraction and ICA control (Methods 2.5; S4).

extract: 30-s epochs scored W, N2, N3, R (27 per state, seeded), 250 Hz, 4-s segments, 1 mV peak-to-peak rejection;
  83 EEG channels with positions, EOG and chin EMG.
ica: one FastICA per participant (average reference, 25 components); components flagged as ocular or muscular removed.
Usage: python analysis/scalp_anphy_segments_ica.py --out results/anphy_v2 --jobs 8
"""
import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import resample_poly, welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apex import hdeeg, stats  # noqa: E402
from apex.paths import CACHE, DATA  # noqa: E402

ANPHY = Path(f"{DATA}/ANPHY_Sleep")
EDF_CACHE = Path(f"{CACHE}/anphy")
SEG_CACHE = Path(f"{CACHE}/anphy_segments")
STATES = ("W", "N2", "N3", "R")
FS = 250


def positions():
    rows = [l.split() for l in open(ANPHY / "Co-registered average positions.pos").read().splitlines() if l.strip()]
    return {r[1].upper(): np.array([float(v) for v in r[2:]]) * 10.0 for r in rows if len(r) == 5}


def edf_for(k):
    zname = sorted(ANPHY.glob(f"EPCTL{k:02d}*.zip"))[0]
    EDF_CACHE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zname) as zf:
        edf = [n for n in zf.namelist() if n.lower().endswith(".edf")][0]
        txt = [n for n in zf.namelist() if n.lower().endswith(".txt")][0]
        dst = EDF_CACHE / Path(edf).name.replace(" ", "_")
        if not dst.exists():
            tmp = dst.with_suffix(".part")
            with zf.open(edf) as src, open(tmp, "wb") as out:
                while True:
                    b = src.read(1 << 24)
                    if not b:
                        break
                    out.write(b)
            tmp.rename(dst)
        sc = re.findall(r"([A-Za-z0-9]+)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)", zf.read(txt).decode(errors="replace"))
    return dst, [(lab, float(t), float(d)) for lab, t, d in sc]


def extract(k, n_epochs=27):
    out = SEG_CACHE / f"EPCTL{k:02d}.npz"
    if out.exists():
        return out
    import mne
    edf, sc = edf_for(k)
    raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
    P = positions()
    chans = [c for c in raw.ch_names if c.upper() in P]
    aux = [c for c in ("EOG1", "EOG2", "ChEMG1", "ChEMG2") if c in raw.ch_names]
    fs = raw.info["sfreq"]
    rng = np.random.default_rng(k)
    res = {"chans": np.array(chans), "aux": np.array(aux), "pos": np.stack([P[c.upper()] for c in chans])}
    for st in STATES:
        ep = [(t, d) for lab, t, d in sc if lab == st]
        if len(ep) < 5:
            continue
        ep = [ep[i] for i in sorted(rng.choice(len(ep), size=min(n_epochs, len(ep)), replace=False))]
        E, Aux = [], []
        for t, d in ep:
            a0 = int(t * fs)
            X = raw.get_data(picks=chans + aux, start=a0, stop=a0 + int(d * fs))
            X = resample_poly(X, FS, int(fs), axis=-1)
            for s in range(0, X.shape[1] - 4 * FS + 1, 4 * FS):
                seg = X[:, s:s + 4 * FS]
                if np.ptp(seg[:len(chans)], axis=1).max() < 1e-3:
                    E.append(seg[:len(chans)].astype(np.float32))
                    Aux.append(seg[len(chans):].astype(np.float32))
        if len(E) >= 20:
            res[f"{st}_eeg"] = np.stack(E)
            res[f"{st}_aux"] = np.stack(Aux)
    SEG_CACHE.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **res)
    tmp.rename(out)
    return out


def ica_clean(z):
    """Fit one ICA on all states; return dict state -> cleaned [seg, ch, t] and the component counts."""
    import mne
    chans, aux, pos = list(z["chans"]), list(z["aux"]), z["pos"]
    states = [s for s in STATES if f"{s}_eeg" in z]
    segs = {s: z[f"{s}_eeg"].astype(float) for s in states}
    auxs = {s: z[f"{s}_aux"].astype(float) for s in states}
    cat = np.concatenate([np.concatenate(list(segs[s]), axis=1) for s in states], axis=1)
    cat_aux = np.concatenate([np.concatenate(list(auxs[s]), axis=1) for s in states], axis=1)
    types = ["eeg"] * len(chans) + ["eog" if a.startswith("EOG") else "emg" for a in aux]
    info = mne.create_info(chans + aux, FS, types)
    raw = mne.io.RawArray(np.vstack([cat, cat_aux]), info, verbose="ERROR")
    raw.set_montage(mne.channels.make_dig_montage(dict(zip(chans, pos / 1000.0)), coord_frame="head"), on_missing="ignore", verbose="ERROR")
    raw.set_eeg_reference("average", verbose="ERROR")
    raw.filter(1.0, None, verbose="ERROR")
    ica = mne.preprocessing.ICA(n_components=25, method="fastica", random_state=0, max_iter=800, verbose="ERROR")
    ica.fit(raw, picks="eeg", verbose="ERROR")
    eog = [c for c in aux if c.startswith("EOG")]
    bad_eog = ica.find_bads_eog(raw, ch_name=eog, verbose="ERROR")[0] if eog else []
    bad_mus = ica.find_bads_muscle(raw, verbose="ERROR")[0]
    ica.exclude = sorted(set(bad_eog) | set(bad_mus))
    out = {}
    for s in states:
        cleaned = []
        for seg in segs[s]:
            r = mne.io.RawArray(seg, mne.create_info(chans, FS, "eeg"), verbose="ERROR")
            r.set_eeg_reference("average", verbose="ERROR")
            cleaned.append(ica.apply(r, verbose="ERROR").get_data())
        out[s] = np.stack(cleaned)
    return out, len(bad_eog), len(bad_mus)


def process(k):
    try:
        z = dict(np.load(extract(k), allow_pickle=True))
        pos = z["pos"]
        rows = []
        cleaned, n_eog, n_mus = ica_clean(z)
        for st in STATES:
            if f"{st}_eeg" not in z:
                continue
            S = z[f"{st}_eeg"].astype(float)
            S = S - S.mean(1, keepdims=True)
            aux = list(z["aux"])
            emg = [i for i, a in enumerate(aux) if a.startswith("ChEMG")]
            emg_pow = np.nan
            if emg:
                f, P = welch(z[f"{st}_aux"][:, emg].astype(float), fs=FS, nperseg=FS)
                emg_pow = float(np.log10(P[..., (f >= 30) & (f <= 100)].mean()))
            for var, data in (("raw", S), ("ica", cleaned[st])):
                m = hdeeg.metrics_v2(data, FS, pos)
                rows.append(dict(subject=f"EPCTL{k:02d}", state=st, variant=var, chin_emg_log10pow=emg_pow,
                                 ica_n_eog=n_eog, ica_n_muscle=n_mus, **m))
        return rows
    except Exception as e:
        return [dict(subject=f"EPCTL{k:02d}", error=f"{type(e).__name__}: {e}")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/anphy_v2")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--subjects", type=int, nargs="*")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    avail = sorted(int(re.match(r"EPCTL(\d+)", p.name).group(1)) for p in ANPHY.glob("EPCTL*.zip"))
    subs = [k for k in (args.subjects or avail) if k in avail]
    rows = [r for rr in Parallel(n_jobs=args.jobs)(delayed(process)(k) for k in subs) for r in rr]
    D = pd.DataFrame(rows)
    D.to_csv(out / "anphy_v2.csv", index=False)
    if "error" in D:
        print(D.loc[D["error"].notna(), ["subject", "error"]].to_string())
        D = D[D["error"].isna()]
    G = {"n_subjects": int(D.subject.nunique())}
    for (st, var), g in D.groupby(["state", "variant"]):
        key = f"{st}_{var}"
        G[key] = {"n": len(g)}
        for c, alt in (("P2_gap_all", "greater"), ("P2_gap_interior", "greater"), ("pair_slope", "greater"),
                       ("pair_slope_incoh_removed", "greater")):
            G[key][c] = stats.signflip_test(g[c].to_numpy(), alt, n_perm=20000)
        for c in ("P3_slope_n5", "P3_slope_n8"):
            G[key][c + "_negative"] = stats.signflip_test(-g[c].to_numpy(), "greater", n_perm=20000)
        G[key]["incoh_frac_at_nn_median"] = float(g.incoh_frac_at_nn.median())
        G[key]["incoh_exponent_median"] = float(g.incoh_exponent.median())
        G[key]["chin_emg_median"] = float(g.chin_emg_log10pow.median())
    # REM vs wake (atonia) — does the pair slope follow chin EMG?
    raw = D[D.variant == "raw"].pivot_table(index="subject", columns="state", values=["pair_slope", "chin_emg_log10pow", "P2_gap_interior"])
    for a, b in (("R", "W"), ("N3", "W"), ("N2", "W")):
        try:
            G[f"{a}_minus_{b}_pair_slope"] = stats.signflip_test((raw["pair_slope"][a] - raw["pair_slope"][b]).dropna().to_numpy(), "two-sided")
            G[f"{a}_minus_{b}_chin_emg"] = stats.signflip_test((raw["chin_emg_log10pow"][a] - raw["chin_emg_log10pow"][b]).dropna().to_numpy(), "two-sided")
        except KeyError:
            pass
    # ICA effect
    piv = D.pivot_table(index=["subject", "state"], columns="variant", values=["P2_gap_interior", "pair_slope", "P3_slope_n5"])
    for c in ("P2_gap_interior", "pair_slope", "P3_slope_n5"):
        G[f"ica_minus_raw_{c}"] = stats.signflip_test((piv[c]["ica"] - piv[c]["raw"]).dropna().to_numpy(), "two-sided", n_perm=20000)
    json.dump(G, open(out / "group.json", "w"), indent=2, default=float)
    print(json.dumps(G, indent=1, default=float))


if __name__ == "__main__":
    main()
