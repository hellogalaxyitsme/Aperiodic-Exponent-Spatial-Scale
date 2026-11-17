"""Prespecified tests C1-C4 in ds004752 (Table 2).

C1: per-patient slope of the within-shaft pair exponent on ln(contact separation) (lateral temporal grey matter,
3-80 mm, IRASA, 2-40 Hz). C2: scalp pair slope. C3: average reference minus Laplacian (P-2). C4: session-level scalp
exponent on depth exponent. One-sided sign-flip tests with Holm adjustment across the four tests.
Usage: python analysis/depth_prespecified_tests.py --config configs/stageBC_confirm.yaml --out results/stageBC_confirm
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apex import pairs, stats  # noqa: E402
from apex.paths import CACHE  # noqa: E402

DEV = {f"sub-{i:02d}" for i in range(1, 6)}


def load(cache, sub):
    fs = sorted(Path(cache).glob(f"{sub}_ses-*.npz"))
    Z = [np.load(f, allow_pickle=True) for f in fs]
    z0 = Z[0]
    for z in Z[1:]:
        assert list(z["depth_names"]) == list(z0["depth_names"]), f"{sub}: depth layout differs across sessions"
        assert list(z["scalp_names"]) == list(z0["scalp_names"]), f"{sub}: scalp layout differs across sessions"
    cat = lambda k: np.concatenate([z[k] for z in Z], 0).astype(float)
    d = {k: z0[k] for k in ["centers", "depth_names", "depth_xyz", "depth_class", "depth_shaft", "depth_gyrus", "scalp_names", "scalp_pos"]}
    d["sessions"] = Z
    for k in ["Dd_ir", "Dd_mt", "De_ir", "De_mt", "scalp_avg_ir", "scalp_lap_ir", "scalp_vertex_ir", "scalp_avg_mt", "scalp_lap_mt",
              "scalp_hf_ratio"]:
        d[k] = cat(k)
    d["n_sessions"] = len(Z)
    return d


def slope_on_logdist(chi, dist):
    x = np.log(dist)
    x = x - x.mean()
    return (chi - chi.mean(-1, keepdims=True)) @ x / (x @ x)


def coherence_length(centers, D, dist, far_mm=45.0):
    """Per band: y = D / median(D[far]) fitted by a (1 - exp(-delta/lam)). Returns lam [nb] (nan if not estimable)."""
    far = dist >= far_mm
    lam = np.full(len(centers), np.nan)
    if far.sum() < 5:
        return lam
    for k in range(len(centers)):
        y = D[:, k] / np.median(D[far, k])
        r = least_squares(lambda th: th[0] * (1 - np.exp(-dist / np.exp(th[1]))) - y, [1.0, np.log(10.0)],
                          bounds=([0.1, np.log(0.5)], [5.0, np.log(500.0)]), loss="soft_l1", f_scale=0.2)
        lam[k] = np.exp(r.x[1]) if np.log(0.5) + 1e-3 < r.x[1] < np.log(500.0) - 1e-3 else np.nan
    return lam


def rise_fit(chi, dist):
    """Saturating model chi(delta) = chi_inf - Delta exp(-delta / ell). Delta > 0: near pairs flatter (coherence falls
    with frequency); Delta < 0: near pairs steeper. Returns (Delta, ell, chi_inf)."""
    r = least_squares(lambda th: th[0] - th[1] * np.exp(-dist / np.exp(th[2])) - chi, [np.median(chi), 0.0, np.log(8.0)],
                      bounds=([-2, -3, np.log(1.0)], [6, 3, np.log(60.0)]), loss="soft_l1", f_scale=0.2)
    return float(r.x[1]), float(np.exp(r.x[2])), float(r.x[0])


def gamma_from_lam(centers, lam, band):
    m = np.isfinite(lam) & (centers >= band[0]) & (centers <= band[1])
    if m.sum() < 10:
        return np.nan
    return float(-np.polyfit(np.log(centers[m]), np.log(lam[m]), 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default=f"{CACHE}/allpairs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    raw = open(args.config, "rb").read()
    cfg = yaml.safe_load(raw)
    h = hashlib.sha256(raw).hexdigest()[:12]
    subs = cfg["subjects"]
    if cfg["split"] != "confirmatory" and any(s not in DEV for s in subs):
        sys.exit("refusing: confirmatory subjects in a non-confirmatory config")
    A = cfg["analysis"]
    band = tuple(cfg["fit"]["band"])
    band_lo = tuple(A["emg_sensitivity_band"])
    prim = set(A["primary_classes"])
    dmin, dmax = A["dist_range_mm"]
    B = A["n_boot"]
    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bins = np.array(A["dist_bins_mm"], float)

    prow, pair_rows, chan_rows, curves = [], [], [], {}
    for sub in subs:
        d = load(args.cache, sub)
        c = d["centers"]
        T = d["Dd_ir"].shape[0]
        W = rng.multinomial(T, np.ones(T) / T, size=B) / T
        # ---------------- depth pairs
        n = len(d["depth_names"])
        ia, ib = pairs.pair_index(n)
        xyz = d["depth_xyz"]
        dist = np.linalg.norm(xyz[ia] - xyz[ib], axis=1)
        cls = d["depth_class"]
        both_prim = np.array([cls[i] in prim and cls[j] in prim for i, j in zip(ia, ib)])
        same_shaft = d["depth_shaft"][ia] == d["depth_shaft"][ib]
        ipsi = np.sign(xyz[ia, 0]) == np.sign(xyz[ib, 0])
        inr = (dist >= dmin) & (dist <= dmax) & ipsi
        sel = both_prim & inr
        sel_all = inr
        Dir = d["Dd_ir"].mean(0)
        Dmt = d["Dd_mt"].mean(0)
        chi_ir = pairs.band_exponent(c, Dir, band)
        chi_mt = pairs.band_exponent(c, Dmt, band)
        chi_lo = pairs.band_exponent(c, Dir, band_lo)
        chi_hi = pairs.band_exponent(c, Dir, tuple(A["high_band"]))
        for k in range(len(ia)):
            pair_rows.append(dict(sub=sub, a=d["depth_names"][ia[k]], b=d["depth_names"][ib[k]], dist=dist[k], class_a=cls[ia[k]],
                                  class_b=cls[ib[k]], same_shaft=bool(same_shaft[k]), chi_ir=chi_ir[k], chi_mt=chi_mt[k]))
        row = dict(sub=sub, n_trials=T, n_sessions=d["n_sessions"], n_depth=n, n_pairs_primary=int(sel.sum()), n_pairs_all=int(sel_all.sum()))
        if sel.sum() >= 6:
            row["H1_slope_ir"] = float(slope_on_logdist(chi_ir[sel], dist[sel]))
            row["H1_slope_mt"] = float(slope_on_logdist(chi_mt[sel], dist[sel]))
            row["H1_slope_ir_2_20Hz"] = float(slope_on_logdist(chi_lo[sel], dist[sel]))
            row["H1_slope_ir_20_40Hz"] = float(slope_on_logdist(chi_hi[sel], dist[sel]))
            row["rise_Delta"], row["rise_ell_mm"], row["rise_chi_inf"] = rise_fit(chi_ir[sel], dist[sel])
            row["chi_D_near_5mm"] = float(np.median(chi_ir[sel & (dist <= 7)])) if np.any(sel & (dist <= 7)) else np.nan
            row["chi_D_far_ipsi"] = float(np.median(chi_ir[sel & (dist >= 20)])) if np.any(sel & (dist >= 20)) else np.nan
            row["chi_D_contra"] = float(np.median(chi_ir[both_prim & ~ipsi])) if np.any(both_prim & ~ipsi) else np.nan
            row["near_minus_contra"] = row["chi_D_near_5mm"] - row["chi_D_contra"]
            row["H1_slope_ir_crossshaft"] = float(slope_on_logdist(chi_ir[sel & ~same_shaft], dist[sel & ~same_shaft])) if (sel & ~same_shaft).sum() >= 6 else np.nan
            row["H1_slope_ir_withinshaft"] = float(slope_on_logdist(chi_ir[sel & same_shaft], dist[sel & same_shaft])) if (sel & same_shaft).sum() >= 4 else np.nan
            Db = np.einsum("bt,tpk->bpk", W, d["Dd_ir"][:, sel])
            bs = slope_on_logdist(pairs.band_exponent(c, Db, band), dist[sel])
            row["H1_slope_ir_ci_lo"], row["H1_slope_ir_ci_hi"] = stats.bootstrap_ci(bs)
            row["H1_boot_frac_pos"] = float(np.mean(bs > 0))
        row["H1_slope_ir_allgrey"] = float(slope_on_logdist(chi_ir[sel_all], dist[sel_all]))
        # regional-heterogeneity control: cross-shaft pairs whose contacts share the same gyrus label
        gy = d["depth_gyrus"]
        same_gyrus = gy[ia] == gy[ib]
        m_sg = sel & ~same_shaft & same_gyrus
        if m_sg.sum() >= 6:
            row["H1_slope_ir_crossshaft_samegyrus"] = float(slope_on_logdist(chi_ir[m_sg], dist[m_sg]))
        m_dg = sel & ~same_shaft & ~same_gyrus
        if m_dg.sum() >= 6:
            row["H1_slope_ir_crossshaft_diffgyrus"] = float(slope_on_logdist(chi_ir[m_dg], dist[m_dg]))
        if (sel & same_shaft).sum() >= 4:
            row["H1_slope_withinshaft_2_20Hz"] = float(slope_on_logdist(chi_lo[sel & same_shaft], dist[sel & same_shaft]))
            Dws = np.einsum("bt,tpk->bpk", W, d["Dd_ir"][:, sel & same_shaft])
            bw = slope_on_logdist(pairs.band_exponent(c, Dws, band), dist[sel & same_shaft])
            row["H1_withinshaft_ci_lo"], row["H1_withinshaft_ci_hi"] = stats.bootstrap_ci(bw)
        # mesial (secondary): slope where enough ipsilateral mesial pairs exist
        both_mes = np.array([cls[i] == "mesial" and cls[j] == "mesial" for i, j in zip(ia, ib)]) & inr
        if both_mes.sum() >= 6:
            row["H1_slope_ir_mesial"] = float(slope_on_logdist(chi_ir[both_mes], dist[both_mes]))
        # binned curves
        idx = np.digitize(dist, bins) - 1
        for nm, m in (("prim", both_prim), ("all", np.ones_like(both_prim))):
            curves[f"{sub}|depth_{nm}"] = np.array([np.nanmedian(chi_ir[m & (idx == k)]) if np.any(m & (idx == k)) else np.nan for k in range(len(bins) - 1)])
        # ---------------- scalp
        sn = list(d["scalp_names"])
        ns = len(sn)
        row["n_scalp"] = ns
        row["scalp_full_montage"] = ns >= A["full_montage_min"]
        sa, sb = pairs.pair_index(ns)
        sd = np.linalg.norm(d["scalp_pos"][sa] - d["scalp_pos"][sb], axis=1)
        chi_sp = pairs.band_exponent(c, d["De_ir"].mean(0), band)
        row["scalp_pair_slope"] = float(slope_on_logdist(chi_sp, sd))
        row["scalp_pair_slope_2_20Hz"] = float(slope_on_logdist(pairs.band_exponent(c, d["De_ir"].mean(0), band_lo), sd))
        bsp = slope_on_logdist(pairs.band_exponent(c, np.einsum("bt,tpk->bpk", W, d["De_ir"]), band), sd)
        row["scalp_pair_slope_ci_lo"], row["scalp_pair_slope_ci_hi"] = stats.bootstrap_ci(bsp)
        sidx = np.digitize(sd, bins) - 1
        curves[f"{sub}|scalp_pairs_dist"] = sd
        curves[f"{sub}|scalp_pairs_chi"] = chi_sp
        chi_avg = pairs.band_exponent(c, d["scalp_avg_ir"].mean(0), band)
        chi_lap = pairs.band_exponent(c, d["scalp_lap_ir"].mean(0), band)
        chi_vtx = pairs.band_exponent(c, d["scalp_vertex_ir"].mean(0), band)
        chi_avg_lo = pairs.band_exponent(c, d["scalp_avg_ir"].mean(0), band_lo)
        chi_lap_lo = pairs.band_exponent(c, d["scalp_lap_ir"].mean(0), band_lo)
        chi_avg_mt = pairs.band_exponent(c, d["scalp_avg_mt"].mean(0), band)
        chi_lap_mt = pairs.band_exponent(c, d["scalp_lap_mt"].mean(0), band)
        hf = np.median(d["scalp_hf_ratio"], 0)
        for i, nm in enumerate(sn):
            chan_rows.append(dict(sub=sub, ch=nm, chi_avg=chi_avg[i], chi_lap=chi_lap[i], chi_vertex=chi_vtx[i], hf_ratio=hf[i]))
        # absolute scalp vs depth comparison (H3 descriptive), EMG-robust band
        row["chi_scalp_avg_2_20"] = float(np.mean(chi_avg_lo))
        row["chi_depth_near_2_20"] = float(np.median(chi_lo[sel & same_shaft])) if np.any(sel & same_shaft) else np.nan
        row["chi_depth_all_2_20"] = float(np.median(chi_lo[sel])) if np.any(sel) else np.nan
        row["scalp_minus_depth_2_20"] = row["chi_scalp_avg_2_20"] - row["chi_depth_all_2_20"]
        # session-level tracking (H3-tracking): does the scalp exponent follow the depth exponent across days?
        ses_s, ses_d = [], []
        for z in d["sessions"]:
            ses_s.append(float(np.mean(pairs.band_exponent(c, z["scalp_avg_ir"].astype(float).mean(0), band_lo))))
            ses_d.append(float(np.median(pairs.band_exponent(c, z["Dd_ir"].astype(float).mean(0)[sel], band_lo))) if sel.any() else np.nan)
        ses_s, ses_d = np.array(ses_s), np.array(ses_d)
        curves[f"{sub}|session_scalp"], curves[f"{sub}|session_depth"] = ses_s, ses_d
        if len(ses_s) >= 3 and np.all(np.isfinite(ses_d)) and np.ptp(ses_d) > 0:
            x, y = ses_d - ses_d.mean(), ses_s - ses_s.mean()
            row["track_slope"] = float(x @ y / (x @ x))
            row["track_r"] = float(np.corrcoef(ses_d, ses_s)[0, 1])
        row["chi_scalp_avg"] = float(np.mean(chi_avg))
        row["chi_scalp_lap"] = float(np.mean(chi_lap))
        row["scalp_hf_ratio_median"] = float(np.median(hf))
        if row["scalp_full_montage"]:
            row["P2_gap"] = float(np.mean(chi_avg - chi_lap))
            row["P2_gap_2_20Hz"] = float(np.mean(chi_avg_lo - chi_lap_lo))
            row["P2_gap_mt"] = float(np.mean(chi_avg_mt - chi_lap_mt))
            Wa = pairs.band_exponent(c, np.einsum("bt,tck->bck", W, d["scalp_avg_ir"]), band)
            Wl = pairs.band_exponent(c, np.einsum("bt,tck->bck", W, d["scalp_lap_ir"]), band)
            row["P2_gap_ci_lo"], row["P2_gap_ci_hi"] = stats.bootstrap_ci((Wa - Wl).mean(-1))
            # EMG control: channels in the lowest-HF half only
            low = hf <= np.median(hf)
            row["P2_gap_lowHF_channels"] = float(np.mean((chi_avg - chi_lap)[low]))
        # descriptive comparison: scalp avg-ref exponent vs range of depth pair exponents
        row["depth_chi_pairs_min"], row["depth_chi_pairs_max"] = float(np.nanmin(chi_ir[sel_all])), float(np.nanmax(chi_ir[sel_all]))
        row["depth_chi_far_median"] = float(np.nanmedian(chi_ir[sel_all & (dist >= A["far_mm"])])) if np.any(sel_all & (dist >= A["far_mm"])) else np.nan
        row["depth_chi_near_median"] = float(np.nanmedian(chi_ir[sel_all & (dist <= 7)])) if np.any(sel_all & (dist <= 7)) else np.nan
        prow.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    P = pd.DataFrame(prow)
    P.to_csv(out / "patients.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out / "depth_pairs.csv", index=False)
    pd.DataFrame(chan_rows).to_csv(out / "scalp_channels.csv", index=False)
    np.savez_compressed(out / "curves.npz", bins=bins, centers=c, **{k.replace("|", "__"): v for k, v in curves.items()})
    G = dict(config=args.config, config_sha256_12=h, split=cfg["split"], subjects=subs)
    tests = {"H1_slope_ir": "greater", "H1_slope_mt": "greater", "H1_slope_ir_2_20Hz": "greater", "H1_slope_ir_crossshaft": "greater",
             "H1_slope_ir_withinshaft": "greater", "H1_slope_ir_allgrey": "greater", "H1_slope_ir_20_40Hz": "greater",
             "H1_slope_ir_mesial": "greater", "rise_Delta": "two-sided", "near_minus_contra": "two-sided",
             "P2_gap": "greater", "P2_gap_2_20Hz": "greater", "P2_gap_mt": "greater", "P2_gap_lowHF_channels": "greater",
             "scalp_pair_slope": "greater", "scalp_pair_slope_2_20Hz": "greater", "H1_slope_withinshaft_2_20Hz": "greater",
             "H1_slope_ir_crossshaft_samegyrus": "two-sided", "H1_slope_ir_crossshaft_diffgyrus": "two-sided",
             "scalp_minus_depth_2_20": "two-sided", "track_slope": "greater", "track_r": "greater"}
    for k, alt in tests.items():
        if k in P and P[k].notna().sum() >= 2:
            G[k] = stats.signflip_test(P[k].to_numpy(), alt)
    fam = [k for k in A["primary_family"] if k in G]
    if fam:
        adj = stats.holm([G[k]["p"] for k in fam])
        for k, p in zip(fam, adj):
            G[k]["p_holm_primary"] = float(p)
    G["primary_family"] = fam
    json.dump(G, open(out / "group.json", "w"), indent=2, default=float)
    print(json.dumps(G, indent=2, default=float))


if __name__ == "__main__":
    main()
