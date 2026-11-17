"""Depth robustness tests (Results 3.1, Table S1).

Per patient: C1 reproduction, shaft-centred slope, noise-floor subtraction, evoked response removed, maintenance window,
2-20 Hz, band-resolved slope (2-80 Hz). Group sign-flip tests; mixed model of chi on ln(separation) with patient random
slopes (nominal); leave-one-out sensitivity.
Usage: python analysis/depth_robustness_tests.py --out results/robust
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apex import io, pairs, stats  # noqa: E402
from apex.paths import CACHE  # noqa: E402

CACHE = Path(f"{CACHE}/robust")
ALLP = Path(f"{CACHE}/allpairs")
DEV = {f"sub-{i:02d}" for i in range(1, 6)}
B20, B40 = (2.0, 20.0), (2.0, 40.0)


def slope(chi, ld):
    x = ld - ld.mean()
    return float((chi - chi.mean()) @ x / (x @ x)) if len(x) >= 3 and (x @ x) > 0 else np.nan


def shaft_centred(chi, ld, shaft):
    c, x = chi.copy(), ld.copy()
    for s in np.unique(shaft):
        m = shaft == s
        c[m] -= c[m].mean(); x[m] -= x[m].mean()
    return float(c @ x / (x @ x)) if (x @ x) > 0 else np.nan


def load_subject(sub):
    Z = [np.load(f, allow_pickle=True) for f in sorted(CACHE.glob(f"{sub}_ses-*.npz"))]
    z0 = Z[0]
    d = {k: z0[k] for k in ("depth_names", "depth_xyz", "depth_class", "depth_shaft", "depth_idx", "pair_a", "pair_b",
                            "centers_depth", "centers_scalp", "scalp_pos")}
    for k in ("floor", "Dd_orig", "Dd_evk", "Dd_maint", "De_orig", "De_evk", "De_maint"):
        d[k] = np.concatenate([z[k] for z in Z], 0).astype(float)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/robust")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, pair_rows, curves = [], [], {}
    for sub in io.subjects():
        d = load_subject(sub)
        cd, cs = d["centers_depth"], d["centers_scalp"]
        a, b = d["pair_a"], d["pair_b"]
        xyz, cls, shaft, idx = d["depth_xyz"], d["depth_class"], d["depth_shaft"], d["depth_idx"]
        dist = np.linalg.norm(xyz[a] - xyz[b], axis=1)
        prim = (cls[a] == "lateral_temporal") & (cls[b] == "lateral_temporal") & (dist >= 3) & (dist <= 80)
        ld = np.log(dist[prim])
        sh = shaft[a][prim]
        r = dict(sub=sub, split="dev" if sub in DEV else "confirm", n_pairs=int(prim.sum()),
                 n_shafts=int(len(np.unique(sh))), spacing_mm=float(np.median(dist[prim & (idx[b] - idx[a] == 1)])) if np.any(prim & (idx[b] - idx[a] == 1)) else np.nan)
        if prim.sum() >= 3:
            Dm = {v: d[f"Dd_{v}"].mean(0)[prim] for v in ("orig", "evk", "maint")}
            chi = {v: pairs.band_exponent(cd, Dm[v], B40) for v in Dm}
            r["C1_orig"] = slope(chi["orig"], ld)
            r["C1_shaft_centred"] = shaft_centred(chi["orig"], ld, sh)
            r["C1_evk"] = slope(chi["evk"], ld)
            r["C1_maint"] = slope(chi["maint"], ld)
            r["C1_orig_2_20"] = slope(pairs.band_exponent(cd, Dm["orig"], B20), ld)
            r["C1_maint_2_20"] = slope(pairs.band_exponent(cd, Dm["maint"], B20), ld)
            fl = d["floor"].mean(0)[prim][:, None]
            Dfs = Dm["orig"] - fl
            okf = np.all(Dfs > 0, axis=1)
            r["C1_floorsub"] = slope(pairs.band_exponent(cd, Dfs[okf], B40), ld[okf])
            d40 = np.array([np.exp(np.interp(np.log(40.0), np.log(cd), np.log(row))) for row in Dm["orig"]])
            r["snr_40Hz_median"] = float(np.median(d40 / fl.ravel()))
            # same-midpoint nested contrast
            grp = defaultdict(list)
            ai, bi = idx[a][prim], idx[b][prim]
            for k in range(prim.sum()):
                grp[(sh[k], ai[k] + bi[k])].append(k)
            nest = []
            for ks in grp.values():
                if len(ks) >= 2 and len({bi[k] - ai[k] for k in ks}) >= 2:
                    xk = ld[ks] - ld[ks].mean()                       # 2+ pairs sharing a midpoint
                    nest.append(float((chi["orig"][ks] - chi["orig"][ks].mean()) @ xk / (xk @ xk)))
            r["nested_midpoint_slope"] = float(np.nanmean(nest)) if nest else np.nan
            r["n_nested_groups"] = len(nest)
            # band-resolved shaft-centred slope, sliding 9-band window
            w = 9
            curve = []
            for s0 in range(0, len(cd) - w + 1):
                band = (cd[s0] * 0.999, cd[s0 + w - 1] * 1.001)
                curve.append(shaft_centred(pairs.band_exponent(cd, Dm["orig"], band), ld, sh))
            curves[sub] = np.array(curve)
            for k in range(prim.sum()):
                pair_rows.append(dict(sub=sub, shaft=sh[k], ld=ld[k], chi=chi["orig"][k], chi_maint=chi["maint"][k]))
        # scalp (C2) per variant
        sp = d["scalp_pos"]
        sa, sb = np.triu_indices(len(sp), 1)
        sld = np.log(np.linalg.norm(sp[sa] - sp[sb], axis=1))
        for v in ("orig", "evk", "maint"):
            r[f"C2_{v}"] = slope(pairs.band_exponent(cs, d[f"De_{v}"].mean(0), B20), sld)
        rows.append(r)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    R = pd.DataFrame(rows)
    # ---- like-for-like scalp pairs vs depth pairs (2-20 Hz), from the all-pairs cache
    lfl = []
    for sub in io.subjects():
        Z = [np.load(f, allow_pickle=True) for f in sorted(ALLP.glob(f"{sub}_ses-*.npz"))]
        c = Z[0]["centers"]
        Dd = np.concatenate([z["Dd_ir"] for z in Z], 0).astype(float).mean(0)
        De = np.concatenate([z["De_ir"] for z in Z], 0).astype(float).mean(0)
        xyz, cls = Z[0]["depth_xyz"], Z[0]["depth_class"]
        ia, ib = pairs.pair_index(len(xyz))
        dd = np.linalg.norm(xyz[ia] - xyz[ib], axis=1)
        ipsi = np.sign(xyz[ia, 0]) == np.sign(xyz[ib, 0])
        lt = (cls[ia] == "lateral_temporal") & (cls[ib] == "lateral_temporal") & ipsi
        sp = Z[0]["scalp_pos"]
        sa, sb = np.triu_indices(len(sp), 1)
        ds = np.linalg.norm(sp[sa] - sp[sb], axis=1)
        chi_d = pairs.band_exponent(c, Dd, B20)
        chi_s = pairs.band_exponent(c, De, B20)
        row = dict(sub=sub, depth_pairs_20_80=float(np.median(chi_d[lt & (dd >= 20) & (dd <= 80)])) if np.any(lt & (dd >= 20) & (dd <= 80)) else np.nan,
                   depth_pairs_contra=float(np.median(chi_d[(cls[ia] == "lateral_temporal") & (cls[ib] == "lateral_temporal") & ~ipsi])) if np.any(~ipsi) else np.nan,
                   scalp_pairs_lt80=float(np.median(chi_s[ds < 80])) if np.any(ds < 80) else np.nan,
                   scalp_pairs_all=float(np.median(chi_s)), scalp_pairs_gt120=float(np.median(chi_s[ds > 120])) if np.any(ds > 120) else np.nan)
        row["scalp80_minus_depth20_80"] = row["scalp_pairs_lt80"] - row["depth_pairs_20_80"]
        row["scalp120_minus_depth_contra"] = row["scalp_pairs_gt120"] - row["depth_pairs_contra"]
        lfl.append(row)
    L = pd.DataFrame(lfl)
    R = R.merge(L, on="sub")
    R.to_csv(out / "robust_patients.csv", index=False)
    P = pd.DataFrame(pair_rows)
    P.to_csv(out / "robust_pairs.csv", index=False)
    np.savez_compressed(out / "band_curves.npz", **curves)
    # ---- group statistics
    G = {}
    stat_cols = ["C1_orig", "C1_shaft_centred", "C1_floorsub", "C1_evk", "C1_maint", "C1_orig_2_20", "C1_maint_2_20", "nested_midpoint_slope",
                 "C2_orig", "C2_evk", "C2_maint"]
    for split in ("confirm", "dev", "all"):
        S = R if split == "all" else R[R.split == split]
        G[split] = {c: stats.signflip_test(S[c].to_numpy(), "greater") for c in stat_cols if S[c].notna().sum() >= 2}
        G[split]["scalp80_minus_depth20_80"] = stats.signflip_test(S["scalp80_minus_depth20_80"].to_numpy(), "two-sided")
        G[split]["scalp120_minus_depth_contra"] = stats.signflip_test(S["scalp120_minus_depth_contra"].to_numpy(), "two-sided")
    # sub-10 reconciliation + leave-one-out on the confirmatory set (C1 orig)
    Rc = R[R.split == "confirm"]
    G["confirm_without_sub10"] = {c: stats.signflip_test(Rc[Rc["sub"] != "sub-10"][c].to_numpy(), "greater") for c in ("C1_orig", "C1_shaft_centred")}
    G["confirm_leave_one_out_C1_p"] = {s: stats.signflip_test(Rc[Rc["sub"] != s]["C1_orig"].to_numpy(), "greater")["p"] for s in Rc["sub"]}
    G["sub10"] = R[R["sub"] == "sub-10"].iloc[0].to_dict()
    # mixed model
    try:
        import statsmodels.formula.api as smf
        for split, S in (("confirm", P[~P["sub"].isin(DEV)]), ("all", P)):
            S = S.copy()
            S["shaft_id"] = S["sub"] + "_" + S["shaft"]
            md = smf.mixedlm("chi ~ ld", S, groups=S["sub"], re_formula="~ld", vc_formula={"shaft": "0 + C(shaft_id)"})
            fit = md.fit(method="lbfgs", reml=True)
            G[f"mixed_{split}"] = dict(slope=float(fit.params["ld"]), se=float(fit.bse["ld"]), p_two_sided=float(fit.pvalues["ld"]),
                                       ci=[float(v) for v in fit.conf_int().loc["ld"]], n_pairs=len(S), n_subjects=int(S["sub"].nunique()),
                                       converged=bool(fit.converged))
    except Exception as e:
        G["mixed_error"] = str(e)
    json.dump(G, open(out / "group.json", "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in G.items() if k != "sub10"}, indent=1, default=float))


if __name__ == "__main__":
    main()
