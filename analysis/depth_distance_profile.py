"""Within-shaft pair exponent by separation bin, centred within patient (Figure 1b; per-bin counts in S2).
Usage: python analysis/depth_distance_profile.py
"""
import numpy as np
import pandas as pd

S = pd.read_csv("results/straddle/straddle_pairs.csv")
S["d"] = np.exp(S.ld)
S["chi_c"] = S.chi - S.groupby(["cohort", "sub"]).chi.transform("mean")      # remove patient level
bins = [3, 5, 7, 10, 14, 20, 30, 80]
S["bin"] = pd.cut(S.d, bins)
for coh, g in S.groupby("cohort"):
    t = g.groupby("bin", observed=True).agg(n=("chi_c", "size"), n_sub=("sub", "nunique"), chi_c=("chi_c", "mean"),
                                             adj_share=("n_between", lambda x: float((x == 0).mean())))
    print(coh)
    print(t.round(3).to_string())
    # per-patient slope using only pairs >= 7 mm
    sl = [np.polyfit(h.ld, h.chi, 1)[0] for _, h in g[g.d >= 7].groupby("sub") if len(h) >= 3 and h.ld.std() > 0]
    sl = np.array(sl)
    print(f"  slope using pairs >= 7 mm: mean {sl.mean():+.3f}, {int((sl > 0).sum())}/{len(sl)}")
