"""Comparison of the frozen two-component scalp predictions with the measured scalp contrasts (S3).

ds003478 (blind evaluation) and ANPHY wakefulness. Recordings averaged per participant; bootstrap 95% CI over
participants. Writes results/grid_v2/comparison_scalp.json.
Usage: python analysis/mixture_compare_scalp.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results")
fr = json.load(open(R / "grid_v2/two_component_frozen.json"))
rng = np.random.default_rng(0)
out = {"frozen_sha256_12": fr["sha256_12"]}
for name, csv, key, sub in (("ds003478_raw_irasa", "arbitration/ds003478.csv", "ds003478_64", lambda s: s.str.split("_").str[0]),
                            ("anphy_wake_raw_irasa", "arbitration/anphy.csv", "anphy83", lambda s: s)):
    D = pd.read_csv(R / csv)
    D = D[(D.variant == "raw") & (D.method == "irasa")].copy()
    D["subject"] = sub(D["rec"].astype(str))
    S = D.groupby("subject")[["P2_gap_interior", "pair_slope", "P3_slope_n5"]].mean()
    res = {"n_subjects": len(S), "n_recordings": len(D)}
    for m in ("P2_gap_interior", "pair_slope", "P3_slope_n5"):
        v = S[m].to_numpy()
        b = rng.choice(v, (5000, len(v))).mean(1)
        lo, hi = np.quantile(b, [0.025, 0.975])
        res[m] = {"measured_mean": float(v.mean()), "ci95": [float(lo), float(hi)], "n_positive": int((v > 0).sum()),
                  "predictions": {k: float(fr["scalp"][key][k][m]) for k in ("l0_1mm", "l0_2mm", "l0_3mm")}}
        p = res[m]["predictions"]["l0_2mm"]
        res[m]["primary_inside_ci"] = bool(lo <= p <= hi)
        res[m]["primary_sign_agrees"] = bool(np.sign(p) == np.sign(v.mean()))
    out[name] = res
json.dump(out, open(R / "grid_v2/comparison_scalp.json", "w"), indent=2)
print(json.dumps(out, indent=1))
