"""Paired specification contrasts with each specification averaged over its own retained spectra (Table S6, 'all').
Descriptive: mean and pointwise bootstrap 95% CI, no p-values.
Usage: python analysis/specification_contrasts.py
"""
import json

import numpy as np
import pandas as pd

S = pd.read_csv("results/revision/models_scalp_units.csv")
C = pd.read_csv("results/revision/models_seeg_units.csv")
CONTRASTS = {"range": ("F_wide", "F_band"), "model": ("K_wide", "F_wide"), "slope_definition": ("K_wide_asym", "K_wide")}


def ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    b = np.random.default_rng(0).choice(v, (10000, len(v))).mean(1)
    return dict(mean=float(v.mean()), ci95=[float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))], n=int(len(v)))


J = {}
for (ds, cond), g in S.groupby(["dataset", "cond"]):
    J[f"P2_{ds}_{cond}"] = {k: ci(g[f"{a}_P2"] - g[f"{b}_P2"]) for k, (a, b) in CONTRASTS.items()}
P = S[S.dataset == "ds003478"].pivot(index="sub", columns="cond")


def inter(s):
    return (P[f"{s}_avg"]["EO"] - P[f"{s}_avg"]["EC"]) - (P[f"{s}_lap"]["EO"] - P[f"{s}_lap"]["EC"])


J["EOEC_interaction"] = {k: ci(inter(a) - inter(b)) for k, (a, b) in CONTRASTS.items()}
for name, g in (("confirm", C[C.split == "confirm"]), ("ds004752_all", C[C.cohort == "ds004752"]), ("HUP", C[C.cohort == "HUP"])):
    J[f"C1_{name}"] = {k: ci(g[f"{a}_C1"] - g[f"{b}_C1"]) for k, (a, b) in CONTRASTS.items()}
json.dump(J, open("results/revision/paired_specs.json", "w"), indent=1)
for k, v in J.items():
    print(k, {c: (round(d["mean"], 3), [round(x, 3) for x in d["ci95"]], "excl0" if d["ci95"][0] > 0 or d["ci95"][1] < 0 else "")
              for c, d in v.items()})
