"""Comparison of the frozen single-scale predictions with the ANPHY measurements (S3).
Usage: python analysis/kernel_compare_scalp.py
"""
import json

import numpy as np
import pandas as pd

pred = json.load(open("results/predict_anphy/predictions.json"))
D = pd.read_csv("results/anphy_v2/anphy_v2.csv")
if "error" in D:
    D = D[D["error"].isna()]
out = {"predictions_sha256_12": pred["sha256_12"]}
for (st, var), g in D.groupby(["state", "variant"]):
    row = {}
    for m in ("P2_gap_interior", "pair_slope", "P3_slope_n5"):
        v = g[m].dropna().to_numpy()
        mu, se = float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))
        row[m] = {"measured_mean": mu, "sem": se, "n": len(v), "ci95": [mu - 1.96 * se, mu + 1.96 * se]}
        for tag, p in pred["predictions"].items():
            row[m][f"pred_{tag}"] = p[m]
            row[m][f"pred_{tag}_inside_ci95"] = bool(mu - 1.96 * se <= p[m] <= mu + 1.96 * se)
            row[m][f"ratio_measured_over_pred_{tag}"] = float(mu / p[m]) if p[m] != 0 else None
            row[m][f"sign_agrees_{tag}"] = bool(np.sign(mu) == np.sign(p[m]))
    out[f"{st}_{var}"] = row
json.dump(out, open("results/predict_anphy/comparison.json", "w"), indent=2)
print(json.dumps(out, indent=1))
