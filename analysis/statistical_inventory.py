"""Inventory of the exploratory family: every participant-level sign-flip p-value with its source and
Benjamini-Hochberg q (Supporting Information Data S1). Writes results/revision/statistical_inventory.csv.
Usage: python analysis/statistical_inventory.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


R = Path("results")
FAMILY = ["robust/group.json", "anphy_v2/group.json", "straddle/group.json", "grid_v2/measure_group.json",
          "ds3478_followup/group.json", "final_round/seeg_group.json", "final_round/grid_robustness.json",
          "final_round/scalp_group.json"]
CHECKS = ["revision/models_group.json", "revision/eog_eyestate_group.json", "revision/interp_group.json", "revision/ecg_group.json"]


def walk(o, path, rows, src, n_group=None, scalp=False):
    if isinstance(o, dict):
        if "p" in o and isinstance(o["p"], (int, float)):
            if scalp:
                key = path.split("/")[0]
                if not np.isfinite(o.get("mean", np.nan)) or (n_group is not None and o.get("n", n_group) < n_group):
                    return
                if key.startswith("modma_") and "irasa_all_pair" in key:
                    return
            rows.append(dict(source=src, test=path, n=o.get("n"), estimate=o.get("mean"), n_positive=o.get("n_positive"),
                             p=float(o["p"])))
            return
        ng = o["n"] if scalp and isinstance(o.get("n"), int) and "p" not in o else n_group
        for k, v in o.items():
            walk(v, f"{path}/{k}" if path else k, rows, src, ng, scalp)


rows = []
for f in FAMILY + [e for e in CHECKS if (R / e).exists()]:
    walk(json.load(open(R / f)), "", rows, f, scalp=f.endswith("scalp_group.json"))
D = pd.DataFrame(rows)
m = len(D)
order = np.argsort(D.p.to_numpy())
q = np.empty(m)
q[order] = np.minimum.accumulate((D.p.to_numpy()[order] * m / np.arange(1, m + 1))[::-1])[::-1]
D["q_bh"] = np.minimum(q, 1.0)
D["survives_bh_5pct"] = D.q_bh <= 0.05
out = R / "revision" / "statistical_inventory.csv"
out.parent.mkdir(parents=True, exist_ok=True)
D.to_csv(out, index=False)
print(f"family {m}; p<.05 {(D.p < .05).sum()}; BH survivors {D.survives_bh_5pct.sum()}; sources:", D.source.value_counts().to_dict())
