"""Contact inventory for all ds004752 participants (metadata only; no spectra).

Lists every depth contact and session with its shaft, index and anatomical class, and the runs that can be paired.
Outputs: contacts.csv, runs.csv, summary.txt.
Usage: python analysis/depth_contact_inventory.py --out results/inventory
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from apex import io  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/inventory")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

rows, runrows = [], []
for sub in io.subjects():
    for s in io.sessions(sub):
        try:
            el = s.electrodes()
        except FileNotFoundError:
            continue
        el.insert(0, "ses", s.ses)
        el.insert(0, "sub", sub)
        rows.append(el)
        for r in io.build_runs(el, sub):
            d = np.linalg.norm(np.diff(r.xyz, axis=0), axis=1)
            runrows.append(dict(sub=sub, ses=s.ses, shaft=r.shaft, sclass=r.sclass, n=r.n,
                                contacts="|".join(r.contacts), max_lag=r.n - 1,
                                spacing_mm_mean=float(d.mean()), spacing_mm_sd=float(d.std())))
C = pd.concat(rows, ignore_index=True)
R = pd.DataFrame(runrows)
C.to_csv(out / "contacts.csv", index=False)
R.to_csv(out / "runs.csv", index=False)

# electrode layout constant across sessions?
lay = C.groupby(["sub", "ses"])["name"].apply(lambda s: "|".join(sorted(s)))
const = lay.groupby(level=0).nunique()

lines = []
lines.append("Per-subject inventory (session ses-01 layout; 'const' = identical contact list in all sessions)\n")
for sub in sorted(C["sub"].unique()):
    c1 = C[(C["sub"] == sub) & (C["ses"] == C[C["sub"] == sub]["ses"].min())]
    r1 = R[(R["sub"] == sub) & (R["ses"] == c1["ses"].iloc[0])]
    kinds = c1["kind"].value_counts().to_dict()
    cls = c1["sclass"].value_counts().to_dict()
    shafts = sorted(c1["shaft"].unique())
    lag2 = r1[r1["max_lag"] >= 2].groupby("sclass").size().to_dict()
    lag3 = r1[r1["max_lag"] >= 3].groupby("sclass").size().to_dict()
    nses = C[C["sub"] == sub]["ses"].nunique()
    lines.append(f"{sub}: sessions={nses} const_layout={const[sub] == 1} kinds={kinds}")
    lines.append(f"   shafts={shafts}")
    lines.append(f"   classes={cls}")
    lines.append(f"   runs with lag>=2 by class={lag2}; lag>=3={lag3}")
    if len(r1):
        lines.append(f"   adjacent spacing mm: mean {r1['spacing_mm_mean'].mean():.2f}")
txt = "\n".join(lines)
(out / "summary.txt").write_text(txt)
print(txt)
