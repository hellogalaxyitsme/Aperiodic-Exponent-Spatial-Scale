"""IRASA sleep stage x derivation interaction on the matched channels (Results 3.4; S4).

Recomputes the IRASA interaction (2-20 Hz) on all interior channels, on the matched channels defined by
scalp_anphy_sleep_matched_channels.py, and on the remaining interior channels. Descriptive. Needs the scalp unit cache
written by scalp_derivations.py (measure).
Usage: python analysis/scalp_anphy_sleep_matched_channels_irasa.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apex import pairs  # noqa: E402
from apex.paths import CACHE  # noqa: E402

SCALP = Path(f"{CACHE}/final_round/scalp")
RES = ROOT / "results" / "revision"
_, CEN = pairs.log_bands(2.0, 20.0, 30)
MIN_CH = 5


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, x.size, size=(n, x.size))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def chis(sub, cond):
    z = np.load(SCALP / f"anphy_{sub}_{cond}.npz", allow_pickle=True)
    return pairs.band_exponent(CEN, z["Ia"], (2.0, 20.0)), pairs.band_exponent(CEN, z["Il"], (2.0, 20.0)), z["interior"]


def main():
    C = pd.read_csv(RES / "common_scalp_channels.csv.gz")
    C = C[C.dataset == "anphy"]
    W = C.pivot_table(index=["sub", "ch"], columns=["cond", "der"], values="F_band")
    J, rows = {}, []
    for S in ("N2", "N3", "R"):
        v_all, v_four, v_rest, only_lw = [], [], [], []
        for sub, g in W.groupby(level=0):
            if (S, "avg") not in g.columns or g[(S, "avg")].isna().all() or not (SCALP / f"anphy_{sub}_{S}.npz").exists():
                continue
            cells = g[[("W", "avg"), ("W", "lap"), (S, "avg"), (S, "lap")]].notna()
            ok = cells.all(axis=1).to_numpy()
            ch = g.index.get_level_values(1).to_numpy()
            chans, rest = ch[ok], ch[~ok]
            if len(chans) < MIN_CH:
                continue
            aw, lw, inner = chis(sub, "W")
            as_, ls, _ = chis(sub, S)
            d = (as_ - aw) - (ls - lw)
            v_all.append(float(d[inner].mean()))
            v_four.append(float(d[chans].mean()))
            r = dict(stage=S, sub=sub, irasa_interaction_all_interior=v_all[-1], irasa_interaction_four_cell=v_four[-1],
                     n_ch=len(chans), n_rest=len(rest))
            if len(rest) >= MIN_CH:                                    # complement set
                v_rest.append(float(d[rest].mean()))
                r["irasa_interaction_excluded"] = v_rest[-1]
            fail = ~cells.to_numpy()[~ok]
            only_lw += list(fail[:, 1] & ~fail[:, [0, 2, 3]].any(axis=1))
            rows.append(r)
        J[f"{S}minusW"] = {k: dict(mean=float(np.mean(v)), ci95=boot(v), n_positive=int((np.asarray(v) > 0).sum()), n=len(v))
                           for k, v in (("all_interior", v_all), ("four_cell", v_four), ("excluded", v_rest))}
        J[f"{S}minusW"]["excluded_fail_only_lap_wake_fraction"] = float(np.mean(only_lw))
    (RES / "sleep_common_irasa.json").write_text(json.dumps(J, indent=1), encoding="utf-8")
    pd.DataFrame(rows).to_csv(RES / "sleep_common_irasa_units.csv", index=False)
    print(json.dumps(J, indent=1))


if __name__ == "__main__":
    main()
