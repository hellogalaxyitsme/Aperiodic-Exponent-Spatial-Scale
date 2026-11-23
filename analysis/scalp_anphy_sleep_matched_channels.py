"""Sleep stage x derivation interaction on matched channels, specparam fixed mode (Results 3.4; S4).

For each participant and stage, uses only the interior channels whose specparam fixed-mode fit (2-20 Hz, R^2 >= 0.8)
was retained under both derivations in both wakefulness and the stage, and recomputes the stage effects and their
difference on exactly those channels. Also reproduces the unmatched values as a check. Descriptive: mean, bootstrap
95% CI, number positive; no p-values. Reads results/revision/common_scalp_channels.csv.gz
(spectral_specification_checks.py common).
Usage: python analysis/scalp_anphy_sleep_matched_channels.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision"
MIN_CH = 5


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, x.size, size=(n, x.size))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summary(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), ci95=boot(v), n_positive=int((v > 0).sum()), n=int(v.size))


def main():
    C = pd.read_csv(OUT / "common_scalp_channels.csv.gz")
    C = C[C.dataset == "anphy"]
    W = C.pivot_table(index=["sub", "ch"], columns=["cond", "der"], values="F_band")
    J, rows = {}, []
    for S in ("N2", "N3", "R"):
        rep, four = [], []
        for sub, g in W.groupby(level=0):
            if (S, "avg") not in g.columns or g[(S, "avg")].isna().all():
                continue
            aw, lw, as_, ls = g[("W", "avg")], g[("W", "lap")], g[(S, "avg")], g[(S, "lap")]
            okw, oks = aw.notna() & lw.notna(), as_.notna() & ls.notna()       # E028 rule: per state
            if okw.sum() >= MIN_CH and oks.sum() >= MIN_CH:
                rep.append((as_[oks] - ls[oks]).mean() - (aw[okw] - lw[okw]).mean())
            ok = okw & oks                                                       # four-cell rule
            n_ok = int(ok.sum())
            if n_ok >= MIN_CH:
                a, l_ = (as_ - aw)[ok].mean(), (ls - lw)[ok].mean()
                four.append(dict(sub=sub, avg=a, lap=l_, interaction=a - l_, n_ch=n_ok,
                                 n_ch_wake_pair=int(okw.sum()), n_ch_stage_pair=int(oks.sum()), n_interior=int(len(g))))
            else:
                four.append(dict(sub=sub, avg=np.nan, lap=np.nan, interaction=np.nan, n_ch=n_ok,
                                 n_ch_wake_pair=int(okw.sum()), n_ch_stage_pair=int(oks.sum()), n_interior=int(len(g))))
        F = pd.DataFrame(four)
        Fk = F.dropna(subset=["interaction"])
        J[f"{S}minusW"] = dict(
            reproduction_E028_rule=summary(rep),
            four_cell=dict(avg=summary(Fk.avg), lap=summary(Fk.lap), avg_minus_lap=summary(Fk.interaction),
                           channels_median=float(Fk.n_ch.median()), channels_range=[int(Fk.n_ch.min()), int(Fk.n_ch.max())],
                           interior_median=float(F.n_interior.median()),
                           excluded_lt5=int(F.interaction.isna().sum())))
        rows += [dict(stage=S, **r) for r in four]
    (OUT / "sleep_common_channels.json").write_text(json.dumps(J, indent=1), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "sleep_common_channels_units.csv", index=False)
    print(json.dumps(J, indent=1))


if __name__ == "__main__":
    main()
