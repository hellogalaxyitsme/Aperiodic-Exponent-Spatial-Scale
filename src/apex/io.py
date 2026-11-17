"""Data access for ds004752 (read-only).

Layout: sub-XX/ses-YY/{ieeg,eeg}/sub-XX_ses-YY_task-verbalWM_run-01_{ieeg,eeg}.edf
Each EDF is a concatenation of 8-s trials; events.tsv gives begSample/endSample (1-based, inclusive) and an Artifact flag.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from .paths import CACHE, DATA
import pandas as pd

DS004752 = Path(os.environ.get("APEX_DS004752", f"{DATA}/ds004752"))

# Brainnetome gyral abbreviation (first token of AnatomicalLocation) -> structure class used for pairing.
STRUCTURE_CLASS = {
    "Hipp": "mesial", "Amyg": "mesial", "PhG": "mesial",
    "STG": "lateral_temporal", "MTG": "lateral_temporal", "ITG": "lateral_temporal", "pSTS": "lateral_temporal",
    "FuG": "basal_temporal",
}
WHITE = "white_or_unlabelled"


def structure_class(label: str) -> str:
    if not isinstance(label, str) or label.strip() == "" or label.startswith("no_label"):
        return WHITE
    tok = label.split(",")[0].strip()
    return STRUCTURE_CLASS.get(tok, "other_cortex")


def gyrus(label: str) -> str:
    if not isinstance(label, str) or label.startswith("no_label"):
        return "none"
    return label.split(",")[0].strip()


_GRID = re.compile(r"^(?P<shaft>.*_G[A-Z]*_)(?P<row>[A-Z])(?P<col>\d+)$")
_SHAFT = re.compile(r"^(?P<shaft>.*?)(?P<idx>\d+)$")


def parse_contact(name: str) -> dict:
    """Split a contact name into shaft/strip id and position.

    Depth:  'mAHL3'  -> shaft 'mAHL', idx 3, kind 'depth'
    Grid:   '_GL_B4' -> shaft '_GL_', row 'B', col 4, kind 'grid'
    """
    m = _GRID.match(name)
    if m:
        return dict(shaft=m["shaft"], row=ord(m["row"]) - ord("A"), idx=int(m["col"]), kind="grid")
    m = _SHAFT.match(name)
    if m:
        return dict(shaft=m["shaft"], row=-1, idx=int(m["idx"]), kind="depth")
    return dict(shaft=name, row=-1, idx=-1, kind="unknown")


@dataclass
class Session:
    sub: str
    ses: str
    root: Path = DS004752

    @property
    def stem(self) -> str:
        return f"{self.sub}_{self.ses}_task-verbalWM_run-01"

    def path(self, mod: str, suffix: str) -> Path:
        return self.root / self.sub / self.ses / mod / f"{self.stem}_{suffix}"

    def events(self, mod: str = "ieeg") -> pd.DataFrame:
        return pd.read_csv(self.path(mod, "events.tsv"), sep="\t")

    def electrodes(self) -> pd.DataFrame:
        el = pd.read_csv(self.path("ieeg", "electrodes.tsv"), sep="\t")
        parsed = pd.DataFrame([parse_contact(n) for n in el["name"]])
        el = pd.concat([el, parsed], axis=1)
        el["gyrus"] = el["AnatomicalLocation"].map(gyrus)
        el["sclass"] = el["AnatomicalLocation"].map(structure_class)
        return el

    def load(self, mod: str = "ieeg", drop_artifacts: bool = True, picks=None):
        """Return (epochs [n_trials, n_ch, n_samp] in volts, ch_names, sfreq, events_kept)."""
        import mne
        src = self.path(mod, f"{mod}.edf")
        try:
            raw = mne.io.read_raw_edf(src, preload=True, verbose="ERROR")
        except ValueError as e:
            if "must be in" not in str(e):
                raise
            raw = mne.io.read_raw_edf(_patched_edf_copy(src), preload=True, verbose="ERROR")
        if picks is not None:
            raw.pick(picks)
        X = raw.get_data()
        sf = raw.info["sfreq"]
        ev = self.events(mod)
        if drop_artifacts and "Artifact" in ev:
            ev = ev[ev["Artifact"] == 0]
        n = int(round(8.0 * sf))
        segs, keep = [], []
        for i, r in ev.iterrows():
            b = int(r["begSample"]) - 1
            if b < 0 or b + n > X.shape[1]:
                continue
            segs.append(X[:, b:b + n])
            keep.append(i)
        return np.stack(segs), list(raw.ch_names), sf, ev.loc[keep].reset_index(drop=True)


PATCH_DIR = Path(os.environ.get("APEX_CACHE", f"{CACHE}")) / "edf_header_patched"


def _patched_edf_copy(src: Path) -> Path:
    """Copy of an EDF whose start date/time header fields (bytes 168-183) are invalid, with them set to
    01.01.00 / 00.00.00. Signal bytes are untouched; the raw dataset is never modified."""
    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    dst = PATCH_DIR / src.name
    if not dst.exists():
        b = bytearray(src.read_bytes())
        b[168:184] = b"01.01.0000.00.00"
        tmp = dst.with_suffix(".tmp")
        tmp.write_bytes(bytes(b))
        tmp.rename(dst)
    return dst


def subjects(root: Path = DS004752) -> list[str]:
    return sorted(p.name for p in root.glob("sub-*") if p.is_dir())


def sessions(sub: str, root: Path = DS004752) -> list[Session]:
    return [Session(sub, p.name, root) for p in sorted((root / sub).glob("ses-*")) if p.is_dir()]


@dataclass
class Run:
    """Consecutive contacts on one shaft sharing a structure class (white matter excluded)."""
    sub: str
    shaft: str
    sclass: str
    contacts: list = field(default_factory=list)   # contact names in order
    xyz: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.contacts)


def build_runs(el: pd.DataFrame, sub: str, min_len: int = 2) -> list[Run]:
    """Depth shafts only (grids handled separately)."""
    runs = []
    for shaft, g in el[el["kind"] == "depth"].groupby("shaft"):
        g = g.sort_values("idx")
        cur = None
        prev_idx = None
        for _, r in g.iterrows():
            ok = r["sclass"] != WHITE
            contiguous = prev_idx is not None and r["idx"] == prev_idx + 1
            if ok and cur is not None and cur.sclass == r["sclass"] and contiguous:
                cur.contacts.append(r["name"])
            else:
                if cur is not None and cur.n >= min_len:
                    runs.append(cur)
                cur = Run(sub, shaft, r["sclass"], [r["name"]]) if ok else None
            prev_idx = r["idx"]
        if cur is not None and cur.n >= min_len:
            runs.append(cur)
    xyz = el.set_index("name")[["x", "y", "z"]]
    for r in runs:
        r.xyz = xyz.loc[r.contacts].to_numpy(float)
    return runs


def run_pairs(run: Run) -> list[tuple[str, str, int, float]]:
    """All within-run pairs (a, b, lag, distance_mm)."""
    out = []
    for i in range(run.n):
        for j in range(i + 1, run.n):
            out.append((run.contacts[i], run.contacts[j], j - i, float(np.linalg.norm(run.xyz[j] - run.xyz[i]))))
    return out
