"""Locations of the raw datasets and of the working cache, set by environment variables.

APEX_DATA    folder containing the datasets (default: ./data); expected layout in README.md
APEX_WORK    working folder for intermediate caches (default: ./work); caches are written to $APEX_WORK/cache
APEX_NYHEAD  New York Head model file sa_nyhead.mat (default: $APEX_DATA/nyhead/sa_nyhead.mat)
"""
import os
from pathlib import Path

DATA = Path(os.environ.get("APEX_DATA", "data"))
CACHE = Path(os.environ.get("APEX_WORK", "work")) / "cache"
NYHEAD = Path(os.environ.get("APEX_NYHEAD", str(DATA / "nyhead" / "sa_nyhead.mat")))
