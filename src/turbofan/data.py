"""Load the NASA C-MAPSS turbofan datasets and build remaining-useful-life labels."""

from pathlib import Path

import numpy as np
import pandas as pd

DATASET_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
DEFAULT_ROOT = Path("data/raw/CMAPSSData")

SETTINGS = ["setting1", "setting2", "setting3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTINGS, *SENSORS]

# Table 2 of Saxena et al., "Damage Propagation Modeling for Aircraft Engine
# Run-to-Failure Simulation" (PHM 2008), in the column order of the data files.
SENSOR_INFO = {
    "s1": ("T2", "Total temperature at fan inlet"),
    "s2": ("T24", "Total temperature at LPC outlet"),
    "s3": ("T30", "Total temperature at HPC outlet"),
    "s4": ("T50", "Total temperature at LPT outlet"),
    "s5": ("P2", "Pressure at fan inlet"),
    "s6": ("P15", "Total pressure in bypass-duct"),
    "s7": ("P30", "Total pressure at HPC outlet"),
    "s8": ("Nf", "Physical fan speed"),
    "s9": ("Nc", "Physical core speed"),
    "s10": ("epr", "Engine pressure ratio (P50/P2)"),
    "s11": ("Ps30", "Static pressure at HPC outlet"),
    "s12": ("phi", "Ratio of fuel flow to Ps30"),
    "s13": ("NRf", "Corrected fan speed"),
    "s14": ("NRc", "Corrected core speed"),
    "s15": ("BPR", "Bypass ratio"),
    "s16": ("farB", "Burner fuel-air ratio"),
    "s17": ("htBleed", "Bleed enthalpy"),
    "s18": ("Nf_dmd", "Demanded fan speed"),
    "s19": ("PCNfR_dmd", "Demanded corrected fan speed"),
    "s20": ("W31", "HPT coolant bleed"),
    "s21": ("W32", "LPT coolant bleed"),
}

SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def read_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    df[["unit", "cycle"]] = df[["unit", "cycle"]].astype(int)
    df[SETTINGS + SENSORS] = df[SETTINGS + SENSORS].astype(float)
    return df


def add_rul(df: pd.DataFrame, cap: int | None = 125) -> pd.DataFrame:
    """RUL = cycles left before failure. Capped (piecewise-linear target), because
    early in life the engine is healthy and the exact distance to failure is not
    observable from the sensors."""
    out = df.copy()
    last = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = last - out["cycle"]
    if cap is not None:
        out["rul"] = out["rul"].clip(upper=cap)
    return out


def load_subset(subset: str, root: Path = DEFAULT_ROOT, cap: int | None = 125):
    """Return (train, test, test_rul) for one subset, e.g. "FD001".

    train: full run-to-failure trajectories with an `rul` column.
    test: trajectories truncated before failure.
    test_rul: true RUL at the last test cycle of each unit (official labels).
    """
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset!r}, expected one of {SUBSETS}")
    root = Path(root)
    if not (root / f"train_{subset}.txt").exists():
        raise FileNotFoundError(
            f"{root}/train_{subset}.txt not found. Run: python scripts/download_data.py"
        )
    train = add_rul(read_table(root / f"train_{subset}.txt"), cap=cap)
    test = read_table(root / f"test_{subset}.txt")
    test_rul = np.loadtxt(root / f"RUL_{subset}.txt", dtype=float)
    return train, test, test_rul


def split_units(df: pd.DataFrame, val_fraction: float = 0.2, seed: int = 0):
    """Split by engine, never by row, so no engine leaks into both sides."""
    units = df["unit"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_val = max(1, int(len(units) * val_fraction))
    val_units = set(units[:n_val].tolist())
    is_val = df["unit"].isin(val_units)
    return df[~is_val].copy(), df[is_val].copy()
