"""Synthetic C-MAPSS-like data: engines degrade exponentially until failure, so the
whole pipeline can be tested offline without downloading the NASA dataset."""

import numpy as np
import pandas as pd
import pytest

from turbofan.data import SENSORS, SETTINGS, add_rul

DEGRADING = ["s2", "s3", "s4", "s7", "s9", "s11", "s12", "s14"]


def make_fleet(n_units=24, seed=0, truncate=False):
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_units + 1):
        life = int(rng.integers(120, 220))
        end = int(rng.integers(40, life - 10)) if truncate else life
        for cycle in range(1, end + 1):
            wear = np.exp((cycle - life) / 30.0)  # ~0 when healthy, ~1 at failure
            row = {"unit": unit, "cycle": cycle, "setting1": 0.0, "setting2": 0.0,
                   "setting3": 100.0}
            for i, s in enumerate(SENSORS):
                base = 100.0 + i
                drift = (3.0 * wear * (1 if i % 2 else -1)) if s in DEGRADING else 0.0
                noise = rng.normal(0, 0.3) if s not in ("s1", "s5", "s18") else 0.0
                row[s] = base + drift + noise
            rows.append(row)
    df = pd.DataFrame(rows)
    df[SETTINGS + SENSORS] = df[SETTINGS + SENSORS].astype(float)
    return df


@pytest.fixture(scope="session")
def fleet():
    return add_rul(make_fleet())


@pytest.fixture(scope="session")
def tmp_cmapss(tmp_path_factory):
    """A fake CMAPSSData folder for FD001 written in the original text format."""
    root = tmp_path_factory.mktemp("CMAPSSData")
    train, test = make_fleet(20, seed=1), make_fleet(8, seed=2, truncate=True)
    for name, df in (("train_FD001.txt", train), ("test_FD001.txt", test)):
        df.to_csv(root / name, sep=" ", header=False, index=False)
    np.savetxt(root / "RUL_FD001.txt", np.full(8, 50), fmt="%d")
    return root
