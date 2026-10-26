import numpy as np

from turbofan.data import add_rul, load_subset, split_units
from turbofan.metrics import nasa_score, rmse
from turbofan.preprocess import Normalizer
from turbofan.windows import last_windows, training_windows, window_features


def test_rul_counts_down_to_zero_and_is_capped(fleet):
    unit = fleet[fleet["unit"] == 1]
    assert unit["rul"].iloc[-1] == 0
    assert unit["rul"].max() == 125
    raw = add_rul(fleet.drop(columns="rul"), cap=None)
    assert raw[raw["unit"] == 1]["rul"].iloc[0] == len(unit) - 1


def test_split_is_by_engine(fleet):
    train, val = split_units(fleet, 0.25, seed=0)
    assert not set(train["unit"]) & set(val["unit"])
    assert len(set(val["unit"])) == 6


def test_normalizer_drops_constant_sensors_and_standardizes(fleet):
    norm = Normalizer().fit(fleet)
    assert {"s1", "s5", "s18"}.isdisjoint(norm.sensors)
    out = norm.transform(fleet)
    assert np.allclose(out[norm.sensors].mean(), 0, atol=1e-6)
    assert np.allclose(out[norm.sensors].std(ddof=0), 1, atol=1e-6)


def test_windows_shapes_and_padding(fleet):
    sensors = ["s2", "s3"]
    x, y, units = training_windows(fleet, sensors, length=30)
    assert x.shape[1:] == (30, 2) and len(x) == len(y) == len(units)
    short = fleet[(fleet["unit"] == 1) & (fleet["cycle"] <= 10)]
    xl, _ = last_windows(short, sensors, length=30)
    assert xl.shape == (1, 30, 2)
    assert np.allclose(xl[0, :20], xl[0, 20])  # left-padded with the first cycle
    assert window_features(x[:5]).shape == (5, 8)


def test_metrics():
    assert rmse([10, 20], [10, 20]) == 0
    assert rmse([0, 0], [3, 4]) == np.sqrt(12.5)
    # being 10 cycles late costs more than being 10 cycles early
    assert nasa_score([50], [60]) > nasa_score([50], [40])
    assert np.isclose(nasa_score([50], [60]), np.e - 1)


def test_load_subset_reads_original_format(tmp_cmapss):
    train, test, rul = load_subset("FD001", tmp_cmapss)
    assert "rul" in train and len(rul) == test["unit"].nunique() == 8
