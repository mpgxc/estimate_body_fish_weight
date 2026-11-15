import numpy as np
import pandas as pd
import pytest

from fishweight.metrics import bootstrap_ci, regression_metrics, weight_band

Y_TRUE, Y_PRED = [100.0, 200.0, 400.0], [110.0, 180.0, 400.0]


def test_metrics_exact_values():
    m = regression_metrics(Y_TRUE, Y_PRED)
    assert m["n"] == 3
    assert m["mae"] == pytest.approx(10.0)
    assert m["rmse"] == pytest.approx(12.909944)
    assert m["mape"] == pytest.approx(6.666667)
    assert m["bias"] == pytest.approx(-3.333333)
    assert m["r2"] == pytest.approx(0.9892857)


def test_non_positive_true_weight_is_rejected():
    with pytest.raises(ValueError):
        regression_metrics([0.0, 100.0], [1.0, 100.0])


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        regression_metrics([100.0, 200.0], [100.0])


def test_bootstrap_ci_brackets_the_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    y = rng.uniform(100, 1000, 200)
    p = y + rng.normal(0, 30, 200)
    lo, hi = bootstrap_ci(y, p, "mae")
    assert lo < regression_metrics(y, p)["mae"] < hi
    assert (lo, hi) == bootstrap_ci(y, p, "mae")


def test_bootstrap_ci_by_group_ignores_repeated_rows():
    rng = np.random.default_rng(2)
    y = rng.uniform(100, 1000, 50)
    p = y + rng.normal(0, 30, 50)
    fish = np.arange(50)
    single = bootstrap_ci(y, p, "mae", groups=fish)
    tripled = bootstrap_ci(np.tile(y, 3), np.tile(p, 3), "mae", groups=np.tile(fish, 3))
    assert tripled == pytest.approx(single)


def test_weight_band_splits_into_terciles():
    bands = weight_band(pd.Series(range(1, 10)))
    assert bands.tolist() == ["baixo"] * 3 + ["medio"] * 3 + ["alto"] * 3
