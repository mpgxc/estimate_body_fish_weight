import numpy as np

from fishweight.regressors import make_models

NAMES = {"GradientBoostingRegressor", "LinearRegression", "LassoRegression", "RidgeRegression",
         "RandomForestRegressor", "PCARegression", "PolyRegression"}


def test_seven_models_fresh_on_every_call():
    first, second = make_models(), make_models()
    assert set(first) == NAMES
    assert all(first[name] is not second[name] for name in NAMES)


def test_models_fit_and_are_reproducible():
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 10, (60, 3))
    y = 5 * X[:, 0] + X[:, 1] ** 2 + rng.normal(0, 0.1, 60)
    for name in NAMES:
        a = make_models()[name].fit(X[:40], y[:40]).predict(X[40:])
        b = make_models()[name].fit(X[:40], y[:40]).predict(X[40:])
        assert np.allclose(a, b), name
        assert np.isfinite(a).all(), name
