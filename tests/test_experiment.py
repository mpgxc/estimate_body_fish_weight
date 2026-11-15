import json

import numpy as np
import pandas as pd
import pytest

from fishweight import experiment
from fishweight.experiment import run_experiment

FOLDS = [{"name": "a", "train": ["3", "4", "5", "6"], "test": ["1", "2"]},
         {"name": "b", "train": ["1", "2", "5", "6"], "test": ["3", "4"]},
         {"name": "c", "train": ["1", "2", "3", "4"], "test": ["5", "6"]}]


@pytest.fixture
def table(monkeypatch):
    monkeypatch.setattr(experiment, "load_splits", lambda protocol: FOLDS)
    return pd.DataFrame({
        "img": list("123456"), "species": "tambaqui", "species_group": "tambaqui",
        "session_group": list("aabbcc"), "weight": [100.0, 200, 300, 400, 500, 600]})


def train_mean(train, test):
    return np.full(len(test), train["weight"].mean())


def test_writes_predictions_and_metrics(table, tmp_path):
    metrics = run_experiment("fake", "p3", table, train_mean, results_dir=tmp_path)
    preds = pd.read_csv(tmp_path / "fake/p3/predictions.csv")
    assert list(preds.columns) == ["img", "fold", "species", "species_group", "session_group", "y_true", "y_pred"]
    assert len(preds) == 6
    assert preds.loc[preds["fold"] == "a", "y_pred"].tolist() == [450.0, 450.0]
    assert metrics == json.loads((tmp_path / "fake/p3/metrics.json").read_text())
    assert metrics["overall"]["n"] == 6
    assert set(metrics["per_fold"]) == {"a", "b", "c"}
    assert set(metrics["per_weight_band"]) == {"baixo", "medio", "alto"}
    assert metrics["overall"]["mae_ci"][0] <= metrics["overall"]["mae"] <= metrics["overall"]["mae_ci"][1]


def test_model_cannot_see_test_weights(table, tmp_path):
    def cheats(train, test):
        return test["weight"].to_numpy()
    with pytest.raises(KeyError):
        run_experiment("cheat", "p3", table, cheats, results_dir=tmp_path)


def test_stale_split_is_rejected(table, tmp_path):
    with pytest.raises(ValueError, match="6"):
        run_experiment("stale", "p3", table[table["img"] != "6"], train_mean, results_dir=tmp_path)


def test_wrong_number_of_predictions_is_rejected(table, tmp_path):
    with pytest.raises(ValueError, match="dobra a:"):
        run_experiment("short", "p3", table, lambda train, test: np.zeros(1), results_dir=tmp_path)
