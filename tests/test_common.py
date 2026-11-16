import json

import numpy as np
import pandas as pd
import pytest

from experiments.common import best_finetuned, run_once, save_cross_domain
from fishweight import experiment

FOLDS = [{"name": "a", "train": ["3", "4"], "test": ["1", "2"]},
         {"name": "b", "train": ["1", "2"], "test": ["3", "4"]}]
TABLE = pd.DataFrame({
    "img": list("1234"), "species": "tambaqui", "species_group": "tambaqui",
    "session_group": list("aabb"), "weight": [100.0, 200, 300, 400]})


def test_run_once_does_not_retrain_a_finished_run(tmp_path, monkeypatch):
    """Sessões de GPU caem; rodar o script de novo não pode repetir horas de treino."""
    monkeypatch.setattr(experiment, "load_splits", lambda protocol: FOLDS)
    calls = []

    def fit_predict(train, test):
        calls.append(len(test))
        return np.full(len(test), train["weight"].mean())

    run_once("deep", "p3", TABLE, fit_predict, results_dir=tmp_path)
    run_once("deep", "p3", TABLE, fit_predict, results_dir=tmp_path)
    assert calls == [2, 2]


def _result(results_dir, name, mae):
    folder = results_dir / name / "p3"
    folder.mkdir(parents=True)
    (folder / "metrics.json").write_text(json.dumps({"overall": {"mae": mae}}))


def test_best_finetuned_picks_the_lowest_p3_error_across_targets(tmp_path):
    _result(tmp_path, "e2_resnet50_ajuste", 40.0)
    _result(tmp_path, "e2_convnext_tiny_ajuste", 30.0)
    _result(tmp_path, "e2_convnext_tiny_ajuste_log", 20.0)
    _result(tmp_path, "e2_dinov2_vits14_congelado_ridge", 10.0)  # não é ajuste fino: fica de fora
    assert best_finetuned(tmp_path) == ("convnext_tiny", True)


def test_best_finetuned_explains_what_to_run_first(tmp_path):
    with pytest.raises(ValueError, match="e2_direct"):
        best_finetuned(tmp_path)


def test_cross_domain_result_records_both_datasets(tmp_path):
    metrics = save_cross_domain(
        "f2_x", ["a", "b", "c"], [100.0, 200.0, 400.0], [110.0, 180.0, 400.0],
        train="coleta", test="fdwe", results_dir=tmp_path)
    stored = json.loads((tmp_path / "f2_x/cruzado/metrics.json").read_text())
    assert stored == metrics
    assert (stored["protocol"], stored["train"], stored["test"], stored["mask"]) == ("cruzado", "coleta", "fdwe", "nenhuma")
    assert stored["overall"]["mae"] == pytest.approx(10.0) and len(stored["overall"]["mae_ci"]) == 2
    predictions = pd.read_csv(tmp_path / "f2_x/cruzado/predictions.csv")
    assert predictions.to_dict("list") == {"id": ["a", "b", "c"], "y_true": [100.0, 200.0, 400.0], "y_pred": [110.0, 180.0, 400.0]}
