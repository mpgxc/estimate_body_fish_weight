import json

from experiments.report import build_comparison, build_tables


def _metrics(path, name, mae, reference_only=False, protocol="p3"):
    path.mkdir(parents=True)
    base = {"n": 10, "mae": mae, "rmse": mae, "mape": 5.0, "r2": 0.9, "bias": 0.0}
    (path / "metrics.json").write_text(json.dumps({
        "experiment": name, "protocol": protocol, "reference_only": reference_only,
        "overall": {**base, "mae_ci": [mae - 1, mae + 1], "mape_ci": [4.0, 6.0]},
        "per_fold": {}, "per_species_group": {"tambaqui": base}, "per_weight_band": {}}))


def test_tables_rank_by_mae_and_mark_reference_runs(tmp_path):
    _metrics(tmp_path / "pior/p3", "pior", 40.0)
    _metrics(tmp_path / "melhor/p3", "melhor", 20.0)
    _metrics(tmp_path / "oraculo/p3", "oraculo", 10.0, reference_only=True)
    text = build_tables(tmp_path)
    assert text.index("oraculo (ref.)") < text.index("melhor") < text.index("pior")
    assert "20,0 [19,0; 21,0]" in text
    assert "Por espécie: melhor" in text


def test_small_species_group_carries_a_warning(tmp_path):
    _metrics(tmp_path / "m/p3", "m", 20.0)
    path = tmp_path / "m/p3/metrics.json"
    data = json.loads(path.read_text())
    data["per_species_group"]["outras"] = data["per_species_group"]["tambaqui"]
    path.write_text(json.dumps(data))
    assert "amostra pequena" in build_tables(tmp_path)


def test_comparison_has_one_row_per_technique_ranked_by_p3(tmp_path):
    _metrics(tmp_path / "a/p2", "a", 25.0, protocol="p2")
    _metrics(tmp_path / "a/p3", "a", 30.0)
    _metrics(tmp_path / "b/p3", "b", 20.0)
    _metrics(tmp_path / "c/p1", "c", 10.0, reference_only=True, protocol="p1")
    rows = [line for line in build_comparison(tmp_path).splitlines() if line.startswith("| ")][1:]
    assert [row.split("|")[1].strip() for row in rows] == ["b", "a", "c (ref.)"]
    assert rows[1] == "| a | – | 25,0 | 30,0 [29,0; 31,0] | 5,0 | 0,900 | 0,0 |"
    assert rows[2] == "| c (ref.) | 10,0 | – | – | – | – | – |"
