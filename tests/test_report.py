import json

import pandas as pd

from experiments.report import (
    build_comparison,
    build_cross_domain,
    build_segmentation,
    build_sensitivity,
    build_tables,
)


def _metrics(path, name, mae, reference_only=False, protocol="p3", mask="manual", **extra):
    path.mkdir(parents=True)
    base = {"n": 10, "mae": mae, "rmse": mae, "mape": 5.0, "r2": 0.9, "bias": 0.0}
    (path / "metrics.json").write_text(json.dumps({
        "experiment": name, "protocol": protocol, "reference_only": reference_only, "mask": mask,
        "overall": {**base, "mae_ci": [mae - 1, mae + 1], "mape_ci": [4.0, 6.0]},
        "per_fold": {}, "per_species_group": {"tambaqui": base}, "per_weight_band": {}, **extra}))


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
    _metrics(tmp_path / "d/p3", "d", 35.0, mask="predita")
    rows = [line for line in build_comparison(tmp_path).splitlines() if line.startswith("| ")][1:]
    assert [row.split("|")[1].strip() for row in rows] == ["b", "a", "d", "c (ref.)"]
    assert rows[1] == "| a | manual | – | 25,0 | 30,0 [29,0; 31,0] | 5,0 | 0,900 | 0,0 |"
    assert rows[2] == "| d | predita | – | – | 35,0 [34,0; 36,0] | 5,0 | 0,900 | 0,0 |"
    assert rows[3] == "| c (ref.) | manual | 10,0 | – | – | – | – | – |"


def test_breakdown_only_considers_runs_the_tool_can_reproduce(tmp_path):
    """A ferramenta não terá a máscara manual: concorrem a máscara prevista e os modelos só de foto."""
    _metrics(tmp_path / "manual/p3", "manual", 20.0)
    _metrics(tmp_path / "automatica/p3", "automatica", 30.0, mask="predita")
    _metrics(tmp_path / "oraculo/p3", "oraculo", 10.0, reference_only=True, mask="predita")
    assert "Por espécie: automatica" in build_tables(tmp_path)
    _metrics(tmp_path / "so_foto/p3", "so_foto", 25.0, mask="nenhuma")
    assert "Por espécie: so_foto" in build_tables(tmp_path)


def test_segmentation_summary_per_protocol(tmp_path):
    folder = tmp_path / "segmentation/p3"
    folder.mkdir(parents=True)
    pd.DataFrame({"img": ["a", "b"], "fold": ["x", "y"], "dice": [0.9, 1.0], "iou": [0.8, 1.0]}).to_csv(
        folder / "metrics.csv", index=False)
    rows = [line for line in build_segmentation(tmp_path).splitlines() if line.startswith("| ")][1:]
    assert rows == ["| P3 | 2 | 0,950 | 0,900 | 0,900 |"]


def test_cross_domain_runs_get_their_own_table(tmp_path):
    """Treinar num conjunto e testar no outro não cabe nas colunas P1–P3."""
    _metrics(tmp_path / "a/p3", "a", 30.0)
    _metrics(tmp_path / "f2/cruzado", "f2", 50.0, protocol="cruzado", mask="nenhuma", train="coleta", test="fdwe")
    assert "f2" not in build_comparison(tmp_path)
    rows = [line for line in build_cross_domain(tmp_path).splitlines() if line.startswith("| ")][1:]
    assert rows == ["| f2 | coleta | fdwe | 10 | 50,0 [49,0; 51,0] | 5,0 | 0,900 | 0,0 |"]


def test_sensitivity_shows_the_candidate_with_and_without_suspect_weights(tmp_path):
    """O curimatã de 1.760 g fica no conjunto; a tabela mostra quanto ele pesa no resultado."""
    _metrics(tmp_path / "modelo/p3", "modelo", 30.0, mask="predita")
    pd.DataFrame({
        "img": ["a.jpg", "b.jpg", "IMG_20220110_152533.jpg"],
        "y_true": [100.0, 200.0, 1760.0], "y_pred": [110.0, 190.0, 1060.0],
    }).to_csv(tmp_path / "modelo/p3/predictions.csv", index=False)
    rows = [line for line in build_sensitivity(tmp_path).splitlines() if line.startswith("| ")][1:]
    assert rows[0].startswith("| modelo | todos | 3 | 240,0 |")
    assert rows[1].startswith("| modelo | sem pesos suspeitos | 2 | 10,0 |")
