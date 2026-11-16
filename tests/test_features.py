import numpy as np
import pandas as pd
import pytest
from skimage.io import imsave

from fishweight.data import load_manifest
from fishweight.features import (
    FEATURE_NAMES,
    build_feature_table,
    calibrate_features,
    load_mask,
    mask_features,
    session_scale,
    with_predicted_test_features,
)


def _write_mask(path, shape=(3472, 4624), boxes=((1000, 1800, 1000, 2600),)):
    full = np.zeros(shape, dtype=np.uint8)
    for r0, r1, c0, c1 in boxes:
        full[r0:r1, c0:c1] = 255
    imsave(path, full, check_contrast=False)
    return path


def test_rectangle_is_measured_at_one_eighth_scale(tmp_path):
    mask = load_mask(_write_mask(tmp_path / "m.png"))  # retângulo de 800×1600 px
    assert mask.shape == (434, 578)
    f = mask_features(mask)
    assert list(f) == FEATURE_NAMES
    assert f["area"] == 100 * 200
    assert f["area_bbox"] == 100 * 200
    assert f["extent"] == pytest.approx(1.0)
    assert f["solidity"] == pytest.approx(1.0)


def test_only_the_largest_component_is_measured(tmp_path):
    boxes = ((1000, 1800, 1000, 2600), (100, 180, 100, 180))  # peixe + mancha de 80×80
    f = mask_features(load_mask(_write_mask(tmp_path / "m.png", boxes=boxes)))
    assert f["area"] == 100 * 200


def test_wrong_resolution_is_rejected_with_file_and_size(tmp_path):
    path = _write_mask(tmp_path / "small.png", shape=(1080, 1920), boxes=((100, 200, 100, 300),))
    with pytest.raises(ValueError, match=r"small\.png.*1080.*1920"):
        load_mask(path)


def test_empty_mask_is_rejected():
    with pytest.raises(ValueError):
        mask_features(np.zeros((434, 578), dtype=bool))


def test_feature_table_covers_every_fish(tmp_path):
    table = build_feature_table(load_manifest(), cache_path=tmp_path / "manual.csv")
    assert len(table) == 244
    assert not table[FEATURE_NAMES].isna().any().any()
    assert (table["area"] > 0).all()
    cached = build_feature_table(load_manifest(), cache_path=tmp_path / "manual.csv")
    assert cached[FEATURE_NAMES].equals(table[FEATURE_NAMES])


def _frame(img, value, **extra):
    return pd.DataFrame({"img": [img], **{name: [value] for name in FEATURE_NAMES}, **extra})


PREDICTED = pd.concat([_frame("1", 10.0, fold=["a"]), _frame("1", 20.0, fold=["b"])]).set_index(["fold", "img"])


def test_predicted_features_replace_only_the_test_rows_of_that_fold():
    seen = {}

    def inner(train, test):
        seen.update(train=train["area"].tolist(), test=test["area"].tolist())
        return np.zeros(len(test))

    train, test = _frame("2", 1.0, weight=[5.0]), _frame("1", 1.0, fold=["b"])
    with_predicted_test_features(inner, PREDICTED)(train, test)
    assert seen == {"train": [1.0], "test": [20.0]}
    assert test["area"].tolist() == [1.0]  # a tabela original não é alterada


def test_missing_predicted_mask_is_rejected():
    wrapped = with_predicted_test_features(lambda train, test: np.zeros(len(test)), PREDICTED)
    with pytest.raises(ValueError, match="'c'.*'1'"):
        wrapped(_frame("2", 1.0, weight=[5.0]), _frame("1", 1.0, fold=["c"]))


def _scale_table():
    """Dois peixes idênticos de 20 cm fotografados a distâncias diferentes: 10 e 12 px/cm."""
    rows = []
    for session, scale in (("a", 10.0), ("b", 12.0)):
        for img in range(3):
            rows.append({
                "img": f"{session}{img}", "session_group": session, "length": 20.0,
                **{name: 1.0 for name in FEATURE_NAMES},
                "feret_diameter_max": 20.0 * scale, "perimeter": 50.0 * scale,
                "area": 100.0 * scale**2, "solidity": 0.9,
            })
    return pd.DataFrame(rows)


def test_calibration_makes_the_same_fish_measure_the_same_at_any_distance():
    table = _scale_table()
    scale = session_scale(table)
    assert scale.tolist() == pytest.approx([10.0] * 3 + [12.0] * 3)
    calibrated = calibrate_features(table, scale)
    assert calibrated["area"].tolist() == pytest.approx([100.0] * 6)              # cm²
    assert calibrated["feret_diameter_max"].tolist() == pytest.approx([20.0] * 6)  # cm
    assert calibrated["perimeter"].tolist() == pytest.approx([50.0] * 6)
    assert calibrated["solidity"].tolist() == pytest.approx([0.9] * 6)            # razões não mudam
    assert table["area"].iloc[0] == 100.0 * 10.0**2                                # original intacta


def test_session_scale_ignores_unreliable_tape_measures():
    from fishweight.data import UNRELIABLE_MEASURES

    table = _scale_table()
    table.loc[0, "img"] = next(iter(UNRELIABLE_MEASURES))
    table.loc[0, "length"] = 2.0       # comprimento absurdo: daria 100 px/cm
    table.loc[1, "length"] = 25.0      # um erro de fita comum: 8 px/cm; a mediana segura
    assert session_scale(table).iloc[:3].tolist() == pytest.approx([9.0] * 3)  # mediana de 8 e 10
