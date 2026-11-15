import numpy as np
import pytest
from skimage.io import imsave

from fishweight.data import load_manifest
from fishweight.features import FEATURE_NAMES, build_feature_table, load_mask, mask_features


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
