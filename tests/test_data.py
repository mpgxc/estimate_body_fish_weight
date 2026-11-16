import pandas as pd
import pytest

from fishweight.data import EXCLUDED_IMAGES, load_manifest


def test_manifest_counts():
    m = load_manifest()
    assert len(m) == 244
    assert not set(EXCLUDED_IMAGES) & set(m["img"])
    assert m["species_group"].value_counts().to_dict() == {
        "tambaqui": 148, "tilápia": 61, "curimatã": 20, "outras": 15}
    assert m["session_group"].value_counts().to_dict() == {
        "2022-11-29": 79, "2022-11-26": 65, "2022-11-22": 48, "2022-10-29": 25, "2022-inicio": 27}
    assert m["img"].is_monotonic_increasing and m["weight"].min() == 50


def test_manifest_works_from_another_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert len(load_manifest()) == 244


def _dataset(tmp_path, rows):
    (tmp_path / "images").mkdir(); (tmp_path / "masks").mkdir()
    for img, *_ in rows:
        (tmp_path / "images" / img).touch(); (tmp_path / "masks" / img).touch()
    pd.DataFrame(rows, columns=["img", "specie", "weight", "width", "length"]).to_csv(
        tmp_path / "annotations.csv", index=False)
    return tmp_path


def test_duplicate_image_is_rejected(tmp_path):
    base = _dataset(tmp_path, [("IMG_20221029_000001.jpg", "panga", 100, 5, 20)] * 2)
    with pytest.raises(ValueError, match="IMG_20221029_000001.jpg"):
        load_manifest(base)


def test_missing_mask_is_rejected(tmp_path):
    base = _dataset(tmp_path, [("IMG_20221029_000001.jpg", "panga", 100, 5, 20)])
    (base / "masks" / "IMG_20221029_000001.jpg").unlink()
    with pytest.raises(ValueError, match="IMG_20221029_000001.jpg"):
        load_manifest(base)


def test_non_positive_weight_is_rejected(tmp_path):
    base = _dataset(tmp_path, [("IMG_20221029_000001.jpg", "panga", 0, 5, 20)])
    with pytest.raises(ValueError, match="IMG_20221029_000001.jpg"):
        load_manifest(base)
