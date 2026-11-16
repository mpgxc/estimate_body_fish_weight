import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from fishweight.fdwe import (
    FDWE_DIR,
    SPECIES,
    ZIP_NAME,
    crop_fish,
    load_fdwe_crops,
    load_fdwe_manifest,
    parse_labels,
)

REAL_ZIP = FDWE_DIR / ZIP_NAME
needs_dataset = pytest.mark.skipif(not REAL_ZIP.is_file(), reason="FDWE não baixado em data/external/fdwe")


def test_label_line_becomes_one_fish_in_grams():
    text = "0 0.470313 0.66875 0.157812 0.179688 0.091 31_60\n3 0.5 0.5 0.2 0.1 1.878 00_00\n"
    first, second = parse_labels(text, "2021_02_25_den_helder_1000_rgb")
    assert first == {"species": "plaice", "cx": 0.470313, "cy": 0.66875, "w": 0.157812,
                     "h": 0.179688, "weight": pytest.approx(91.0), "occlusion": 60}
    assert (second["species"], second["weight"], second["occlusion"]) == ("grey_gurnard", pytest.approx(1878.0), 0)


@pytest.mark.parametrize("line", ["0 0.5 0.5 0.2 0.1 0.091", "0 0.5 0.5 0.2 0.1 0.091 99_99", "9 0.5 0.5 0.2 0.1 0.091 00_00"])
def test_malformed_label_names_the_image(line):
    with pytest.raises(ValueError, match="foto_7"):
        parse_labels(line, "foto_7")


def _box_image(width_px, height_px):
    image = np.zeros((1280, 1280, 3), dtype=np.uint8)
    image[640 - height_px // 2:640 + height_px // 2, 640 - width_px // 2:640 + width_px // 2] = 255
    return image, dict(cx=0.5, cy=0.5, w=width_px / 1280, h=height_px / 1280)


def test_crop_lays_the_fish_horizontally_on_a_fixed_canvas():
    image, box = _box_image(width_px=200, height_px=400)  # peixe "em pé"
    crop = crop_fish(image, size=(288, 384), **box)
    assert crop.shape == (288, 384, 3) and crop.dtype == np.uint8
    white = crop[..., 0] > 200
    rows, cols = np.where(white)
    assert cols.max() - cols.min() > rows.max() - rows.min()          # ficou deitado
    assert abs(rows.mean() - 144) < 2 and abs(cols.mean() - 192) < 2  # e centrado
    assert crop[0, 0].tolist() == [128, 128, 128]                      # fundo neutro


def test_crop_keeps_the_scale_between_fish():
    """Um peixe com o dobro do lado ocupa quatro vezes a área: o recorte não amplia o peixe pequeno."""
    small_image, small_box = _box_image(100, 200)
    large_image, large_box = _box_image(200, 400)
    small = (crop_fish(small_image, size=(288, 384), **small_box)[..., 0] > 200).sum()
    large = (crop_fish(large_image, size=(288, 384), **large_box)[..., 0] > 200).sum()
    assert large / small == pytest.approx(4.0, rel=0.05)
    assert large == pytest.approx(200 * 400 * (288 / 792) ** 2, rel=0.05)


def test_box_crossing_the_image_border_is_clipped():
    image = np.zeros((1280, 1280, 3), dtype=np.uint8)
    image[:100, :300] = 255
    crop = crop_fish(image, cx=0.1, cy=0.0, w=0.3, h=0.2, size=(288, 384))  # metade da caixa fora da imagem
    assert crop.shape == (288, 384, 3) and (crop[..., 0] > 200).sum() > 0


def test_corrupted_download_is_rejected(tmp_path, monkeypatch):
    from fishweight import fdwe

    monkeypatch.setattr(fdwe.urllib.request, "urlretrieve", lambda url, target: open(target, "wb").write(b"lixo"))
    with pytest.raises(ValueError, match="MD5"):
        fdwe.download_fdwe(tmp_path)
    assert not (tmp_path / ZIP_NAME).exists()


def _tiny_dataset(path):
    image, _ = _box_image(200, 400)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    root = "Fish_Detection_and_Weight_Estimation_dataset"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{root}/images/train/2021_02_25_den_helder_1000_rgb.png", buffer.getvalue())
        archive.writestr(f"{root}/labels/train/2021_02_25_den_helder_1000_rgb.txt",
                         "0 0.5 0.5 0.15625 0.3125 0.250 00_00\n5 0.5 0.5 0.1 0.1 0.080 61_90\n")
        archive.writestr(f"{root}/images/test/2021_03_01_scheveningen_7_rgb.png", buffer.getvalue())
        archive.writestr(f"{root}/labels/test/2021_03_01_scheveningen_7_rgb.txt", "1 0.5 0.5 0.15625 0.3125 0.300 01_30\n")
    return path


def test_manifest_has_one_row_per_fish(tmp_path):
    manifest = load_fdwe_manifest(_tiny_dataset(tmp_path / "fdwe.zip")).set_index("fish_id")
    assert sorted(manifest.index) == [
        "2021_02_25_den_helder_1000_rgb_0", "2021_02_25_den_helder_1000_rgb_1", "2021_03_01_scheveningen_7_rgb_0"]
    second = manifest.loc["2021_02_25_den_helder_1000_rgb_1"]
    assert (second["split"], second["session"], second["species"]) == ("train", "2021_02_25_den_helder", "dab")
    assert (second["weight"], second["occlusion"]) == (pytest.approx(80.0), 90)
    other = manifest.loc["2021_03_01_scheveningen_7_rgb_0"]
    assert (other["split"], other["session"], other["weight"]) == ("test", "2021_03_01_scheveningen", pytest.approx(300.0))


def test_crops_follow_the_manifest_rows(tmp_path):
    archive = _tiny_dataset(tmp_path / "fdwe.zip")
    manifest = load_fdwe_manifest(archive)
    crops = load_fdwe_crops(manifest, (288, 384), archive, cache_dir=tmp_path)
    assert crops.shape == (3, 288, 384, 3)
    area = dict(zip(manifest["fish_id"], (crops[..., 0] > 200).sum(axis=(1, 2))))
    big, small = area["2021_02_25_den_helder_1000_rgb_0"], area["2021_02_25_den_helder_1000_rgb_1"]
    assert big > 3 * small > 0  # caixa de 200×400 px contra caixa de 128×128 px
    subset = load_fdwe_crops(manifest[manifest["split"] == "test"], (288, 384), archive, cache_dir=tmp_path)
    assert subset.shape[0] == 1  # outro conjunto de peixes não reaproveita o cache


@needs_dataset
def test_real_dataset_matches_the_published_counts():
    manifest = load_fdwe_manifest(REAL_ZIP)
    assert len(manifest) == 2216 and manifest["image"].nunique() == 1086
    assert manifest["split"].value_counts().to_dict() == {"train": 1595, "val": 424, "test": 197}
    assert set(manifest["species"]) == set(SPECIES)
    assert manifest["weight"].min() == pytest.approx(28.0) and manifest["weight"].max() == pytest.approx(1878.0)
    assert manifest["fish_id"].is_unique
