import numpy as np
import pytest
from PIL import Image
from skimage.io import imsave

from fishweight.data import load_manifest
from fishweight.features import load_mask
from fishweight.photos import load_image_small, load_masks, load_photos, resize_mask

SIZE = (192, 256)


def test_small_image_keeps_layout(tmp_path):
    photo = np.zeros((3472, 4624, 3), dtype=np.uint8)
    photo[:1736, :2312] = (255, 0, 0)  # quadrante superior esquerdo vermelho
    imsave(tmp_path / "photo.png", photo, check_contrast=False)
    small = load_image_small(tmp_path / "photo.png", SIZE)
    assert small.shape == (192, 256, 3) and small.dtype == np.uint8
    assert small[:92, :124].mean(axis=(0, 1)) == pytest.approx([255, 0, 0], abs=2)
    assert small[100:, 132:].max() <= 2


def test_exif_orientation_is_applied(tmp_path):
    """Celulares gravam a foto deitada e anotam a rotação no EXIF; a máscara está no quadro já girado."""
    upright = np.zeros((3472, 4624, 3), dtype=np.uint8)
    upright[:1736, :2312] = (255, 0, 0)
    exif = Image.Exif()
    exif[274] = 6  # "girar 90° no sentido horário para exibir"
    Image.fromarray(np.rot90(upright, 1)).save(tmp_path / "photo.jpg", exif=exif, quality=95)
    small = load_image_small(tmp_path / "photo.jpg", SIZE)
    assert small.shape == (192, 256, 3)
    assert small[:92, :124].mean(axis=(0, 1)) == pytest.approx([255, 0, 0], abs=6)
    assert small[100:, 132:].max() <= 6


def test_wrong_photo_resolution_is_rejected(tmp_path):
    imsave(tmp_path / "small.png", np.zeros((1080, 1920, 3), dtype=np.uint8), check_contrast=False)
    with pytest.raises(ValueError, match=r"small\.png.*1080.*1920"):
        load_image_small(tmp_path / "small.png", SIZE)


@pytest.mark.parametrize("shape", [(192, 256), (434, 578)])
def test_resize_mask_preserves_area_fraction(shape):
    mask = np.zeros((400, 800), dtype=bool)
    mask[100:200, 200:600] = True  # 1/8 da imagem
    resized = resize_mask(mask, shape)
    assert resized.shape == shape and resized.dtype == bool
    assert resized.mean() == pytest.approx(1 / 8, rel=0.01)


def test_photos_and_manual_masks_are_aligned():
    """Se a foto viesse girada pelo EXIF, o peixe não cairia dentro da máscara."""
    manifest = load_manifest().iloc[::12].head(20)
    for row in manifest.itertuples():
        photo = load_image_small(row.image_path, SIZE).astype(float)
        mask = resize_mask(load_mask(row.mask_path), SIZE)
        inside, outside = photo[mask].mean(axis=0), photo[~mask].mean(axis=0)
        assert np.abs(inside - outside).max() > 10, row.img


def test_cached_arrays_follow_the_manifest(tmp_path):
    manifest = load_manifest().head(3)
    photos = load_photos(manifest, SIZE, cache_dir=tmp_path)
    small, work = load_masks(manifest, SIZE, cache_dir=tmp_path)
    assert photos.shape == (3, 192, 256, 3) and small.shape == (3, 192, 256) and work.shape == (3, 434, 578)
    assert np.array_equal(photos[0], load_image_small(manifest["image_path"].iloc[0], SIZE))
    assert np.array_equal(work[2], load_mask(manifest["mask_path"].iloc[2]))
    # Outro manifesto não pode receber o cache antigo.
    other = load_photos(load_manifest().iloc[3:5], SIZE, cache_dir=tmp_path)
    assert other.shape[0] == 2 and not np.array_equal(other[0], photos[0])
