import numpy as np
import pandas as pd
import pytest
from helpers import rectangles
from skimage.io import imread, imsave

from experiments.segmentation import predicted_features, run_fold
from fishweight.unet import dice_iou

IMAGES, MASKS = rectangles(12)
NAMES = [f"IMG_{i:02d}.jpg" for i in range(12)]
FOLD = {"name": "f0", "train": NAMES[:8], "test": NAMES[8:]}


def _run(out_dir, epochs):
    return run_fold(FOLD, IMAGES, MASKS, NAMES, reference_masks=MASKS, out_dir=out_dir,
                    epochs=epochs, pretrained=False, device="cpu")


@pytest.mark.slow
def test_run_fold_writes_one_mask_per_test_image_and_scores_it(tmp_path):
    scores = _run(tmp_path, epochs=25)
    assert list(scores.columns) == ["img", "fold", "dice", "iou"]
    assert scores["img"].tolist() == FOLD["test"] and set(scores["fold"]) == {"f0"}
    assert scores[["dice", "iou"]].ge(0).all().all() and scores[["dice", "iou"]].le(1).all().all()
    saved = imread(tmp_path / "masks/f0/IMG_08.png") > 127
    assert dice_iou(saved, MASKS[8])[0] == pytest.approx(scores["dice"].iloc[0])


@pytest.mark.slow
def test_run_fold_skips_training_when_masks_exist(tmp_path):
    first = _run(tmp_path, epochs=25)
    second = _run(tmp_path, epochs=0)  # se treinasse de novo, a rede sem treino daria outras máscaras
    pd.testing.assert_frame_equal(first, second)


def test_empty_predicted_mask_is_rejected(tmp_path):
    folder = tmp_path / "masks/f0"
    folder.mkdir(parents=True)
    imsave(folder / "IMG_08.png", np.zeros((32, 64), dtype=np.uint8), check_contrast=False)
    with pytest.raises(ValueError, match=r"f0.*IMG_08"):
        predicted_features(tmp_path, {"name": "f0", "train": [], "test": ["IMG_08.jpg"]})
