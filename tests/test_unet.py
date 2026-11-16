import numpy as np
import pytest
from helpers import rectangles

from fishweight.unet import dice_iou, predict_masks, train_unet


def test_dice_iou_exact_values():
    pred = np.zeros((4, 4), dtype=bool); pred[0, :] = True         # 4 pixels
    true = np.zeros((4, 4), dtype=bool); true[0, 2:] = True; true[1, 2:] = True  # 4 pixels, 2 em comum
    dice, iou = dice_iou(pred, true)
    assert dice == pytest.approx(0.5)
    assert iou == pytest.approx(1 / 3)
    assert dice_iou(np.zeros((4, 4), dtype=bool), np.zeros((4, 4), dtype=bool)) == (1.0, 1.0)


@pytest.mark.slow
def test_unet_learns_to_segment_a_bright_rectangle():
    images, masks = rectangles(24)
    model = train_unet(images[:16], masks[:16], epochs=25, batch_size=8, pretrained=False, device="cpu")
    predicted = predict_masks(model, images[16:], device="cpu") > 0.5
    dice = np.mean([dice_iou(p, t)[0] for p, t in zip(predicted, masks[16:])])
    assert dice > 0.9


@pytest.mark.slow
def test_predict_returns_probabilities_with_input_shape():
    images, masks = rectangles(4)
    model = train_unet(images, masks, epochs=1, batch_size=4, pretrained=False, device="cpu")
    probabilities = predict_masks(model, images, device="cpu")
    assert probabilities.shape == (4, 32, 64)
    assert probabilities.dtype == np.float32
    assert 0.0 <= probabilities.min() and probabilities.max() <= 1.0
