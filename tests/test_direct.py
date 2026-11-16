import numpy as np
import pandas as pd
import pytest
from helpers import rectangles

from fishweight.direct import (
    BACKBONES,
    add_embeddings,
    direct_fit_predict,
    embed,
    input_size,
    train_regressor,
)

TINY = "resnet10t"  # rede pequena do timm, sem pesos pré-treinados, só para os testes
FAST = dict(backbone=TINY, pretrained=False, device="cpu", lr=3e-3, batch_size=8)

PHOTOS, MASKS = rectangles(64)
WEIGHTS = MASKS.sum(axis=(1, 2)).astype(float)  # "peso" proporcional à área do retângulo
NAMES = [f"IMG_{i:02d}.jpg" for i in range(64)]


def _rows(indices):
    return pd.DataFrame({
        "img": [NAMES[i] for i in indices],
        "weight": WEIGHTS[indices],
        "species": ["a" if i % 2 else "b" for i in indices],
        "length": WEIGHTS[indices] ** 0.5,
        "width": WEIGHTS[indices] ** 0.5 / 2,
        "area": WEIGHTS[indices],
    })


TRAIN, TEST = _rows(range(48)), _rows(range(48, 64))


def test_every_backbone_keeps_the_photo_proportion():
    for name in BACKBONES:
        rows, cols = input_size(name)
        assert rows / cols == pytest.approx(3 / 4)


@pytest.mark.slow
def test_embeddings_do_not_depend_on_batch_size():
    one = embed(PHOTOS[:6], TINY, pretrained=False, device="cpu", batch_size=1)
    all_at_once = embed(PHOTOS[:6], TINY, pretrained=False, device="cpu", batch_size=6)
    assert one.shape == (6, 512)
    assert np.allclose(one, all_at_once, atol=1e-4)


@pytest.mark.slow
def test_regressor_learns_weight_from_the_photo():
    fitted = train_regressor(PHOTOS[:48], WEIGHTS[:48], epochs=40, **FAST)
    error = np.abs(fitted.predict(PHOTOS[48:]) - WEIGHTS[48:]).mean()
    baseline = np.abs(WEIGHTS[:48].mean() - WEIGHTS[48:]).mean()
    assert error < 0.5 * baseline


@pytest.mark.slow
def test_log_target_gives_positive_weights():
    fitted = train_regressor(PHOTOS[:16], WEIGHTS[:16], epochs=2, log_target=True, **FAST)
    predicted = fitted.predict(PHOTOS[48:])
    assert np.isfinite(predicted).all() and (predicted > 0).all()


@pytest.mark.slow
def test_predictions_ignore_the_annotations_of_test_fish():
    """Espécie e medidas de fita são saídas auxiliares no treino; no teste só a foto entra."""
    fit_predict = direct_fit_predict(PHOTOS, NAMES, aux=("species", "measures"), epochs=2, **FAST)
    tampered = TEST.assign(species="nunca_vista", length=999.0, width=999.0)
    assert np.array_equal(fit_predict(TRAIN, TEST.drop(columns="weight")),
                          fit_predict(TRAIN, tampered.drop(columns="weight")))


@pytest.mark.slow
def test_hybrid_uses_the_mask_features_of_the_test_rows():
    fit_predict = direct_fit_predict(PHOTOS, NAMES, tabular_columns=["area"], epochs=2, **FAST)
    changed = TEST.assign(area=TEST["area"] * 3)
    assert not np.allclose(fit_predict(TRAIN, TEST.drop(columns="weight")),
                           fit_predict(TRAIN, changed.drop(columns="weight")))


def test_add_embeddings_joins_by_image_name(tmp_path):
    file = tmp_path / "embeddings.npz"
    np.savez(file, names=np.array(["b.jpg", "a.jpg"]), embeddings=np.array([[1.0, 2.0], [3.0, 4.0]]))
    table, columns = add_embeddings(pd.DataFrame({"img": ["a.jpg", "b.jpg"], "weight": [10, 20]}), file)
    assert columns == ["emb_000", "emb_001"]
    assert table[columns].to_numpy().tolist() == [[3.0, 4.0], [1.0, 2.0]]
    with pytest.raises(ValueError, match="c.jpg"):
        add_embeddings(pd.DataFrame({"img": ["a.jpg", "c.jpg"]}), file)


@pytest.mark.slow
def test_training_can_start_from_a_backbone_trained_elsewhere():
    """Pré-treino no FDWE: o ajuste fino parte dos pesos da rede pré-treinada, não dos de fábrica."""
    source = train_regressor(PHOTOS[:16], WEIGHTS[:16], epochs=1, **FAST)
    state = {key: value.cpu().clone() for key, value in source.model.backbone.state_dict().items()}
    started = train_regressor(PHOTOS[:16], WEIGHTS[:16], epochs=0, backbone_state=state, **FAST)
    fresh = train_regressor(PHOTOS[:16], WEIGHTS[:16], epochs=0, **FAST)
    key = next(k for k in state if k.endswith("weight") and state[k].ndim == 4)
    assert np.array_equal(started.model.backbone.state_dict()[key].cpu().numpy(), state[key].numpy())
    assert not np.array_equal(fresh.model.backbone.state_dict()[key].cpu().numpy(), state[key].numpy())


@pytest.mark.slow
def test_unreliable_tape_measures_do_not_reach_the_auxiliary_loss():
    """As duas traíras com comprimento duvidoso treinam o peso, mas não ensinam medida à rede."""
    from fishweight.data import UNRELIABLE_MEASURES

    names = [next(iter(UNRELIABLE_MEASURES))] + NAMES[1:]
    fit_predict = direct_fit_predict(PHOTOS, names, aux=("measures",), epochs=2, **FAST)
    train = TRAIN.assign(img=names[:48])
    absurd = train.copy()
    absurd.loc[0, ["length", "width"]] = [999.0, 999.0]
    test = TEST.drop(columns="weight")
    assert np.array_equal(fit_predict(train, test), fit_predict(absurd, test))
