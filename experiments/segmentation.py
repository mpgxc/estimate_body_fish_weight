"""Máscaras previstas fora da amostra: uma U-Net por dobra de P1, P2 e P3.

Cada U-Net treina só com as imagens de treino da dobra e prevê as de teste,
de modo que nenhuma máscara prevista vem de um modelo que viu aquela foto.

Uso: ``uv run python -m experiments.segmentation`` (retoma de onde parou).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.io import imread, imsave
from skimage.transform import resize

from fishweight.data import REPO_ROOT, load_manifest
from fishweight.features import mask_features, predicted_features_file
from fishweight.photos import load_masks, load_photos
from fishweight.torch_utils import pick_device
from fishweight.unet import IMAGE_SIZE, dice_iou, predict_masks, train_unet
from fishweight.splits import PROTOCOLS, Fold, load_splits

RESULTS = REPO_ROOT / "results/segmentation"


def _mask_file(out_dir: Path, fold_name: str, img: str) -> Path:
    return Path(out_dir) / "masks" / fold_name / f"{Path(img).stem}.png"


def run_fold(
    fold: Fold,
    images: np.ndarray,
    masks: np.ndarray,
    image_names: list[str],
    reference_masks: np.ndarray,
    out_dir: Path,
    epochs: int = 20,
    pretrained: bool = True,
    device: str | None = None,
) -> pd.DataFrame:
    """Treina na dobra, grava a máscara prevista de cada imagem de teste e a avalia.

    Se todas as máscaras da dobra já existem em ``out_dir``, não treina de
    novo: relê os arquivos e mede.

    Args:
        images, masks: Fotos e máscaras manuais reduzidas, alinhadas a ``image_names``.
        reference_masks: Máscaras manuais na resolução de trabalho; a máscara
            prevista é levada a essa resolução antes de gravar e medir.

    Returns:
        DataFrame ``img, fold, dice, iou``, na ordem de ``fold["test"]``.
    """
    position = {name: i for i, name in enumerate(image_names)}
    test_idx = [position[name] for name in fold["test"]]
    files = [_mask_file(out_dir, fold["name"], name) for name in fold["test"]]

    if not all(f.is_file() for f in files):
        train_idx = [position[name] for name in fold["train"]]
        model = train_unet(
            images[train_idx], masks[train_idx], epochs=epochs, pretrained=pretrained, device=device
        )
        probabilities = predict_masks(model, images[test_idx], device=device)
        files[0].parent.mkdir(parents=True, exist_ok=True)
        for file, probability, i in zip(files, probabilities, test_idx):
            predicted = resize(probability, reference_masks[i].shape, order=1) > 0.5
            imsave(file, predicted.astype(np.uint8) * 255, check_contrast=False)

    scores = [dice_iou(imread(f) > 127, reference_masks[i]) for f, i in zip(files, test_idx)]
    return pd.DataFrame({
        "img": fold["test"],
        "fold": fold["name"],
        "dice": [s[0] for s in scores],
        "iou": [s[1] for s in scores],
    })


def predicted_features(out_dir: Path, fold: Fold) -> pd.DataFrame:
    """Mede as máscaras previstas de uma dobra: ``fold, img`` e os 12 atributos.

    Raises:
        ValueError: Alguma máscara prevista está vazia (a rede não achou o peixe).
    """
    rows = []
    for img in fold["test"]:
        mask = imread(_mask_file(out_dir, fold["name"], img)) > 127
        if not mask.any():
            # shortcut: falha em vez de prever; trocar por uma regra de rejeição
            # quando a ferramenta (Fase 2) definir o que fazer sem máscara.
            raise ValueError(f"dobra {fold['name']}: máscara prevista vazia para {img}")
        rows.append({"fold": fold["name"], "img": img, **mask_features(mask)})
    return pd.DataFrame(rows)


def main() -> None:
    print(f"dispositivo: {pick_device()}", flush=True)
    manifest = load_manifest()
    images = load_photos(manifest, IMAGE_SIZE)
    masks, work_masks = load_masks(manifest, IMAGE_SIZE)
    names = manifest["img"].tolist()
    for protocol in PROTOCOLS:
        out_dir = RESULTS / protocol
        scores, features = [], []
        for fold in load_splits(protocol):
            start = time.perf_counter()
            fold_scores = run_fold(fold, images, masks, names, work_masks, out_dir)
            scores.append(fold_scores)
            features.append(predicted_features(out_dir, fold))
            print(
                f"{protocol} {fold['name']}: Dice={fold_scores['dice'].mean():.4f} "
                f"IoU={fold_scores['iou'].mean():.4f} ({time.perf_counter() - start:.0f} s)",
                flush=True,
            )
        pd.concat(scores, ignore_index=True).round(5).to_csv(out_dir / "metrics.csv", index=False)
        features_file = predicted_features_file(protocol)
        features_file.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(features, ignore_index=True).to_csv(features_file, index=False)


if __name__ == "__main__":
    main()
