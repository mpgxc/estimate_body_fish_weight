"""Atributos de tamanho e forma da máscara, na geometria nativa da foto."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from skimage.io import imread
from skimage.measure import block_reduce, label, regionprops_table

from fishweight.data import REPO_ROOT, load_manifest

EXPECTED_SHAPE = (3472, 4624)  # linhas, colunas das fotos da coleta
WORK_SCALE = 8  # redução inteira, sem esticar: 434×578

# Posição, orientação e momentos brutos ficam de fora: dependem de onde o
# peixe foi colocado na caixa, não do peixe.
FEATURE_NAMES = [
    "area",
    "area_bbox",
    "area_convex",
    "area_filled",
    "axis_major_length",
    "axis_minor_length",
    "eccentricity",
    "equivalent_diameter_area",
    "extent",
    "feret_diameter_max",
    "perimeter",
    "solidity",
]


def load_mask(path: Path) -> np.ndarray:
    """Lê uma máscara em resolução cheia e devolve a versão booleana reduzida.

    Raises:
        ValueError: A máscara não tem a resolução das fotos da coleta.
    """
    arr = imread(path)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    if arr.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"{path}: máscara {arr.shape[0]}×{arr.shape[1]}, "
            f"esperado {EXPECTED_SHAPE[0]}×{EXPECTED_SHAPE[1]}"
        )
    return block_reduce(arr > 127, (WORK_SCALE, WORK_SCALE), np.mean) > 0.5


def mask_features(mask: np.ndarray) -> dict[str, float]:
    """Mede o maior componente conexo da máscara.

    Raises:
        ValueError: A máscara está vazia.
    """
    labels = label(mask)
    if labels.max() == 0:
        raise ValueError("Máscara vazia: nenhum componente para medir")
    largest = np.argmax(np.bincount(labels.ravel())[1:]) + 1
    table = regionprops_table((labels == largest).astype(np.uint8), properties=FEATURE_NAMES)
    return {name: float(table[name][0]) for name in FEATURE_NAMES}


def _measure(mask_path: str) -> dict[str, float]:
    return mask_features(load_mask(Path(mask_path)))


def build_feature_table(
    manifest: pd.DataFrame,
    cache_path: Path = REPO_ROOT / "data/features/manual.csv",
) -> pd.DataFrame:
    """Devolve o manifesto com as colunas de :data:`FEATURE_NAMES`.

    Usa ``cache_path`` se ele cobrir exatamente as imagens do manifesto;
    caso contrário mede as máscaras de novo e regrava o arquivo.
    """
    cache_path = Path(cache_path)
    features = None
    if cache_path.is_file():
        cached = pd.read_csv(cache_path)
        if list(cached["img"]) == list(manifest["img"]):
            features = cached

    if features is None:
        rows = Parallel(n_jobs=-2)(
            delayed(_measure)(path) for path in manifest["mask_path"]
        )
        features = pd.DataFrame(rows)
        features.insert(0, "img", manifest["img"].to_numpy())
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(cache_path, index=False)

    return manifest.merge(features, on="img", validate="one_to_one")


if __name__ == "__main__":
    table = build_feature_table(load_manifest())
    print(f"{len(table)} máscaras → data/features/manual.csv")
