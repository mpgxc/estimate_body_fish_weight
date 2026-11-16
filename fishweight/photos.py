"""Fotos e máscaras reduzidas para as redes, com cache em disco."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image, ImageOps
from skimage.transform import resize

from fishweight.data import REPO_ROOT
from fishweight.features import EXPECTED_SHAPE, load_mask

CACHE_DIR = REPO_ROOT / "data/cache"  # fora do Git; refeito sob demanda


def load_image_small(path: Path, size: tuple[int, int]) -> np.ndarray:
    """Lê a foto, aplica a orientação EXIF e reduz para ``size`` (linhas, colunas).

    As máscaras foram anotadas no quadro já girado pelo EXIF; ler a foto sem
    aplicar a orientação desalinha 22 das 245 fotos da coleta. ``size`` deve
    ter a proporção 3:4 das fotos, para não deformar o peixe.

    Raises:
        ValueError: A foto não tem a resolução das fotos da coleta.
    """
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if (image.height, image.width) != EXPECTED_SHAPE:
            raise ValueError(
                f"{path}: foto {image.height}×{image.width}, "
                f"esperado {EXPECTED_SHAPE[0]}×{EXPECTED_SHAPE[1]}"
            )
        small = image.resize((size[1], size[0]), Image.Resampling.BILINEAR, reducing_gap=2.0)
    return np.asarray(small, dtype=np.uint8)


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Redimensiona uma máscara booleana preservando a fração de área."""
    return resize(mask.astype(float), shape, order=1, anti_aliasing=True) > 0.5


def cached_arrays(cache_file: Path, names: np.ndarray, build: Callable[[], dict[str, np.ndarray]]) -> dict:
    """Devolve os arrays do cache se ele cobrir as mesmas imagens; senão refaz e grava."""
    if cache_file.is_file():
        cached = np.load(cache_file)
        if np.array_equal(cached["names"], names):
            return {key: cached[key] for key in cached.files if key != "names"}
    arrays = build()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, names=names, **arrays)
    return arrays


def load_photos(
    manifest: pd.DataFrame, size: tuple[int, int], cache_dir: Path = CACHE_DIR
) -> np.ndarray:
    """Fotos reduzidas, ``uint8`` N×linhas×colunas×3, na ordem do manifesto."""

    def build() -> dict[str, np.ndarray]:
        photos = Parallel(n_jobs=-2)(
            delayed(load_image_small)(Path(path), size) for path in manifest["image_path"]
        )
        return {"photos": np.stack(photos)}

    cache_file = Path(cache_dir) / f"fotos_{size[0]}x{size[1]}.npz"
    return cached_arrays(cache_file, manifest["img"].to_numpy(dtype=str), build)["photos"]


def load_masks(
    manifest: pd.DataFrame, size: tuple[int, int], cache_dir: Path = CACHE_DIR
) -> tuple[np.ndarray, np.ndarray]:
    """Máscaras manuais em ``size`` e na resolução de trabalho (434×578), booleanas."""

    def build() -> dict[str, np.ndarray]:
        work = np.stack(
            Parallel(n_jobs=-2)(delayed(load_mask)(Path(path)) for path in manifest["mask_path"])
        )
        return {"small": np.stack([resize_mask(mask, size) for mask in work]), "work": work}

    cache_file = Path(cache_dir) / f"mascaras_{size[0]}x{size[1]}.npz"
    arrays = cached_arrays(cache_file, manifest["img"].to_numpy(dtype=str), build)
    return arrays["small"], arrays["work"]
