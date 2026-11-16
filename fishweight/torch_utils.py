"""Peças de PyTorch compartilhadas pela segmentação e pela regressão direta."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def pick_device() -> str:
    """``cuda`` se houver GPU NVIDIA, ``mps`` em Apple Silicon, senão ``cpu``."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def to_tensor(
    images: np.ndarray,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> torch.Tensor:
    """Converte fotos ``uint8`` N×H×W×3 em tensor normalizado N×3×H×W."""
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255
    return (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)


def augment(
    x: torch.Tensor, rng: torch.Generator, y: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Espelha e varia brilho e contraste; nunca muda o tamanho do peixe na imagem.

    A escala da foto é o que liga pixels a gramas, então zoom e recorte
    redimensionado ficam de fora. ``y`` (máscara) recebe os mesmos espelhamentos.
    """
    if torch.rand(1, generator=rng) < 0.5:
        x, y = x.flip(3), None if y is None else y.flip(3)
    if torch.rand(1, generator=rng) < 0.5:
        x, y = x.flip(2), None if y is None else y.flip(2)
    contrast = 1 + (torch.rand(x.shape[0], 1, 1, 1, generator=rng) - 0.5) * 0.4
    brightness = (torch.rand(x.shape[0], 1, 1, 1, generator=rng) - 0.5) * 0.4
    return x * contrast + brightness, y


def shuffled_batches(n: int, batch_size: int, rng: torch.Generator) -> Iterator[torch.Tensor]:
    """Índices de uma época em lotes embaralhados; descarta lote de um (BatchNorm não aceita)."""
    order = torch.randperm(n, generator=rng)
    for start in range(0, n, batch_size):
        batch = order[start:start + batch_size]
        if len(batch) > 1:
            yield batch
