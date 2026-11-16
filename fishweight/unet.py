"""U-Net (encoder MobileNetV2) para segmentar o corpo do peixe."""

from __future__ import annotations

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

from fishweight.torch_utils import augment, pick_device, shuffled_batches, to_tensor

# Linhas, colunas: mesma proporção 3:4 das fotos, múltiplos de 32. Escolhida numa
# divisão interna do treino de P1 (Dice 0,981 com 20 épocas).
IMAGE_SIZE = (192, 256)


def dice_iou(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    """Coeficientes de Dice e IoU entre duas máscaras booleanas."""
    intersection = int(np.logical_and(pred, true).sum())
    total = int(pred.sum()) + int(true.sum())
    if total == 0:
        return 1.0, 1.0
    return 2 * intersection / total, intersection / (total - intersection)


def build_unet(pretrained: bool = True) -> torch.nn.Module:
    """U-Net com encoder MobileNetV2 e uma saída (logit de 'é peixe')."""
    return smp.Unet("mobilenet_v2", encoder_weights="imagenet" if pretrained else None, classes=1)


def _loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    intersection = (prob * target).sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1) / (prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1)
    return bce + (1 - dice).mean()


def train_unet(
    images: np.ndarray,
    masks: np.ndarray,
    epochs: int = 20,
    batch_size: int = 16,
    seed: int = 0,
    pretrained: bool = True,
    device: str | None = None,
) -> torch.nn.Module:
    """Treina a U-Net por um número fixo de épocas (sem parada antecipada).

    Args:
        images: Fotos ``uint8`` no formato N×H×W×3; H e W múltiplos de 32.
        masks: Máscaras booleanas N×H×W.
    """
    device = device or pick_device()
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    model = build_unet(pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_all = to_tensor(images)
    y_all = torch.from_numpy(masks).float().unsqueeze(1)

    model.train()
    for _ in range(epochs):
        for batch in shuffled_batches(len(x_all), batch_size, rng):
            x, y = augment(x_all[batch], rng, y_all[batch])
            optimizer.zero_grad()
            loss = _loss(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict_masks(
    model: torch.nn.Module, images: np.ndarray, device: str | None = None, batch_size: int = 16
) -> np.ndarray:
    """Probabilidade de 'é peixe' por pixel, ``float32`` no formato N×H×W."""
    device = device or pick_device()
    model.to(device).eval()
    x_all = to_tensor(images)
    parts = [
        torch.sigmoid(model(x_all[i:i + batch_size].to(device))).squeeze(1).cpu()
        for i in range(0, len(x_all), batch_size)
    ]
    return torch.cat(parts).numpy().astype(np.float32)
