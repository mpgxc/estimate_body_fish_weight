"""Regressão direta: rede que prevê o peso a partir da foto.

Cobre três usos, todos com o mesmo backbone do timm:

* **congelado** (:func:`embed`): a rede pré-treinada só resume a foto num
  vetor; um regressor simples é ajustado sobre esses vetores.
* **ajuste fino** (:func:`train_regressor`): a rede inteira é treinada para
  prever o peso, com saídas auxiliares opcionais (espécie, medidas de fita).
* **híbrido**: o ajuste fino recebe também atributos da máscara
  (``tabular``), fundidos ao vetor da foto antes das saídas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from torch import nn

from fishweight.data import REPO_ROOT, UNRELIABLE_MEASURES
from fishweight.torch_utils import augment, pick_device, shuffled_batches, to_tensor

# nome curto → (modelo do timm com os pesos fixados, entrada em linhas×colunas).
# As entradas mantêm a proporção 3:4 das fotos; a do ViT é múltipla de 14 (o patch).
BACKBONES: dict[str, tuple[str, tuple[int, int]]] = {
    "resnet50": ("resnet50.a1_in1k", (288, 384)),
    "efficientnet_b3": ("efficientnet_b3.ra2_in1k", (288, 384)),
    "convnext_tiny": ("convnext_tiny.fb_in1k", (288, 384)),
    "dinov2_vits14": ("vit_small_patch14_dinov2.lvd142m", (294, 392)),
}
AUX_WEIGHT = 0.3  # peso das perdas auxiliares em relação à do peso


def input_size(backbone: str) -> tuple[int, int]:
    """Tamanho de entrada (linhas, colunas) de um backbone de :data:`BACKBONES`."""
    return BACKBONES[backbone][1]


def embedding_file(backbone: str) -> Path:
    """Arquivo onde ``experiments.e2_direct`` grava os vetores de um backbone."""
    return REPO_ROOT / f"data/features/embeddings_{backbone}.npz"


def _build_backbone(backbone: str, pretrained: bool) -> tuple[nn.Module, tuple, tuple]:
    """Cria o backbone sem a camada de classificação e devolve sua normalização."""
    timm_name, size = BACKBONES.get(backbone, (backbone, None))
    # O ViT tem tamanho de entrada fixo; os pesos de posição são reamostrados para o nosso.
    extra = {"img_size": size} if timm_name.startswith("vit") else {}
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, **extra)
    config = timm.data.resolve_model_data_config(model)
    return model, tuple(config["mean"]), tuple(config["std"])


@torch.no_grad()
def embed(
    images: np.ndarray,
    backbone: str,
    pretrained: bool = True,
    device: str | None = None,
    batch_size: int = 16,
    seed: int = 0,
) -> np.ndarray:
    """Resume cada foto num vetor com o backbone pré-treinado, sem treinar nada.

    Como nenhum parâmetro é ajustado nos nossos dados, os vetores podem ser
    calculados uma vez para todas as fotos sem vazamento entre treino e teste.
    """
    device = device or pick_device()
    torch.manual_seed(seed)
    model, mean, std = _build_backbone(backbone, pretrained)
    model.to(device).eval()
    x_all = to_tensor(images, mean, std)
    parts = [model(x_all[i:i + batch_size].to(device)).cpu() for i in range(0, len(x_all), batch_size)]
    return torch.cat(parts).numpy().astype(np.float32)


def add_embeddings(table: pd.DataFrame, file: Path) -> tuple[pd.DataFrame, list[str]]:
    """Acrescenta à tabela as colunas ``emb_000…`` lidas de ``file``, casando por imagem.

    Raises:
        ValueError: Alguma imagem da tabela não tem vetor no arquivo.
    """
    stored = np.load(file)
    columns = [f"emb_{i:03d}" for i in range(stored["embeddings"].shape[1])]
    embeddings = pd.DataFrame(stored["embeddings"], columns=columns)
    embeddings.insert(0, "img", stored["names"])
    missing = sorted(set(table["img"]) - set(embeddings["img"]))
    if missing:
        raise ValueError(f"Sem vetor em {file} para: {missing[:5]}")
    return table.merge(embeddings, on="img", validate="one_to_one"), columns


class WeightRegressor(nn.Module):
    """Backbone, ramo opcional de atributos tabulares e as saídas."""

    def __init__(self, backbone: nn.Module, n_species: int = 0, n_measures: int = 0, n_tabular: int = 0):
        super().__init__()
        self.backbone = backbone
        width = backbone.num_features
        self.tabular = None
        if n_tabular:
            self.tabular = nn.Sequential(nn.Linear(n_tabular, 32), nn.ReLU())
            width += 32
        self.weight = nn.Linear(width, 1)
        self.species = nn.Linear(width, n_species) if n_species else None
        self.measures = nn.Linear(width, n_measures) if n_measures else None

    def forward(self, x: torch.Tensor, tabular: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        if self.tabular is not None:
            features = torch.cat([features, self.tabular(tabular)], dim=1)
        out = {"weight": self.weight(features).squeeze(1)}
        if self.species is not None:
            out["species"] = self.species(features)
        if self.measures is not None:
            out["measures"] = self.measures(features)
        return out


def _standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, std = values.mean(axis=0), values.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return (values - mean) / std, mean, std


@dataclass
class FittedRegressor:
    """Rede treinada mais tudo que é preciso para prever em gramas."""

    model: WeightRegressor
    image_mean: tuple
    image_std: tuple
    target_mean: float
    target_std: float
    log_target: bool
    tabular_mean: np.ndarray | None
    tabular_std: np.ndarray | None
    device: str

    @torch.no_grad()
    def predict(self, images: np.ndarray, tabular: np.ndarray | None = None, batch_size: int = 16) -> np.ndarray:
        """Peso previsto, em gramas, para cada foto."""
        self.model.eval()
        x_all = to_tensor(images, self.image_mean, self.image_std)
        tab_all = None
        if self.tabular_mean is not None:
            scaled = (np.asarray(tabular, dtype=float) - self.tabular_mean) / self.tabular_std
            tab_all = torch.from_numpy(scaled).float()
        parts = []
        for i in range(0, len(x_all), batch_size):
            tab = None if tab_all is None else tab_all[i:i + batch_size].to(self.device)
            parts.append(self.model(x_all[i:i + batch_size].to(self.device), tab)["weight"].cpu())
        target = torch.cat(parts).numpy().astype(float) * self.target_std + self.target_mean
        return np.exp(target) if self.log_target else target


def train_regressor(
    images: np.ndarray,
    weights: np.ndarray,
    backbone: str,
    species: Sequence[str] | None = None,
    measures: np.ndarray | None = None,
    tabular: np.ndarray | None = None,
    log_target: bool = False,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-4,
    seed: int = 0,
    pretrained: bool = True,
    device: str | None = None,
    backbone_state: dict | None = None,
) -> FittedRegressor:
    """Ajuste fino do backbone para prever o peso, por um número fixo de épocas.

    Args:
        images: Fotos ``uint8`` N×H×W×3 no tamanho de :func:`input_size`.
        weights: Peso em gramas de cada foto.
        species: Espécie anotada; se dada, a rede ganha uma saída auxiliar
            de classificação. Só entra na perda, nunca como entrada.
        measures: Medidas de fita N×k (comprimento, largura); saída auxiliar.
        tabular: Atributos da máscara N×k; entrada do ramo híbrido.
        log_target: Treina em ``log(peso)`` em vez de gramas.
        backbone_state: Pesos de um backbone já treinado em outro conjunto
            (pré-treino no FDWE); substituem os de fábrica antes do ajuste.

    A perda é Huber sobre o alvo padronizado com a média e o desvio do
    treino. Não há parada antecipada nem escolha de época.
    """
    device = device or pick_device()
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    backbone_module, image_mean, image_std = _build_backbone(backbone, pretrained)
    if backbone_state is not None:
        backbone_module.load_state_dict(backbone_state)

    weights = np.asarray(weights, dtype=float)
    target, target_mean, target_std = _standardize(np.log(weights) if log_target else weights)
    x_all = to_tensor(images, image_mean, image_std)
    y_all = torch.from_numpy(target).float()

    tab_all = tabular_mean = tabular_std = None
    if tabular is not None:
        scaled, tabular_mean, tabular_std = _standardize(np.asarray(tabular, dtype=float))
        tab_all = torch.from_numpy(scaled).float()
    species_all = None
    classes: list[str] = []
    if species is not None:
        classes = sorted(set(species))
        species_all = torch.tensor([classes.index(s) for s in species])
    measures_all = None
    if measures is not None:
        # Linhas com NaN (medida de fita não confiável) ficam fora da perda auxiliar.
        measures = np.asarray(measures, dtype=float)
        mean, std = np.nanmean(measures, axis=0), np.nanstd(measures, axis=0)
        measures_all = torch.from_numpy((measures - mean) / np.where(std > 0, std, 1.0)).float()

    model = WeightRegressor(
        backbone_module,
        n_species=len(classes),
        n_measures=0 if measures_all is None else measures_all.shape[1],
        n_tabular=0 if tab_all is None else tab_all.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    model.train()
    for _ in range(epochs):
        for batch in shuffled_batches(len(x_all), batch_size, rng):
            x, _unused = augment(x_all[batch], rng)
            tab = None if tab_all is None else tab_all[batch].to(device)
            out = model(x.to(device), tab)
            loss = F.smooth_l1_loss(out["weight"], y_all[batch].to(device))
            if species_all is not None:
                loss = loss + AUX_WEIGHT * F.cross_entropy(out["species"], species_all[batch].to(device))
            if measures_all is not None:
                target = measures_all[batch].to(device)
                known = ~torch.isnan(target).any(dim=1)
                if known.any():
                    loss = loss + AUX_WEIGHT * F.smooth_l1_loss(out["measures"][known], target[known])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    return FittedRegressor(
        model, image_mean, image_std, float(target_mean), float(target_std),
        log_target, tabular_mean, tabular_std, device,
    )


def _reliable_measures(rows: pd.DataFrame) -> np.ndarray:
    """Comprimento e largura de fita, com NaN onde a anotação não é confiável."""
    measures = rows[["length", "width"]].to_numpy(dtype=float, copy=True)
    measures[rows["img"].isin(UNRELIABLE_MEASURES).to_numpy()] = np.nan
    return measures


def direct_fit_predict(
    photos: np.ndarray,
    names: Sequence[str],
    backbone: str,
    aux: Sequence[str] = (),
    tabular_columns: Sequence[str] = (),
    **train_kwargs,
) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]:
    """Monta o ``fit_predict`` de um modelo de regressão direta para o executor.

    Args:
        photos: Fotos de todos os peixes, no tamanho do backbone.
        names: Nome da imagem de cada foto, na mesma ordem.
        aux: Saídas auxiliares: ``"species"``, ``"measures"`` (comprimento e
            largura de fita) ou ambas. Usam só as linhas de treino.
        tabular_columns: Colunas da tabela que entram no ramo híbrido.
        **train_kwargs: Repassados a :func:`train_regressor`.
    """
    position = {name: i for i, name in enumerate(names)}
    tabular_columns = list(tabular_columns)

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        fitted = train_regressor(
            photos[[position[img] for img in train["img"]]],
            train["weight"].to_numpy(),
            backbone,
            species=train["species"].tolist() if "species" in aux else None,
            measures=_reliable_measures(train) if "measures" in aux else None,
            tabular=train[tabular_columns].to_numpy() if tabular_columns else None,
            **train_kwargs,
        )
        return fitted.predict(
            photos[[position[img] for img in test["img"]]],
            test[tabular_columns].to_numpy() if tabular_columns else None,
        )

    return fit_predict
