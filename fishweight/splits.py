"""Divisões de treino e teste compartilhadas por todos os experimentos."""

from __future__ import annotations

import json
from typing import TypedDict

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from fishweight.data import REPO_ROOT, load_manifest

PROTOCOLS = ("p1", "p2", "p3")
SPLITS_DIR = REPO_ROOT / "data/splits"
# As 74 imagens deste arquivo são o conjunto de teste da monografia.
P1_TEST_SOURCE = REPO_ROOT / "features_from_unet_predictions.csv"


class Fold(TypedDict):
    name: str
    train: list[str]
    test: list[str]


def _fold(name: str, images: pd.Series, test_mask) -> Fold:
    return {
        "name": name,
        "train": sorted(images[~test_mask]),
        "test": sorted(images[test_mask]),
    }


def build_splits(manifest: pd.DataFrame) -> dict[str, list[Fold]]:
    """Constrói P1 (reprodução), P2 (aleatório estratificado) e P3 (por sessão).

    Args:
        manifest: Saída de :func:`fishweight.data.load_manifest`.

    Returns:
        Dicionário protocolo → lista de dobras com nomes de imagem.
    """
    images = manifest["img"]

    p1_test = set(pd.read_csv(P1_TEST_SOURCE)["label"])
    p1 = [_fold("p1", images, images.isin(p1_test).to_numpy())]

    p2 = []
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    for i, (_, test_idx) in enumerate(splitter.split(images, manifest["species_group"])):
        p2.append(_fold(f"r{i // 5}f{i % 5}", images, images.index.isin(test_idx)))

    p3 = [
        _fold(group, images, (manifest["session_group"] == group).to_numpy())
        for group in sorted(manifest["session_group"].unique())
    ]
    return {"p1": p1, "p2": p2, "p3": p3}


def load_splits(protocol: str) -> list[Fold]:
    """Lê as dobras versionadas de um protocolo (``p1``, ``p2`` ou ``p3``)."""
    return json.loads((SPLITS_DIR / f"{protocol}.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for protocol, folds in build_splits(load_manifest()).items():
        (SPLITS_DIR / f"{protocol}.json").write_text(
            json.dumps(folds, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"{protocol}: {len(folds)} dobras")
