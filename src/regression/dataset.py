"""Carregamento e pré-processamento dos dados de peso/features de peixes."""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from src.feature_extraction import extract_features_from_images


def load_dataset(base_path: str) -> pd.DataFrame:
    """Carrega e une anotações de peso com features extraídas das máscaras.

    Se o arquivo CSV de features não existir, ele é gerado automaticamente
    via :func:`~src.feature_extraction.extract_features_from_images`.

    Args:
        base_path: Caminho para o diretório do dataset (deve conter
            ``masks/`` e ``annotations.csv``).

    Returns:
        DataFrame com colunas de peso, espécie, comprimento, largura e
        todas as features morfológicas da máscara.
    """
    *_, base_path_last = base_path.split("/")
    features_file_name = f"{base_path_last}_features.csv"

    if not os.path.isfile(features_file_name):
        extract_features_from_images(
            images_arr=[],
            images_path=f"{base_path}/masks",
            output_file_name=features_file_name,
            resize_shape=(224, 224),
        )

    weights = pd.read_csv(f"{base_path}/annotations.csv")
    features = pd.read_csv(features_file_name)

    weights.index = weights.img
    features.index = features.label

    weights = weights.drop("img", axis=1)
    features = features.drop("label", axis=1)

    data = pd.concat(
        [weights, features], axis=1, ignore_index=False, verify_integrity=True
    )
    return data.reset_index()


def load_unet_dataset(
    df: pd.DataFrame,
    features_file: str = "features_from_unet_predictions.csv",
) -> pd.DataFrame:
    """Une o subconjunto de anotações com as features preditas pela U-Net.

    Args:
        df: DataFrame com colunas ``weight``, ``length``, ``specie``,
            ``width`` e ``index`` (geralmente o conjunto de teste).
        features_file: Caminho para o CSV de features gerado pelas
            predições da U-Net.

    Returns:
        DataFrame com as colunas de anotação alinhadas às features U-Net.
    """
    weights = df[["weight", "length", "specie", "width"]]
    weights.index = df["index"]

    features = pd.read_csv(features_file)
    features.index = features.label
    features = features.drop("label", axis=1)

    data = pd.concat(
        [weights, features], axis=1, ignore_index=False, verify_integrity=True
    )
    return data.reset_index()
