"""Segmentação clássica via K-means e threshold adaptativo."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage import exposure, feature, filters, measure, morphology
from skimage.color import rgb2gray
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle

from src.commons import (
    auto_invert_image_mask,
    binarize_image,
    check_colision_border,
    find_bighest_cluster,
    smoothing_mask_edges,
)


def recreate_image(
    codebook: np.ndarray, labels: np.ndarray, w: int, h: int
) -> np.ndarray:
    """Reconstrói a imagem quantizada a partir do codebook K-means.

    Args:
        codebook: Centroides do K-means (n_clusters, n_features).
        labels: Rótulo de cluster para cada pixel (w*h,).
        w: Largura da imagem original.
        h: Altura da imagem original.

    Returns:
        Imagem reconstruída com shape (w, h, n_features).
    """
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


def kmeans_segmentation(
    image: np.ndarray,
    n_clusters: int = 3,
    n_samples: int = 1000,
    random_state: int = 0,
) -> np.ndarray:
    """Segmenta uma imagem RGB usando K-means e retorna a máscara do maior cluster.

    Args:
        image: Imagem RGB de entrada com shape (H, W, 3).
        n_clusters: Número de clusters K-means.
        n_samples: Número de amostras para ajuste do K-means.
        random_state: Semente aleatória para reprodutibilidade.

    Returns:
        Máscara booleana do maior cluster foreground.
    """
    w, h, d = image.shape
    image_array = image.reshape((w * h, d))
    image_array_sample = shuffle(image_array, random_state=random_state, n_samples=n_samples)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto").fit(
        image_array_sample
    )
    labels = pairwise_distances_argmin(kmeans.cluster_centers_, image_array)

    segmented = recreate_image(kmeans.cluster_centers_, labels, w, h)
    gray = rgb2gray(segmented)
    binary = binarize_image(gray)
    return find_bighest_cluster(binary)


def apply_canny_mask(image: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Gera uma máscara de borda usando o detector de Canny.

    Args:
        image: Imagem grayscale de entrada.
        sigma: Desvio padrão do filtro gaussiano no Canny.

    Returns:
        Imagem binária com as bordas detectadas.
    """
    return feature.canny(image, sigma=sigma)


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Preenche buracos em uma máscara binária.

    Args:
        mask: Máscara binária de entrada.

    Returns:
        Máscara com buracos preenchidos.
    """
    return ndi.binary_fill_holes(mask)
