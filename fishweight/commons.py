from __future__ import annotations

import statistics
from typing import List, Optional, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage import exposure, filters, measure, morphology
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm


def check_colision_border(mask: np.ndarray) -> bool:
    """Verifica se a máscara binária toca alguma borda da imagem.

    Args:
        mask: Máscara binária 2-D.

    Returns:
        True se algum pixel da borda for não-zero, False caso contrário.
    """
    x, *_ = mask.shape

    left = mask[:1, ].flatten()
    right = mask[x - 1: x, ].flatten()
    top = mask[:, :1].flatten()
    bottom = mask[:, x - 1: x].flatten()

    borders_flatten = [left, right, top, bottom]

    return bool(np.concatenate(borders_flatten).sum())


def binarize_image(arr: np.ndarray) -> np.ndarray:
    """Binariza um array usando o limiar de triangle threshold.

    Args:
        arr: Array de entrada (grayscale ou float).

    Returns:
        Máscara booleana com os pixels acima do limiar como True.
    """
    return arr > filters.threshold_triangle(arr)


def plot(
    arr_images: List[np.ndarray] = [],
    grid: Tuple[int, int] = (2, 2),
    cmap: str = "inferno",
) -> None:
    """Exibe uma grade de imagens com matplotlib.

    Args:
        arr_images: Lista de arrays de imagem a exibir.
        grid: Tupla (linhas, colunas) da grade de exibição.
        cmap: Colormap do matplotlib a usar.
    """
    fig = plt.figure(figsize=(20, 10))

    image_grid = ImageGrid(fig, 111, nrows_ncols=grid, axes_pad=0.1)

    for ax, img in zip(image_grid, arr_images):
        ax.imshow(img, cmap)
        ax.axis("off")

    plt.show()


def load_images_from_directory_resize(
    arr_paths: List[str],
    is_gray: bool = False,
    dim: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Carrega e redimensiona imagens em paralelo a partir de uma lista de caminhos.

    Args:
        arr_paths: Lista de caminhos para os arquivos de imagem.
        is_gray: Se True, converte as imagens para escala de cinza.
        dim: Dimensão (altura, largura) para redimensionamento.

    Returns:
        Array NumPy com as imagens carregadas e redimensionadas.
    """

    def process(img_path: str) -> np.ndarray:
        image = imread(img_path)
        image = resize(image, dim, anti_aliasing=True)
        if is_gray:
            return rgb2gray(image)
        return image

    return np.asarray(
        Parallel(n_jobs=-2)(delayed(process)(path) for path in arr_paths)
    )


def load_images_from_directory(
    arr_paths: List[str],
    is_gray: bool = False,
) -> np.ndarray:
    """Carrega imagens a partir de uma lista de caminhos sem redimensionamento.

    Args:
        arr_paths: Lista de caminhos para os arquivos de imagem.
        is_gray: Se True, converte as imagens para escala de cinza.

    Returns:
        Array NumPy com as imagens carregadas.
    """
    arr_images = []

    for img_path in tqdm(arr_paths):
        image = imread(img_path)
        if is_gray:
            arr_images.append(rgb2gray(image))
        else:
            arr_images.append(image)

    return np.asarray(arr_images)


def auto_invert_image_mask(arr: np.ndarray) -> np.ndarray:
    """Inverte a máscara binária se a maioria dos pixels for True (foreground).

    Garante que as máscaras tenham sempre fundo preto (0) e ROI branco (1).

    Args:
        arr: Máscara binária de entrada.

    Returns:
        Máscara possivelmente invertida.
    """
    img = arr.copy()

    if statistics.mode(img.flatten()):
        img = np.invert(img)

    return img


def find_bighest_cluster_area(clusters: np.ndarray) -> int:
    """Retorna a área (em pixels) do maior cluster rotulado.

    Args:
        clusters: Imagem com labels de regiões (saída de measure.label).

    Returns:
        Área em pixels do maior cluster.

    Raises:
        ValueError: Se não houver regiões na imagem.
    """
    regions = measure.regionprops(clusters)
    if not regions:
        raise ValueError("Nenhuma região encontrada na imagem de clusters.")
    return max(r.area for r in regions)


def find_bighest_cluster(img: np.ndarray) -> np.ndarray:
    """Mantém somente o maior cluster foreground de uma máscara binária.

    Args:
        img: Máscara binária de entrada.

    Returns:
        Máscara booleana contendo apenas o maior cluster.
    """
    clusters = auto_invert_image_mask(img)
    clusters = measure.label(clusters, background=0)
    cluster_size = find_bighest_cluster_area(clusters)

    return morphology.remove_small_objects(
        clusters.astype(bool),
        max_size=(cluster_size - 2),  # equivale ao antigo min_size=cluster_size - 1
        connectivity=8,
    )


def find_roi(img: np.ndarray) -> np.ndarray:
    """Detecta a região de interesse (ROI) em uma imagem grayscale.

    Aplica equalização de histograma, binarização e seleção do maior cluster.

    Args:
        img: Imagem grayscale de entrada.

    Returns:
        Máscara booleana da ROI.
    """
    binary_img = binarize_image(exposure.equalize_hist(img))
    best_cluster = find_bighest_cluster(binary_img)
    merged = binarize_image(binary_img + best_cluster)
    return binarize_image(find_bighest_cluster(merged))


def smoothing_mask_edges(mask: np.ndarray) -> np.ndarray:
    """Suaviza as bordas de uma máscara binária com filtro gaussiano.

    Args:
        mask: Máscara binária de entrada.

    Returns:
        Máscara binarizada após suavização gaussiana.
    """
    return binarize_image(filters.gaussian(mask, sigma=0.5))
