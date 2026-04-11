#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

from glob import glob
from typing import List, Optional, Tuple

import pandas as pd
from joblib import Parallel, delayed
from skimage.io import imread
from skimage.measure import label, regionprops_table
from skimage.transform import resize
from tqdm import tqdm

ALL_REGION_PROPS_PROPERTIES = [
    "area",
    "bbox",
    "bbox_area",
    "centroid",
    "convex_area",
    "eccentricity",
    "equivalent_diameter",
    "euler_number",
    "extent",
    "filled_area",
    "label",
    "major_axis_length",
    "minor_axis_length",
    "moments",
    "moments_central",
    "orientation",
    "perimeter",
    "feret_diameter_max",
    "solidity",
]

def extract_features_from_images(
    images_arr: Optional[List] = None,
    images_path: Optional[str] = None,
    output_file_name: str = "features.csv",
    resize_shape: Tuple[int, int] = (224, 224),
    properties: List[str] = ALL_REGION_PROPS_PROPERTIES,
) -> None:
    """Extrai propriedades de regiões de máscaras binárias e salva em CSV.

    Args:
        images_arr: Array/lista de máscaras pré-carregadas (opcional).
            Se fornecido junto com ``images_path``, cada máscara é pareada
            com o caminho correspondente para gerar o label.
        images_path: Caminho para o diretório contendo as imagens, usado
            quando ``images_arr`` não for fornecido.
        output_file_name: Nome do arquivo CSV de saída.
        resize_shape: Dimensão (altura, largura) para redimensionamento.
        properties: Lista de propriedades de região a extrair via
            ``skimage.measure.regionprops_table``.

    Returns:
        None
    """
    df_output_props = pd.DataFrame()

    if images_arr is not None and np.asarray(images_arr).size > 0:
        rows = []
        for path, mask in tqdm(zip(images_path, images_arr)):
            try:
                props = pd.DataFrame(
                    regionprops_table(label(mask), properties=properties)
                )
                props["label"] = path.split("/")[-1]
                rows.append(props)
            except Exception as image_error:
                print("Erro na imagem: ", image_error)
                continue

        if rows:
            df_output_props = pd.concat(rows, ignore_index=True)

    else:

        def make_row(image_path: str) -> pd.DataFrame:
            image = imread(image_path)
            mask = resize(image, resize_shape, anti_aliasing=True) > 0.5
            props = pd.DataFrame(
                regionprops_table(label(mask), properties=properties)
            )
            props["label"] = image_path.split("/")[-1]
            return props

        try:
            results = Parallel(n_jobs=-2)(
                delayed(make_row)(image_path)
                for image_path in tqdm(glob(f"{images_path}/*"))
            )
            valid = [r for r in results if r is not None]
            if valid:
                df_output_props = pd.concat(valid, ignore_index=True)
        except Exception as image_error:
            print("Erro na imagem:", image_error)

    df_output_props.to_csv(output_file_name, index=None)
