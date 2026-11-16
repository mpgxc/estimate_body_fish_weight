"""Leitor do FDWE (Fish Detection and Weight Estimation), dataset público de Wageningen.

Sokolova et al. (2023), DOI 10.4121/a6d5a40e-0358-47cf-9ec1-335df0e4a3c3,
licença CC BY 4.0. São 1.086 imagens 1280×1280 de peixes numa esteira, com
2.216 peixes anotados: espécie, caixa delimitadora, peso e nível de oclusão.

Formato de cada linha de rótulo (YOLO com duas colunas a mais)::

    <espécie 0-8> <cx> <cy> <largura> <altura> <peso em kg> <oclusão>

As caixas são frações da imagem; a oclusão é uma faixa em porcentagem
(``00_00``, ``01_30``, ``31_60`` ou ``61_90``).
"""

from __future__ import annotations

import hashlib
import io
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from fishweight.data import REPO_ROOT
from fishweight.photos import CACHE_DIR, cached_arrays

FDWE_DIR = REPO_ROOT / "data/external/fdwe"  # fora do Git
ZIP_NAME = "Fish_Detection_and_Weight_Estimation_dataset.zip"
DOWNLOAD_URL = (
    "https://data.4tu.nl/file/a6d5a40e-0358-47cf-9ec1-335df0e4a3c3/"
    "643f0aa6-c230-404c-93c9-2af134155660"
)
ZIP_MD5 = "570dca2347b0739e23b085f14c9c1bee"

# Ordem do arquivo train_val_YOLOv5_weight_regression.yaml que acompanha o dataset.
SPECIES = [
    "plaice", "common_sole", "brill", "grey_gurnard", "turbot",
    "dab", "starry_ray", "whiting", "lesser_spotted_dogfish",
]
OCCLUSION = {"00_00": 0, "01_30": 30, "31_60": 60, "61_90": 90}  # limite superior da faixa, em %

# Tela em que cada peixe é colado antes de reduzir (linhas, colunas; 3:4). É a
# menor tela 3:4 que comporta a maior caixa do dataset, que deitada mede
# 1045×650 px. Tela fixa mantém a escala entre peixes: um peixe maior ocupa
# mais pixels, como nas fotos da coleta.
CANVAS = (792, 1056)
BACKGROUND = 128


def download_fdwe(root: Path = FDWE_DIR) -> Path:
    """Baixa o zip do FDWE (2,4 GB) se ainda não estiver em ``root`` e confere o MD5."""
    target = Path(root) / ZIP_NAME
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".parcial")
        print(f"baixando FDWE (2,4 GB) para {target}", flush=True)
        urllib.request.urlretrieve(DOWNLOAD_URL, partial)
        digest = hashlib.md5()
        with open(partial, "rb") as file:
            for block in iter(lambda: file.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != ZIP_MD5:
            raise ValueError(f"{partial}: MD5 {digest.hexdigest()}, esperado {ZIP_MD5}; apague e baixe de novo")
        partial.rename(target)
    return target


def parse_labels(text: str, image: str) -> list[dict]:
    """Converte o arquivo de rótulos de uma imagem numa lista de peixes (peso em gramas).

    Raises:
        ValueError: Linha fora do formato; a mensagem cita a imagem.
    """
    fish = []
    for number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        try:
            species_id, cx, cy, w, h, weight_kg, occlusion = parts
            fish.append({
                "species": SPECIES[int(species_id)],
                "cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h),
                "weight": float(weight_kg) * 1000,
                "occlusion": OCCLUSION[occlusion],
            })
        except (ValueError, KeyError, IndexError) as error:
            raise ValueError(f"{image}, linha {number}: rótulo fora do formato ({line!r})") from error
    return fish


def load_fdwe_manifest(zip_path: Path = FDWE_DIR / ZIP_NAME) -> pd.DataFrame:
    """Uma linha por peixe do FDWE, lida direto do zip (sem extrair).

    Returns:
        DataFrame com ``fish_id, image, member, split, session, species, cx,
        cy, w, h, weight, occlusion``. ``split`` é a divisão oficial do
        dataset (train, val, test); ``session`` é data e local, tirados do
        nome do arquivo; ``member`` é o caminho da imagem dentro do zip.
    """
    rows = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in sorted(archive.namelist()):
            match = re.search(r"/labels/(train|val|test)/([^/]+)\.txt$", member)
            if not match:
                continue
            split, image = match.groups()
            session = re.sub(r"_\d+_rgb$", "", image)
            image_member = member.replace("/labels/", "/images/")[:-4] + ".png"
            for index, fish in enumerate(parse_labels(archive.read(member).decode(), image)):
                rows.append({
                    "fish_id": f"{image}_{index}", "image": image, "member": image_member,
                    "split": split, "session": session, **fish,
                })
    return pd.DataFrame(rows)


def crop_fish(
    image: np.ndarray, cx: float, cy: float, w: float, h: float, size: tuple[int, int]
) -> np.ndarray:
    """Recorta um peixe pela caixa, deita-o e cola no centro de uma tela fixa.

    A tela (:data:`CANVAS`) é reduzida para ``size``, que deve ter a mesma
    proporção 3:4. O peixe nunca é ampliado para preencher o quadro.

    Raises:
        ValueError: A caixa não cabe na tela.
    """
    rows, cols = image.shape[:2]
    top, bottom = round((cy - h / 2) * rows), round((cy + h / 2) * rows)
    left, right = round((cx - w / 2) * cols), round((cx + w / 2) * cols)
    crop = image[max(top, 0):min(bottom, rows), max(left, 0):min(right, cols)]
    if crop.shape[0] > crop.shape[1]:
        crop = np.rot90(crop)
    if crop.shape[0] > CANVAS[0] or crop.shape[1] > CANVAS[1]:
        raise ValueError(f"Caixa de {crop.shape[0]}×{crop.shape[1]} px não cabe na tela {CANVAS}")
    canvas = np.full((*CANVAS, 3), BACKGROUND, dtype=np.uint8)
    y0, x0 = (CANVAS[0] - crop.shape[0]) // 2, (CANVAS[1] - crop.shape[1]) // 2
    canvas[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    small = Image.fromarray(canvas).resize((size[1], size[0]), Image.Resampling.BILINEAR, reducing_gap=2.0)
    return np.asarray(small, dtype=np.uint8)


def load_fdwe_crops(
    manifest: pd.DataFrame,
    size: tuple[int, int],
    zip_path: Path = FDWE_DIR / ZIP_NAME,
    cache_dir: Path = CACHE_DIR,
) -> np.ndarray:
    """Recortes ``uint8`` N×linhas×colunas×3, na ordem do manifesto, com cache em disco."""

    def build() -> dict[str, np.ndarray]:
        crops = []
        with zipfile.ZipFile(zip_path) as archive:
            # Cada imagem é decodificada uma vez e serve a todos os seus peixes.
            for member, fish in manifest.groupby("member", sort=False):
                image = np.asarray(Image.open(io.BytesIO(archive.read(member))).convert("RGB"))
                for row in fish.itertuples():
                    crops.append((row.Index, crop_fish(image, row.cx, row.cy, row.w, row.h, size)))
        order = {index: position for position, index in enumerate(manifest.index)}
        crops.sort(key=lambda item: order[item[0]])
        return {"crops": np.stack([crop for _, crop in crops])}

    cache_file = Path(cache_dir) / f"fdwe_{size[0]}x{size[1]}.npz"
    return cached_arrays(cache_file, manifest["fish_id"].to_numpy(dtype=str), build)["crops"]
