"""Manifesto do conjunto de dados: uma linha por peixe."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = REPO_ROOT / "data/coletas/combined"

# Registros fora de todos os experimentos; annotations.csv não é editado.
EXCLUDED_IMAGES: dict[str, str] = {
    "IMG_20221126_163700.jpg": "peso anotado 0,60 g; os outros lambaris têm 50 e 60 g",
}

# Pesos que destoam dos peixes de mesmo tamanho, sem prova de erro. Ficam no
# conjunto (tirar um peixe só porque o modelo erra seria escolher o resultado);
# o relatório mostra as métricas com e sem eles.
SUSPECT_WEIGHTS: dict[str, str] = {
    "IMG_20220110_152533.jpg": "curimatã de 33 cm com 1.760 g; os de 34 a 37 cm pesam de 1.100 a 1.385 g",
}

# Medidas de fita que não batem com a máscara. O peso desses peixes é usado
# normalmente; o comprimento e a largura não servem de alvo nem de escala.
UNRELIABLE_MEASURES: dict[str, str] = {
    "IMG_20220110_150640.jpg": "comprimento anotado de 8 cm; a máscara indica cerca de 18 cm",
    "IMG_20220110_150758.jpg": "comprimento anotado de 7 cm; a máscara indica cerca de 16 cm",
}

_MAIN_SPECIES = {"tambaqui", "tilápia", "curimatã"}
_EARLY_SESSIONS = {"20220110", "20220212"}


def _session_group(session: str) -> str:
    if session in _EARLY_SESSIONS:
        return "2022-inicio"
    return f"{session[:4]}-{session[4:6]}-{session[6:]}"


def load_manifest(base_path: Path = DATA_DIR) -> pd.DataFrame:
    """Lê as anotações e devolve uma linha por peixe, ordenada por imagem.

    Args:
        base_path: Diretório com ``annotations.csv``, ``images/`` e ``masks/``.

    Returns:
        DataFrame com ``img, species, species_group, weight, width, length,
        session, session_group, image_path, mask_path``.

    Raises:
        ValueError: Imagem duplicada, arquivo ausente ou peso não positivo.
    """
    base_path = Path(base_path)
    df = pd.read_csv(base_path / "annotations.csv").rename(columns={"specie": "species"})

    duplicated = df.loc[df["img"].duplicated(), "img"].tolist()
    if duplicated:
        raise ValueError(f"Imagem duplicada em annotations.csv: {duplicated[0]}")

    df = df[~df["img"].isin(EXCLUDED_IMAGES)].sort_values("img").reset_index(drop=True)
    df["species_group"] = df["species"].where(df["species"].isin(_MAIN_SPECIES), "outras")
    df["session"] = df["img"].str.slice(4, 12)
    df["session_group"] = df["session"].map(_session_group)
    df["image_path"] = [str(base_path / "images" / img) for img in df["img"]]
    df["mask_path"] = [str(base_path / "masks" / img) for img in df["img"]]

    for column in ("image_path", "mask_path"):
        for img, path in zip(df["img"], df[column]):
            if not Path(path).is_file():
                raise ValueError(f"Arquivo ausente para {img}: {path}")

    bad_weight = df.loc[df["weight"] <= 0, "img"].tolist()
    if bad_weight:
        raise ValueError(f"Peso não positivo: {bad_weight[0]}")

    return df[["img", "species", "species_group", "weight", "width", "length",
               "session", "session_group", "image_path", "mask_path"]]
