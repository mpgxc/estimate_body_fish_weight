"""Executor: roda um modelo sobre as dobras fixas e grava predições e métricas."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from fishweight.data import REPO_ROOT
from fishweight.metrics import bootstrap_ci, regression_metrics, weight_band
from fishweight.splits import load_splits

# Recebe as linhas de treino e as de teste (sem ``weight``) e devolve as
# predições na ordem do teste.
FitPredict = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]

_PREDICTION_COLUMNS = ["img", "fold", "species", "species_group", "session_group", "y_true", "y_pred"]


def _metrics(group: pd.DataFrame) -> dict:
    values = regression_metrics(group["y_true"], group["y_pred"])
    # R² não existe quando o grupo tem um único peso; JSON não aceita NaN.
    return {k: None if isinstance(v, float) and math.isnan(v) else v for k, v in values.items()}


def run_experiment(
    name: str,
    protocol: str,
    table: pd.DataFrame,
    fit_predict: FitPredict,
    results_dir: Path = REPO_ROOT / "results",
    reference_only: bool = False,
) -> dict:
    """Avalia ``fit_predict`` em todas as dobras de ``protocol``.

    Grava ``<results_dir>/<name>/<protocol>/predictions.csv`` e
    ``metrics.json`` e devolve o conteúdo deste último.

    Raises:
        ValueError: A tabela não cobre exatamente as imagens das dobras, ou
            o modelo devolveu predições de tamanho errado ou não finitas.
    """
    folds = load_splits(protocol)
    in_splits = {img for fold in folds for img in fold["train"] + fold["test"]}
    in_table = set(table["img"])
    if in_splits != in_table:
        missing = sorted(in_splits - in_table)[:5]
        extra = sorted(in_table - in_splits)[:5]
        raise ValueError(
            f"Tabela e divisão {protocol} não cobrem as mesmas imagens: "
            f"faltam na tabela {missing}; sobram na tabela {extra}"
        )

    rows = table.set_index("img", drop=False)
    bands = weight_band(rows["weight"])
    parts = []
    for fold in folds:
        train = rows.loc[fold["train"]].reset_index(drop=True)
        test = rows.loc[fold["test"]].reset_index(drop=True)
        y_pred = np.asarray(fit_predict(train, test.drop(columns="weight")), dtype=float)
        if y_pred.shape != (len(test),):
            raise ValueError(
                f"dobra {fold['name']}: {y_pred.shape} predições para {len(test)} peixes"
            )
        if not np.isfinite(y_pred).all():
            raise ValueError(f"dobra {fold['name']}: predições não finitas")
        part = test[["img", "species", "species_group", "session_group"]].copy()
        part.insert(1, "fold", fold["name"])
        part["y_true"] = test["weight"].to_numpy(dtype=float)
        part["y_pred"] = y_pred
        parts.append(part)

    predictions = pd.concat(parts, ignore_index=True)[_PREDICTION_COLUMNS]
    band_of = predictions["img"].map(bands)

    overall = _metrics(predictions)
    for metric in ("mae", "mape"):
        overall[f"{metric}_ci"] = list(
            bootstrap_ci(predictions["y_true"], predictions["y_pred"], metric)
        )
    metrics = {
        "experiment": name,
        "protocol": protocol,
        "reference_only": reference_only,
        "overall": overall,
        "per_fold": {k: _metrics(g) for k, g in predictions.groupby("fold")},
        "per_species_group": {k: _metrics(g) for k, g in predictions.groupby("species_group")},
        "per_weight_band": {k: _metrics(g) for k, g in predictions.groupby(band_of)},
    }

    out_dir = Path(results_dir) / name / protocol
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metrics
