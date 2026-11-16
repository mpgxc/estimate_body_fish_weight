"""Rotinas compartilhadas pelos scripts de experimento."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from fishweight.data import REPO_ROOT
from fishweight.direct import BACKBONES
from fishweight.experiment import FitPredict, run_experiment
from fishweight.metrics import bootstrap_ci, regression_metrics
from fishweight.features import (
    load_predicted_features,
    predicted_features_file,
    with_predicted_test_features,
)
from fishweight.splits import PROTOCOLS

RESULTS = REPO_ROOT / "results"
FINE_TUNED = ("resnet50", "efficientnet_b3", "convnext_tiny")  # backbones com ajuste fino


def run_and_report(name: str, protocol: str, table: pd.DataFrame, fit_predict: FitPredict, **kwargs) -> None:
    """Roda o experimento e imprime uma linha com as métricas gerais."""
    overall = run_experiment(name, protocol, table, fit_predict, **kwargs)["overall"]
    print(
        f"{name} {protocol} MAE={overall['mae']:.1f} "
        f"MAPE={overall['mape']:.1f} R2={overall['r2']:.3f}",
        flush=True,
    )


def sklearn_fit_predict(make_model: Callable, columns: Sequence[str]) -> FitPredict:
    """``fit_predict`` de um modelo do scikit-learn, novo a cada dobra, sobre ``columns``."""
    columns = list(columns)

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        return make_model().fit(train[columns], train["weight"]).predict(test[columns])

    return fit_predict


def run_both_masks(name: str, table: pd.DataFrame, fit_predict: FitPredict, reference_only: bool) -> None:
    """Roda ``fit_predict`` em P1, P2 e P3, com máscara manual e com máscara prevista.

    Nos dois casos o treino usa a máscara manual; ``<name>_mascara_predita``
    troca os atributos do teste pelos da máscara que a U-Net previu. As
    máscaras previstas vêm de ``python -m experiments.segmentation``, que
    roda num ambiente com GPU; o protocolo que ainda não as tiver é avisado
    e fica só com a máscara manual.
    """
    for protocol in PROTOCOLS:
        runs = [(name, fit_predict, "manual")]
        if predicted_features_file(protocol).is_file():
            predicted = with_predicted_test_features(fit_predict, load_predicted_features(protocol))
            runs.append((f"{name}_mascara_predita", predicted, "predita"))
        else:
            print(f"{name} {protocol}: sem máscaras previstas; rode experiments.segmentation")
        for run_name, fit, mask in runs:
            run_and_report(run_name, protocol, table, fit, reference_only=reference_only, mask=mask)


def is_done(name: str, protocol: str, results_dir: Path = RESULTS) -> bool:
    """Diz se a rodada já tem ``metrics.json``; para refazer, apague a pasta dela."""
    return (Path(results_dir) / name / protocol / "metrics.json").is_file()


def run_once(
    name: str,
    protocol: str,
    table: pd.DataFrame,
    fit_predict: FitPredict,
    results_dir: Path = RESULTS,
    **kwargs,
) -> None:
    """Roda um experimento caro, a menos que já esteja feito (ver :func:`is_done`)."""
    if is_done(name, protocol, results_dir):
        print(f"{name} {protocol}: já feito, pulando", flush=True)
        return
    run_and_report(name, protocol, table, fit_predict, results_dir=results_dir, **kwargs)


def save_cross_domain(
    name: str,
    ids: Sequence[str],
    y_true: Sequence[float],
    y_pred: Sequence[float],
    train: str,
    test: str,
    results_dir: Path = RESULTS,
) -> dict:
    """Grava o resultado de treinar num conjunto e testar em outro, sem ajuste.

    Fica em ``<results_dir>/<name>/cruzado/``, fora das colunas P1–P3.
    """
    overall = regression_metrics(y_true, y_pred)
    for metric in ("mae", "mape"):
        overall[f"{metric}_ci"] = list(bootstrap_ci(y_true, y_pred, metric))
    metrics = {
        "experiment": name, "protocol": "cruzado", "reference_only": False, "mask": "nenhuma",
        "train": train, "test": test, "overall": overall,
    }
    out_dir = Path(results_dir) / name / "cruzado"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": list(ids), "y_true": list(y_true), "y_pred": list(y_pred)}).to_csv(
        out_dir / "predictions.csv", index=False
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metrics


def best_finetuned(results_dir: Path = RESULTS) -> tuple[str, bool]:
    """Backbone de ajuste fino com menor MAE em P3 e se o alvo era ``log(peso)``.

    Raises:
        ValueError: Nenhum resultado de ajuste fino em P3.
    """
    found = []
    for backbone in BACKBONES:
        for suffix, log_target in (("", False), ("_log", True)):
            file = Path(results_dir) / f"e2_{backbone}_ajuste{suffix}" / "p3" / "metrics.json"
            if file.is_file():
                mae = json.loads(file.read_text(encoding="utf-8"))["overall"]["mae"]
                found.append((mae, backbone, log_target))
    if not found:
        raise ValueError(
            "Nenhum ajuste fino com resultado em P3; rode experiments.e2_direct antes "
            "ou informe --backbone"
        )
    _, backbone, log_target = min(found)
    return backbone, log_target


def training_arguments(description: str, protocols: Sequence[str] = ("p1", "p3")) -> argparse.ArgumentParser:
    """Argumentos comuns aos scripts que treinam redes.

    O padrão deixa P2 de fora: são 15 das 21 dobras e não decidem o vencedor.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(protocols),
                        help="protocolos a rodar (padrão: %(default)s)")
    parser.add_argument("--epochs", type=int, default=30, help="épocas de ajuste fino (padrão: 30)")
    return parser
