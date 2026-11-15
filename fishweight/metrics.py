"""Métricas de regressão de peso, com intervalo de confiança por bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Calcula n, MAE, RMSE, MAPE (%), R² e viés (média de predito − real).

    Raises:
        ValueError: Tamanhos diferentes ou peso real não positivo.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Tamanhos diferentes: {y_true.shape} e {y_pred.shape}")
    if (y_true <= 0).any():
        raise ValueError("Peso real não positivo: o MAPE não é definido")

    error = y_pred - y_true
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "n": int(y_true.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": float(np.mean(np.abs(error) / y_true) * 100),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "bias": float(np.mean(error)),
    }


def bootstrap_ci(
    y_true,
    y_pred,
    metric: str,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Intervalo de confiança por percentis, reamostrando pares (real, predito)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, y_true.size, size=(n_boot, y_true.size))
    values = [regression_metrics(y_true[i], y_pred[i])[metric] for i in idx]
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(values, [tail, 100 - tail])
    return float(lo), float(hi)


def weight_band(weights: pd.Series) -> pd.Series:
    """Classifica cada peso em ``baixo``, ``medio`` ou ``alto`` (tercis)."""
    return pd.qcut(weights, 3, labels=["baixo", "medio", "alto"]).astype(str)
