"""Piso alométrico: peso como potência da área da máscara, ``W = a·Áreaᵇ``."""

from __future__ import annotations

from typing import Self

import numpy as np


def _fit_log_log(area: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    b, log_a = np.polyfit(np.log(area), np.log(weight), 1)
    return float(np.exp(log_a)), float(b)


class AllometricRegressor:
    """Ajusta ``log(W) = log(a) + b·log(Área)`` por mínimos quadrados.

    Com ``species``, cada espécie com pelo menos ``min_species_samples``
    exemplos de treino ganha seu par ``(a, b)``; as demais, e as nunca
    vistas, usam o par global.
    """

    def __init__(self, min_species_samples: int = 5) -> None:
        self.min_species_samples = min_species_samples

    def fit(self, area, weight, species=None) -> Self:
        area = np.asarray(area, dtype=float)
        weight = np.asarray(weight, dtype=float)
        if (area <= 0).any() or (weight <= 0).any():
            raise ValueError("Área e peso precisam ser positivos")

        self.a_, self.b_ = _fit_log_log(area, weight)
        self.per_species_: dict[str, tuple[float, float]] = {}
        if species is not None:
            species = np.asarray(species)
            for name in np.unique(species):
                rows = species == name
                if rows.sum() >= self.min_species_samples:
                    self.per_species_[str(name)] = _fit_log_log(area[rows], weight[rows])
        return self

    def predict(self, area, species=None) -> np.ndarray:
        area = np.asarray(area, dtype=float)
        if species is None:
            return self.a_ * area**self.b_
        coefficients = [self.per_species_.get(str(s), (self.a_, self.b_)) for s in species]
        a, b = np.array(coefficients).T
        return a * area**b
