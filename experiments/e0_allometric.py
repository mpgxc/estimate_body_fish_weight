"""E0: piso alométrico sobre a área da máscara manual, em P1, P2 e P3.

Uso: ``uv run python -m experiments.e0_allometric``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.common import run_and_report, run_both_masks
from fishweight.allometric import AllometricRegressor
from fishweight.data import load_manifest
from fishweight.features import build_feature_table, calibrate_features, session_scale
from fishweight.splits import PROTOCOLS


def fit_global(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return AllometricRegressor().fit(train["area"], train["weight"]).predict(test["area"])


def fit_species_oracle(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    model = AllometricRegressor().fit(train["area"], train["weight"], train["species"])
    return model.predict(test["area"], test["species"])


# nome → (função, usa informação que a ferramenta não terá)
VARIANTS = {
    "e0_global": (fit_global, False),
    "e0_especie_oraculo": (fit_species_oracle, True),
}


def main() -> None:
    table = build_feature_table(load_manifest())
    for name, (fit_predict, reference_only) in VARIANTS.items():
        run_both_masks(name, table, fit_predict, reference_only)

    # Quanto uma referência de escala na caixa ajudaria: a área entra em cm², com a
    # escala de cada sessão estimada pela fita. Referência, porque usa a fita.
    calibrated = calibrate_features(table, session_scale(table))
    for protocol in PROTOCOLS:
        run_and_report("e0_global_escala_sessao", protocol, calibrated, fit_global, reference_only=True)


if __name__ == "__main__":
    main()
