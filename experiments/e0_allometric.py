"""E0: piso alométrico sobre a área da máscara manual, em P1, P2 e P3.

Uso: ``uv run python -m experiments.e0_allometric``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fishweight.allometric import AllometricRegressor
from fishweight.data import load_manifest
from fishweight.experiment import run_experiment
from fishweight.features import build_feature_table
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
        for protocol in PROTOCOLS:
            overall = run_experiment(
                name, protocol, table, fit_predict, reference_only=reference_only
            )["overall"]
            print(
                f"{name} {protocol} MAE={overall['mae']:.1f} "
                f"MAPE={overall['mape']:.1f} R2={overall['r2']:.3f}"
            )


if __name__ == "__main__":
    main()
