"""E1: regressores sobre atributos ``regionprops`` da máscara manual.

Três variantes por modelo:

* ``e1_nativo_<Modelo>``: 12 atributos de forma, P1 a P3.
* ``e1_nativo_especie_oraculo_<Modelo>``: os mesmos atributos mais a espécie
  anotada. Referência: a ferramenta não terá a espécie.
* ``e1_monografia_<Modelo>``: reprodução do notebook
  ``notebooks/04_regression/regression_last_execution.ipynb``, só em P1.
  Referência: a seleção de atributos usa o conjunto inteiro e inclui a
  espécie anotada.

Uso: ``uv run python -m experiments.e1_regionprops``
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from fishweight.data import DATA_DIR, REPO_ROOT, load_manifest
from fishweight.experiment import FitPredict, run_experiment
from fishweight.features import FEATURE_NAMES, build_feature_table
from fishweight.metrics import bootstrap_ci, regression_metrics
from fishweight.regressors import make_models
from fishweight.splits import PROTOCOLS, load_splits

# Tabela 6 da monografia (sem normalização), conjunto de teste: (R², MAE).
MONOGRAPH_TABLE_6 = {
    "GradientBoostingRegressor": (0.8832, 33.79),
    "LinearRegression": (0.8594, 41.85),
    "LassoRegression": (0.8545, 39.65),
    "RidgeRegression": (0.8599, 41.68),
    "RandomForestRegressor": (0.8810, 34.98),
    "PCARegression": (0.7687, 69.55),
    "PolyRegression": (0.9124, 31.92),
}


def native(model_name: str, with_species: bool) -> FitPredict:
    """Modelo novo a cada dobra, sobre os 12 atributos (e a espécie, se pedida)."""
    columns = FEATURE_NAMES + (["species"] if with_species else [])

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        model = make_models(max_iter=200_000)[model_name]
        if with_species:
            encode = make_column_transformer(
                (OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["species"]),
                remainder="passthrough",
            )
            model = make_pipeline(encode, model)
        return model.fit(train[columns], train["weight"]).predict(test[columns])

    return fit_predict


def run_native() -> None:
    table = build_feature_table(load_manifest())
    for model_name in make_models():
        for prefix, with_species in (("e1_nativo", False), ("e1_nativo_especie_oraculo", True)):
            for protocol in PROTOCOLS:
                overall = run_experiment(
                    f"{prefix}_{model_name}",
                    protocol,
                    table,
                    native(model_name, with_species),
                    reference_only=with_species,
                )["overall"]
                print(
                    f"{prefix}_{model_name} {protocol} MAE={overall['mae']:.1f} "
                    f"MAPE={overall['mape']:.1f} R2={overall['r2']:.3f}"
                )


def run_monograph() -> None:
    """Refaz o procedimento do notebook, com as 245 linhas e os CSVs originais."""
    annotations = pd.read_csv(DATA_DIR / "annotations.csv")
    features = pd.read_csv(REPO_ROOT / "combined_features.csv")
    data = annotations.merge(features, left_on="img", right_on="label", validate="one_to_one")
    data = data.drop(columns="label")
    data["specie"] = LabelEncoder().fit_transform(data["specie"])

    # Como no notebook: correlação com o peso calculada no conjunto inteiro.
    correlation = data.drop(columns="img").corr()["weight"]
    selected = [c for c in correlation.index[correlation > 0.85] if c != "weight"] + ["specie"]
    print("e1_monografia selected_features:", selected)

    in_test = data["img"].isin(load_splits("p1")[0]["test"])
    train, test = data[~in_test], data[in_test]
    print(f"{'modelo':28} {'R2':>7} {'R2 monog.':>10} {'MAE':>7} {'MAE monog.':>11}")
    for model_name, model in make_models(scale=False).items():
        y_pred = model.fit(train[selected], train["weight"]).predict(test[selected])
        overall = regression_metrics(test["weight"], y_pred)
        for metric in ("mae", "mape"):
            overall[f"{metric}_ci"] = list(bootstrap_ci(test["weight"], y_pred, metric))
        out_dir = REPO_ROOT / "results" / f"e1_monografia_{model_name}" / "p1"
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "experiment": f"e1_monografia_{model_name}",
            "protocol": "p1",
            "reference_only": True,
            "overall": overall,
            "selected_features": selected,
        }
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        r2_ref, mae_ref = MONOGRAPH_TABLE_6[model_name]
        print(
            f"{model_name:28} {overall['r2']:7.4f} {r2_ref:10.4f} "
            f"{overall['mae']:7.2f} {mae_ref:11.2f}"
        )


if __name__ == "__main__":
    run_monograph()
    run_native()
