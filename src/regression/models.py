"""Definição, avaliação e validação cruzada de modelos de regressão de peso."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


DEFAULT_MODELS: Dict = {
    "GradientBoostingRegressor": GradientBoostingRegressor(),
    "LinearRegression": LinearRegression(),
    "LassoRegression": Lasso(),
    "RidgeRegression": Ridge(),
    "RandomForestRegressor": RandomForestRegressor(),
    "PCARegression": make_pipeline(PCA(n_components=2), LinearRegression()),
    "PolyRegression": make_pipeline(PolynomialFeatures(degree=3), Lasso()),
}


def handle_model_evaluate(
    true: np.ndarray, predicted: np.ndarray, title_column: str = ""
) -> pd.DataFrame:
    """Calcula métricas de regressão e retorna um DataFrame formatado.

    Args:
        true: Valores reais.
        predicted: Valores preditos.
        title_column: Nome da coluna no DataFrame resultante.

    Returns:
        DataFrame com MAE, MSE, RMSE, MAPE e R² como linhas.
    """
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, predicted)
    mape = np.mean(np.abs((true - predicted) / true)) * 100

    return pd.DataFrame(
        {title_column: [mae, mse, rmse, mape, r2]},
        index=["MAE", "MSE", "RMSE", "MAPE", "R-squared"],
    )


def show_model_performance(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Exibe métricas de treino e teste para um modelo ajustado.

    Args:
        model: Modelo scikit-learn já treinado.
        X_train: Features de treino.
        y_train: Alvos de treino.
        X_test: Features de teste.
        y_test: Alvos de teste.
    """
    train = handle_model_evaluate(y_train, model.predict(X_train), "Treino")
    test = handle_model_evaluate(y_test, model.predict(X_test), "Teste")
    print(pd.concat([train, test], axis=1))


def run_all_algs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Optional[Dict] = None,
) -> None:
    """Treina e avalia múltiplos algoritmos de regressão.

    Args:
        X_train: Features de treino.
        y_train: Alvos de treino.
        X_test: Features de teste.
        y_test: Alvos de teste.
        models: Dicionário {nome: modelo}. Usa :data:`DEFAULT_MODELS` se None.
    """
    if models is None:
        models = DEFAULT_MODELS

    for name, model in models.items():
        model.fit(X_train, y_train)
        print("\n", name)
        show_model_performance(model, X_train, y_train, X_test, y_test)
        print("_" * 20)


def apply_kfold(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    models: Optional[Dict] = None,
) -> None:
    """Aplica validação cruzada K-Fold e exibe médias e desvios das métricas.

    Args:
        X: Features.
        y: Alvos.
        k: Número de folds.
        models: Dicionário {nome: modelo}. Usa :data:`DEFAULT_MODELS` se None.
    """
    if models is None:
        models = {
            k: v for k, v in DEFAULT_MODELS.items()
            if k != "PolyRegression"  # evita fuga de dados em pipeline poly
        }
        models["PolyRegression"] = make_pipeline(PolynomialFeatures(degree=2), Lasso())

    scoring = [
        "r2",
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "neg_root_mean_squared_error",
        "neg_mean_absolute_percentage_error",
    ]

    kfold = KFold(n_splits=k, shuffle=True, random_state=0)

    for name, model in models.items():
        cv_results = cross_validate(model, X, y, cv=kfold, scoring=scoring)
        print(f"[{name}]")
        print(
            pd.DataFrame(
                {
                    "Média": [
                        cv_results["test_r2"].mean(),
                        -cv_results["test_neg_mean_absolute_error"].mean(),
                        -cv_results["test_neg_mean_squared_error"].mean(),
                        -cv_results["test_neg_root_mean_squared_error"].mean(),
                        -cv_results["test_neg_mean_absolute_percentage_error"].mean(),
                    ],
                    "Desvio Padrão": [
                        cv_results["test_r2"].std(),
                        cv_results["test_neg_mean_absolute_error"].std(),
                        cv_results["test_neg_mean_squared_error"].std(),
                        cv_results["test_neg_root_mean_squared_error"].std(),
                        cv_results["test_neg_mean_absolute_percentage_error"].std(),
                    ],
                },
                index=["R-squared", "MAE", "MSE", "RMSE", "MAPE"],
            )
        )


def apply_pca_reduce(data: np.ndarray, n: Optional[int] = None) -> np.ndarray:
    """Reduz dimensionalidade via PCA preservando 90% da variância.

    Args:
        data: Array de features 2-D (n_samples, n_features).
        n: Número de componentes a manter. Se None, determina automaticamente
           o mínimo necessário para preservar 90% da variância.

    Returns:
        Array transformado com shape (n_samples, n_components).
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    pca_full = PCA(n_components=data.shape[-1]).fit(X)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    dim = int(np.argmax(cumsum >= 0.90)) + 1

    print(f"Para preservar 90% da variance é necessário {dim} componente(s).")

    n_components = n if n else dim
    pca_refit = PCA(n_components=n_components).fit(X)
    print("Ratio:", pca_refit.explained_variance_ratio_)

    return pca_refit.transform(X)


def corr_view(df: pd.DataFrame, w: int = 8, h: int = 6) -> pd.DataFrame:
    """Exibe mapa de calor da matriz de correlação.

    Args:
        df: DataFrame com as variáveis numéricas.
        w: Largura da figura em polegadas.
        h: Altura da figura em polegadas.

    Returns:
        Matriz de correlação calculada.
    """
    corrmat = df.corr()
    _, ax = plt.subplots(figsize=(w, h))
    sns.heatmap(corrmat, linewidths=5, annot=True)
    plt.show()
    return corrmat
