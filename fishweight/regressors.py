"""Os sete regressores comparados na monografia."""

from __future__ import annotations

from sklearn.base import RegressorMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def make_models(
    random_state: int = 0, scale: bool = True, max_iter: int = 1000
) -> dict[str, RegressorMixin]:
    """Devolve instâncias novas dos sete modelos, prontas para ``fit``.

    Args:
        random_state: Semente dos modelos baseados em árvores.
        scale: Se verdadeiro, cada modelo começa com ``StandardScaler``,
            ajustado apenas nos dados passados a ``fit``.
        max_iter: Limite de iterações do Lasso. O padrão do scikit-learn
            (1000) reproduz a monografia, mas não converge nesses dados.
    """
    models: dict[str, RegressorMixin] = {
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=random_state),
        "LinearRegression": LinearRegression(),
        "LassoRegression": Lasso(max_iter=max_iter),
        "RidgeRegression": Ridge(),
        "RandomForestRegressor": RandomForestRegressor(random_state=random_state),
        "PCARegression": make_pipeline(PCA(n_components=2), LinearRegression()),
        "PolyRegression": make_pipeline(PolynomialFeatures(degree=3), Lasso(max_iter=max_iter)),
    }
    if scale:
        return {name: make_pipeline(StandardScaler(), model) for name, model in models.items()}
    return models
