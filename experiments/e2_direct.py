"""E2: regressão direta, a rede prevê o peso a partir da foto (sem máscara).

Duas famílias por backbone:

* ``e2_<backbone>_congelado_<ridge|mlp>``: a rede pré-treinada só gera um
  vetor por foto (gravado em ``data/features/embeddings_<backbone>.npz``) e
  um regressor simples é ajustado sobre os vetores. Barato: roda nos três
  protocolos.
* ``e2_<backbone>_ajuste`` e ``..._ajuste_log``: ajuste fino da rede inteira,
  com o peso em gramas ou em ``log``. Caro: precisa de GPU e, por padrão,
  roda só em P1 e P3.

Uso: ``python -m experiments.e2_direct [--backbones ...] [--protocols ...] [--epochs N]``
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.common import (
    FINE_TUNED,
    run_and_report,
    run_once,
    sklearn_fit_predict,
    training_arguments,
)
from fishweight.data import load_manifest
from fishweight.direct import (
    BACKBONES,
    add_embeddings,
    direct_fit_predict,
    embed,
    embedding_file,
    input_size,
)
from fishweight.photos import load_photos
from fishweight.splits import PROTOCOLS
from fishweight.torch_utils import pick_device


def make_ridge():
    # O alfa é escolhido por validação cruzada interna, só com o treino da dobra.
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))


def make_mlp():
    # early_stopping separa 10% do treino da dobra para decidir quando parar.
    network = MLPRegressor(hidden_layer_sizes=(256,), early_stopping=True, max_iter=2000, random_state=0)
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), network), transformer=StandardScaler()
    )


FROZEN_MODELS = {"ridge": make_ridge, "mlp": make_mlp}


def ensure_embeddings(backbone: str, manifest) -> None:
    """Calcula e grava os vetores do backbone, se ainda não existirem."""
    file = embedding_file(backbone)
    if file.is_file():
        return
    photos = load_photos(manifest, input_size(backbone))
    file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        file, names=manifest["img"].to_numpy(dtype=str), embeddings=embed(photos, backbone)
    )
    print(f"vetores de {backbone} gravados em {file.name}", flush=True)


def main() -> None:
    parser = training_arguments(__doc__.splitlines()[0])
    parser.add_argument("--backbones", nargs="+", choices=list(BACKBONES), default=list(BACKBONES))
    args = parser.parse_args()
    print(f"dispositivo: {pick_device()}", flush=True)
    manifest = load_manifest()
    names = manifest["img"].tolist()

    for backbone in args.backbones:
        ensure_embeddings(backbone, manifest)
        table, columns = add_embeddings(manifest, embedding_file(backbone))
        for model_name, make_model in FROZEN_MODELS.items():
            for protocol in PROTOCOLS:
                run_and_report(
                    f"e2_{backbone}_congelado_{model_name}",
                    protocol,
                    table,
                    sklearn_fit_predict(make_model, columns),
                    mask="nenhuma",
                )

    for backbone in args.backbones:
        if backbone not in FINE_TUNED:
            continue
        photos = load_photos(manifest, input_size(backbone))
        for suffix, log_target in (("", False), ("_log", True)):
            fit_predict = direct_fit_predict(
                photos, names, backbone, log_target=log_target, epochs=args.epochs
            )
            for protocol in args.protocols:
                run_once(f"e2_{backbone}_ajuste{suffix}", protocol, manifest, fit_predict, mask="nenhuma")


if __name__ == "__main__":
    main()
