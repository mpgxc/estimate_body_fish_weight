"""E4 e E5: híbrido, a foto e os atributos da máscara juntos.

* ``e4_<backbone>_congelado_<ridge|gbr>``: vetor da rede congelada mais os 12
  atributos da máscara, num regressor simples. Barato; usa os vetores que
  ``experiments.e2_direct`` gravou e roda com máscara manual e prevista.
* ``e4_<backbone>_fusao``: ajuste fino com um ramo para os atributos da
  máscara, fundido ao vetor da foto antes da saída.
* ``e5_<backbone>_fusao_especie``: o mesmo, com a espécie como saída auxiliar.

As fusões treinam com os atributos da máscara manual e são testadas com os
da máscara prevista, que é o que a ferramenta terá. Precisam de GPU e das
máscaras previstas (``experiments.segmentation``).

Uso: ``python -m experiments.e4_hybrid [--backbone NOME] [--protocols ...] [--epochs N]``
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor

from experiments.common import (
    FINE_TUNED,
    best_finetuned,
    run_both_masks,
    run_once,
    sklearn_fit_predict,
    training_arguments,
)
from experiments.e2_direct import make_ridge
from fishweight.data import load_manifest
from fishweight.direct import BACKBONES, add_embeddings, direct_fit_predict, embedding_file, input_size
from fishweight.features import (
    FEATURE_NAMES,
    build_feature_table,
    load_predicted_features,
    predicted_features_file,
    with_predicted_test_features,
)
from fishweight.photos import load_photos
from fishweight.torch_utils import pick_device

FROZEN_MODELS = {"ridge": make_ridge, "gbr": lambda: GradientBoostingRegressor(random_state=0)}
FUSIONS = {"e4_{}_fusao": (), "e5_{}_fusao_especie": ("species",)}


def run_frozen(table) -> None:
    for backbone in BACKBONES:
        if not embedding_file(backbone).is_file():
            print(f"e4 {backbone}: sem vetores; rode experiments.e2_direct")
            continue
        with_embeddings, columns = add_embeddings(table, embedding_file(backbone))
        for model_name, make_model in FROZEN_MODELS.items():
            run_both_masks(
                f"e4_{backbone}_congelado_{model_name}",
                with_embeddings,
                sklearn_fit_predict(make_model, columns + FEATURE_NAMES),
                reference_only=False,
            )


def run_fusion(table, backbone: str, log_target: bool, protocols, epochs: int) -> None:
    photos = load_photos(table, input_size(backbone))
    for pattern, aux in FUSIONS.items():
        name = pattern.format(backbone)
        fit_predict = direct_fit_predict(
            photos, table["img"].tolist(), backbone, aux=aux,
            tabular_columns=FEATURE_NAMES, log_target=log_target, epochs=epochs,
        )
        for protocol in protocols:
            if not predicted_features_file(protocol).is_file():
                print(f"{name} {protocol}: sem máscaras previstas; rode experiments.segmentation")
                continue
            predicted = with_predicted_test_features(fit_predict, load_predicted_features(protocol))
            run_once(name, protocol, table, predicted, mask="predita")


def main() -> None:
    parser = training_arguments(__doc__.splitlines()[0])
    parser.add_argument("--backbone", choices=FINE_TUNED,
                        help="padrão: o melhor ajuste fino de E2 em P3, com o mesmo alvo")
    parser.add_argument("--only-frozen", action="store_true",
                        help="roda só as variantes baratas, sem treinar rede")
    args = parser.parse_args()
    table = build_feature_table(load_manifest())
    run_frozen(table)
    if args.only_frozen:
        return
    backbone, log_target = (args.backbone, False) if args.backbone else best_finetuned()
    print(f"dispositivo: {pick_device()} | backbone: {backbone} | alvo em log: {log_target}", flush=True)
    run_fusion(table, backbone, log_target, args.protocols, args.epochs)


if __name__ == "__main__":
    main()
