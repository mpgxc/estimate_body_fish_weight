"""F1 e F2: o que o dataset público FDWE acrescenta.

* ``f2_<backbone>_coleta_para_fdwe``: treina nas 244 fotos da coleta e prevê
  os peixes de teste do FDWE, sem ajuste.
* ``f2_<backbone>_fdwe_para_coleta``: treina no FDWE (treino e validação) e
  prevê as 244 fotos da coleta, sem ajuste. As duas medem a distância entre
  os conjuntos: outras espécies, outra câmera, outra escala.
* ``f1_<backbone>_pre_fdwe``: parte da rede treinada no FDWE e faz o ajuste
  fino na coleta, nas mesmas dobras de E2. Responde se o pré-treino ajuda.

Só entram peixes do FDWE com oclusão de até ``--max-occlusion`` por cento:
acima disso boa parte do corpo está coberta e o peso não é observável.

O FDWE (2,4 GB, CC BY 4.0) é baixado para ``data/external/fdwe/`` se ainda
não estiver lá. Precisa de GPU.

Uso: ``python -m experiments.f_fdwe [--backbone NOME] [--protocols ...] [--epochs N]``
"""

from __future__ import annotations

import torch

from experiments.common import (
    FINE_TUNED,
    best_finetuned,
    is_done,
    run_once,
    save_cross_domain,
    training_arguments,
)
from fishweight.data import load_manifest
from fishweight.direct import direct_fit_predict, input_size, train_regressor
from fishweight.fdwe import download_fdwe, load_fdwe_crops, load_fdwe_manifest
from fishweight.photos import CACHE_DIR, load_photos
from fishweight.torch_utils import pick_device


def main() -> None:
    parser = training_arguments(__doc__.splitlines()[0])
    parser.add_argument("--backbone", choices=FINE_TUNED,
                        help="padrão: o melhor ajuste fino de E2 em P3, com o mesmo alvo")
    parser.add_argument("--max-occlusion", type=int, default=30, choices=(0, 30, 60, 90),
                        help="oclusão máxima, em %%, dos peixes do FDWE usados (padrão: 30)")
    args = parser.parse_args()
    backbone, log_target = (args.backbone, False) if args.backbone else best_finetuned()
    print(f"dispositivo: {pick_device()} | backbone: {backbone} | alvo em log: {log_target}", flush=True)
    size = input_size(backbone)
    train_kwargs = dict(log_target=log_target, epochs=args.epochs)

    own = load_manifest()
    photos = load_photos(own, size)
    fdwe = load_fdwe_manifest(download_fdwe())
    fdwe = fdwe[fdwe["occlusion"] <= args.max_occlusion].reset_index(drop=True)
    crops = load_fdwe_crops(fdwe, size)
    fdwe_test = (fdwe["split"] == "test").to_numpy()
    print(f"FDWE: {len(fdwe)} peixes com oclusão até {args.max_occlusion}% "
          f"({int(fdwe_test.sum())} de teste)", flush=True)

    # F2, coleta → FDWE
    name = f"f2_{backbone}_coleta_para_fdwe"
    if is_done(name, "cruzado"):
        print(f"{name}: já feito, pulando", flush=True)
    else:
        fitted = train_regressor(photos, own["weight"].to_numpy(), backbone, **train_kwargs)
        overall = save_cross_domain(
            name, fdwe.loc[fdwe_test, "fish_id"], fdwe.loc[fdwe_test, "weight"],
            fitted.predict(crops[fdwe_test]), train="coleta", test="fdwe",
        )["overall"]
        print(f"coleta → FDWE: MAE={overall['mae']:.1f} MAPE={overall['mape']:.1f}", flush=True)

    # Pré-treino no FDWE (guardado para não repetir) e F2, FDWE → coleta
    checkpoint = CACHE_DIR / f"fdwe_{backbone}{'_log' if log_target else ''}_oclusao{args.max_occlusion}.pt"
    if checkpoint.is_file():
        backbone_state = torch.load(checkpoint, map_location="cpu")
    else:
        fitted = train_regressor(
            crops[~fdwe_test], fdwe.loc[~fdwe_test, "weight"].to_numpy(), backbone, **train_kwargs
        )
        overall = save_cross_domain(
            f"f2_{backbone}_fdwe_para_coleta", own["img"], own["weight"], fitted.predict(photos),
            train="fdwe", test="coleta",
        )["overall"]
        print(f"FDWE → coleta: MAE={overall['mae']:.1f} MAPE={overall['mape']:.1f}", flush=True)
        backbone_state = {k: v.cpu() for k, v in fitted.model.backbone.state_dict().items()}
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(backbone_state, checkpoint)

    # F1: ajuste fino na coleta partindo da rede treinada no FDWE
    fit_predict = direct_fit_predict(
        photos, own["img"].tolist(), backbone, backbone_state=backbone_state, **train_kwargs
    )
    for protocol in args.protocols:
        run_once(f"f1_{backbone}_pre_fdwe", protocol, own, fit_predict, mask="nenhuma")


if __name__ == "__main__":
    main()
