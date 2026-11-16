"""E3: regressão direta com saídas auxiliares (multitarefa).

Parte do melhor ajuste fino de E2 em P3 e acrescenta, só na perda de treino:

* ``e3_<backbone>_especie``: classificação da espécie anotada;
* ``e3_<backbone>_medidas``: comprimento e largura medidos com fita.

Na predição só a foto entra, então as duas variantes são candidatas.

Uso: ``python -m experiments.e3_multitask [--backbone NOME] [--protocols ...] [--epochs N]``
"""

from __future__ import annotations

from experiments.common import FINE_TUNED, best_finetuned, run_once, training_arguments
from fishweight.data import load_manifest
from fishweight.direct import direct_fit_predict, input_size
from fishweight.photos import load_photos
from fishweight.torch_utils import pick_device

VARIANTS = {"especie": ("species",), "medidas": ("measures",)}


def main() -> None:
    parser = training_arguments(__doc__.splitlines()[0])
    parser.add_argument("--backbone", choices=FINE_TUNED,
                        help="padrão: o melhor ajuste fino de E2 em P3, com o mesmo alvo")
    args = parser.parse_args()
    backbone, log_target = (args.backbone, False) if args.backbone else best_finetuned()
    print(f"dispositivo: {pick_device()} | backbone: {backbone} | alvo em log: {log_target}", flush=True)

    manifest = load_manifest()
    photos = load_photos(manifest, input_size(backbone))
    for variant, aux in VARIANTS.items():
        fit_predict = direct_fit_predict(
            photos, manifest["img"].tolist(), backbone, aux=aux,
            log_target=log_target, epochs=args.epochs,
        )
        for protocol in args.protocols:
            run_once(f"e3_{backbone}_{variant}", protocol, manifest, fit_predict, mask="nenhuma")


if __name__ == "__main__":
    main()
