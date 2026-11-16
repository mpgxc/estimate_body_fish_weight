"""Gera as tabelas do relatório a partir dos ``metrics.json`` em ``results/``.

Uso: ``uv run python -m experiments.report``
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fishweight.data import REPO_ROOT, SUSPECT_WEIGHTS
from fishweight.metrics import regression_metrics

COMPARISON = REPO_ROOT / "docs/experiments/comparacao.md"
COMPARISON_HEADER = """# Comparação de métricas

Arquivo gerado por `uv run python -m experiments.report`. Não edite à mão.

Uma linha por técnica, ordenada pelo MAE em P3 (uma sessão de coleta fora do
treino), que é o protocolo que decide. A coluna **Máscara** diz de onde vêm as
medidas do teste: `manual` (anotada à mão), `predita` (gerada pela U-Net) ou
`nenhuma` (o modelo usa só a foto). A ferramenta não terá a máscara manual,
então só as duas últimas concorrem. `(ref.)` marca variantes que usam informação que a
ferramenta não terá; elas não concorrem. A descrição de cada técnica está em
[`tecnicas.md`](tecnicas.md).

"""
PROTOCOL_TITLES = {
    "p1": "P1 — reprodução (70/30 da monografia)",
    "p2": "P2 — aleatório estratificado (5 partes, 3 repetições)",
    "p3": "P3 — uma sessão de coleta de fora por vez",
}


def _num(value: float | None, digits: int = 1) -> str:
    return "–" if value is None else f"{value:.{digits}f}".replace(".", ",")


def _row(label: str, m: dict) -> str:
    return (
        f"| {label} | {m['n']} | {_num(m['mae'])} | {_num(m['rmse'])} | "
        f"{_num(m['mape'])} | {_num(m['r2'], 3)} | {_num(m['bias'])} |"
    )


def _group_table(title: str, groups: dict[str, dict]) -> list[str]:
    lines = [f"### {title}", "", "| Grupo | n | MAE (g) | RMSE (g) | MAPE (%) | R² | Viés (g) |",
             "|---|---|---|---|---|---|---|"]
    lines += [_row(name, m) for name, m in groups.items()]
    return lines + [""]


def build_tables(results_dir: Path) -> str:
    """Monta, em Markdown, as tabelas comparativas de todos os experimentos."""
    results_dir = Path(results_dir)
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(results_dir.glob("*/*/metrics.json"))]
    lines: list[str] = []

    for protocol, title in PROTOCOL_TITLES.items():
        ranked = sorted((r for r in runs if r["protocol"] == protocol), key=lambda r: r["overall"]["mae"])
        if not ranked:
            continue
        lines += [f"### {title}", "", "| Experimento | MAE (g) [IC 95%] | RMSE (g) | MAPE (%) | R² | Viés (g) |",
                  "|---|---|---|---|---|---|"]
        for run in ranked:
            o = run["overall"]
            name = run["experiment"] + (" (ref.)" if run["reference_only"] else "")
            lo, hi = o["mae_ci"]
            lines.append(
                f"| {name} | {_num(o['mae'])} [{_num(lo)}; {_num(hi)}] | {_num(o['rmse'])} | "
                f"{_num(o['mape'])} | {_num(o['r2'], 3)} | {_num(o['bias'])} |"
            )
        lines.append("")

    winner = _candidate(runs)
    if winner:
        lines += _group_table(f"Por espécie: {winner['experiment']}", winner["per_species_group"])
        if "outras" in winner["per_species_group"]:
            lines += ["`outras` reúne panga, traíra e lambari: amostra pequena, leitura apenas indicativa.", ""]
        lines += _group_table(f"Por sessão: {winner['experiment']}", winner["per_fold"])
        lines += _group_table(f"Por faixa de peso: {winner['experiment']}", winner["per_weight_band"])

    scale_file = results_dir / "checks/escala.csv"
    if scale_file.is_file():
        flagged = pd.read_csv(scale_file).query("flagged")
        lines += ["### Imagens com escala fora do esperado (desvio acima de 15%)", "",
                  "| Imagem | Sessão | Espécie | Comprimento (cm) | px/cm | Desvio (%) |", "|---|---|---|---|---|---|"]
        lines += [
            f"| {r.img} | {r.session_group} | {r.species} | {_num(r.length_cm)} | "
            f"{_num(r.px_per_cm)} | {_num(r.deviation_pct)} |"
            for r in flagged.itertuples()
        ]
        lines.append("")

    return "\n".join(lines)


def _load_runs(results_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(results_dir).glob("*/*/metrics.json"))]


def _candidate(runs: list[dict]) -> dict | None:
    """O melhor em P3 entre as rodadas que a ferramenta consegue reproduzir."""
    candidates = [r for r in runs if r["protocol"] == "p3" and not r["reference_only"]]
    # A ferramenta não terá a máscara manual: havendo rodadas sem ela, só essas concorrem.
    candidates = [r for r in candidates if r.get("mask", "manual") != "manual"] or candidates
    return min(candidates, key=lambda r: r["overall"]["mae"]) if candidates else None


def build_sensitivity(results_dir: Path) -> str:
    """Métricas do candidato em P3 com todos os peixes e sem os de peso suspeito."""
    lines = ["| Experimento | Peixes | n | MAE (g) | RMSE (g) | MAPE (%) | R² | Viés (g) |",
             "|---|---|---|---|---|---|---|---|"]
    winner = _candidate(_load_runs(results_dir))
    predictions_file = Path(results_dir) / winner["experiment"] / "p3/predictions.csv" if winner else None
    if predictions_file and predictions_file.is_file():
        predictions = pd.read_csv(predictions_file)
        subsets = {"todos": predictions, "sem pesos suspeitos": predictions[~predictions["img"].isin(SUSPECT_WEIGHTS)]}
        for label, rows in subsets.items():
            m = regression_metrics(rows["y_true"], rows["y_pred"])
            lines.append(
                f"| {winner['experiment']} | {label} | {m['n']} | {_num(m['mae'])} | {_num(m['rmse'])} | "
                f"{_num(m['mape'])} | {_num(m['r2'], 3)} | {_num(m['bias'])} |"
            )
    return "\n".join(lines) + "\n"


def build_comparison(results_dir: Path) -> str:
    """Monta uma tabela única: uma linha por técnica, os três protocolos lado a lado."""
    by_experiment: dict[str, dict[str, dict]] = {}
    for path in sorted(Path(results_dir).glob("*/*/metrics.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["protocol"] in PROTOCOL_TITLES:  # domínio cruzado tem tabela própria
            by_experiment.setdefault(run["experiment"], {})[run["protocol"]] = run

    def p3_mae(runs: dict[str, dict]) -> float:
        return runs["p3"]["overall"]["mae"] if "p3" in runs else float("inf")

    lines = ["| Técnica | Máscara | P1: MAE (g) | P2: MAE (g) | P3: MAE (g) [IC 95%] | P3: MAPE (%) | P3: R² | P3: viés (g) |",
             "|---|---|---|---|---|---|---|---|"]
    for name, runs in sorted(by_experiment.items(), key=lambda item: (p3_mae(item[1]), item[0])):
        first = next(iter(runs.values()))
        label = name + (" (ref.)" if first["reference_only"] else "")
        mae = {p: _num(runs[p]["overall"]["mae"]) if p in runs else "–" for p in PROTOCOL_TITLES}
        if "p3" in runs:
            o = runs["p3"]["overall"]
            lo, hi = o["mae_ci"]
            p3 = f"{mae['p3']} [{_num(lo)}; {_num(hi)}] | {_num(o['mape'])} | {_num(o['r2'], 3)} | {_num(o['bias'])}"
        else:
            p3 = "– | – | – | –"
        lines.append(f"| {label} | {first.get('mask', 'manual')} | {mae['p1']} | {mae['p2']} | {p3} |")
    return "\n".join(lines) + "\n"


def build_segmentation(results_dir: Path) -> str:
    """Resume Dice e IoU das máscaras previstas, por protocolo."""
    lines = ["| Protocolo | Máscaras | Dice médio | IoU médio | Dice mínimo |", "|---|---|---|---|---|"]
    for protocol in PROTOCOL_TITLES:
        file = Path(results_dir) / "segmentation" / protocol / "metrics.csv"
        if file.is_file():
            scores = pd.read_csv(file)
            lines.append(
                f"| {protocol.upper()} | {len(scores)} | {_num(scores['dice'].mean(), 3)} | "
                f"{_num(scores['iou'].mean(), 3)} | {_num(scores['dice'].min(), 3)} |"
            )
    return "\n".join(lines) + "\n"


def build_cross_domain(results_dir: Path) -> str:
    """Rodadas em que o modelo treina num conjunto e é testado em outro, sem ajuste."""
    lines = ["| Experimento | Treino | Teste | n | MAE (g) [IC 95%] | MAPE (%) | R² | Viés (g) |",
             "|---|---|---|---|---|---|---|---|"]
    for path in sorted(Path(results_dir).glob("*/cruzado/metrics.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        o = run["overall"]
        lo, hi = o["mae_ci"]
        lines.append(
            f"| {run['experiment']} | {run['train']} | {run['test']} | {o['n']} | "
            f"{_num(o['mae'])} [{_num(lo)}; {_num(hi)}] | {_num(o['mape'])} | {_num(o['r2'], 3)} | {_num(o['bias'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    results = REPO_ROOT / "results"
    text = (
        COMPARISON_HEADER
        + build_comparison(results)
        + "\n## Segmentação\n\nMáscara prevista contra a máscara manual, sempre por uma U-Net que não viu a foto.\n\n"
        + build_segmentation(results)
        + "\n## Domínio cruzado\n\nO modelo treina num conjunto (a coleta ou o FDWE) e é testado no outro, sem ajuste.\n\n"
        + build_cross_domain(results)
        + "\n## Sensibilidade a pesos suspeitos\n\nO melhor candidato em P3, com todos os peixes e sem os que têm peso anotado suspeito (`SUSPECT_WEIGHTS`). Os peixes continuam no treino; só saem da conta do erro.\n\n"
        + build_sensitivity(results)
        + "\n## Tabelas por protocolo\n\n"
        + build_tables(results)
    )
    COMPARISON.write_text(text, encoding="utf-8")
    print(f"comparação gravada em {COMPARISON.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
