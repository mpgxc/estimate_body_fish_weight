"""Gera as tabelas do relatório a partir dos ``metrics.json`` em ``results/``.

Uso: ``uv run python -m experiments.report``
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fishweight.data import REPO_ROOT

REPORT = REPO_ROOT / "docs/experiments/relatorio-fase1a.md"
COMPARISON = REPO_ROOT / "docs/experiments/comparacao.md"
COMPARISON_HEADER = """# Comparação de métricas

Arquivo gerado por `uv run python -m experiments.report`. Não edite à mão.

Uma linha por técnica, ordenada pelo MAE em P3 (uma sessão de coleta fora do
treino), que é o protocolo que decide. `(ref.)` marca variantes que usam
informação que a ferramenta não terá; elas não concorrem. A descrição de cada
técnica está em [`tecnicas.md`](tecnicas.md).

"""
BEGIN, END = "<!-- tabelas:início -->", "<!-- tabelas:fim -->"
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

    candidates = [r for r in runs if r["protocol"] == "p3" and not r["reference_only"]]
    if candidates:
        winner = min(candidates, key=lambda r: r["overall"]["mae"])
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


def build_comparison(results_dir: Path) -> str:
    """Monta uma tabela única: uma linha por técnica, os três protocolos lado a lado."""
    by_experiment: dict[str, dict[str, dict]] = {}
    for path in sorted(Path(results_dir).glob("*/*/metrics.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        by_experiment.setdefault(run["experiment"], {})[run["protocol"]] = run

    def p3_mae(runs: dict[str, dict]) -> float:
        return runs["p3"]["overall"]["mae"] if "p3" in runs else float("inf")

    lines = ["| Técnica | P1: MAE (g) | P2: MAE (g) | P3: MAE (g) [IC 95%] | P3: MAPE (%) | P3: R² | P3: viés (g) |",
             "|---|---|---|---|---|---|---|"]
    for name, runs in sorted(by_experiment.items(), key=lambda item: (p3_mae(item[1]), item[0])):
        label = name + (" (ref.)" if next(iter(runs.values()))["reference_only"] else "")
        mae = {p: _num(runs[p]["overall"]["mae"]) if p in runs else "–" for p in PROTOCOL_TITLES}
        if "p3" in runs:
            o = runs["p3"]["overall"]
            lo, hi = o["mae_ci"]
            p3 = f"{mae['p3']} [{_num(lo)}; {_num(hi)}] | {_num(o['mape'])} | {_num(o['r2'], 3)} | {_num(o['bias'])}"
        else:
            p3 = "– | – | – | –"
        lines.append(f"| {label} | {mae['p1']} | {mae['p2']} | {p3} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    tables = build_tables(REPO_ROOT / "results")
    text = REPORT.read_text(encoding="utf-8") if REPORT.is_file() else f"# Relatório da Fase 1A\n\n{BEGIN}\n{END}\n"
    head, rest = text.split(BEGIN, 1)
    tail = rest.split(END, 1)[1]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(f"{head}{BEGIN}\n\n{tables}\n{END}{tail}", encoding="utf-8")
    print(f"tabelas gravadas em {REPORT.relative_to(REPO_ROOT)}")
    COMPARISON.write_text(COMPARISON_HEADER + build_comparison(REPO_ROOT / "results"), encoding="utf-8")
    print(f"comparação gravada em {COMPARISON.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
