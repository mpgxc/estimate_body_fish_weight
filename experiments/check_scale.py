"""Confere se a escala das fotos (pixels por centímetro) é constante.

Compara o comprimento do corpo na máscara com o comprimento medido com
fita. Com a câmera fixa na tampa da caixa, a razão deveria ser a mesma em
todas as fotos.

Uso: ``uv run python -m experiments.check_scale``
"""

from __future__ import annotations

from fishweight.data import REPO_ROOT, load_manifest
from fishweight.features import WORK_SCALE, build_feature_table

OUTPUT = REPO_ROOT / "results/checks/escala.csv"
FLAG_ABOVE_PCT = 15.0


def main() -> None:
    table = build_feature_table(load_manifest())
    scale = table[["img", "session_group", "species"]].copy()
    scale["length_cm"] = table["length"]
    scale["length_px"] = table["feret_diameter_max"] * WORK_SCALE
    scale["px_per_cm"] = scale["length_px"] / scale["length_cm"]
    median = scale["px_per_cm"].median()
    scale["deviation_pct"] = (scale["px_per_cm"] / median - 1) * 100
    scale["flagged"] = scale["deviation_pct"].abs() > FLAG_ABOVE_PCT

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    scale.round({"length_px": 1, "px_per_cm": 2, "deviation_pct": 1}).to_csv(OUTPUT, index=False)

    print(f"mediana global: {median:.1f} px/cm; sinalizadas: {int(scale['flagged'].sum())}")
    by_session = scale.groupby("session_group")["px_per_cm"]
    summary = by_session.agg(n="size", mediana="median", cv=lambda s: s.std(ddof=0) / s.mean())
    summary["sinalizadas"] = scale.groupby("session_group")["flagged"].sum()
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
