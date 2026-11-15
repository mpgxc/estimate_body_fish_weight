# Fase 1A — Base e pisos: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Montar a base reprodutível dos experimentos (projeto, dados, divisões, métricas, executor) e medir os dois pisos, E0 alométrico e E1 `regionprops`, com máscara manual nos protocolos P1, P2 e P3.

**Architecture:** O pacote `fishweight/` (ex-`src/`) ganha módulos pequenos e independentes: manifesto de dados, divisões, métricas, atributos da máscara e um executor que roda qualquer modelo sobre as mesmas divisões e grava predições e métricas. Cada experimento é um script em `experiments/` que entrega ao executor uma função `fit_predict`. Um script de relatório lê `results/` e gera as tabelas.

**Tech Stack:** Python 3.12, `uv`, pandas, scikit-image, scikit-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-11-15-evolucao-fishweight-design.md`

## Global Constraints

- Python `>=3.12,<3.13`, gerenciado com `uv`. Todo comando roda da raiz do repositório com `uv run`.
- O lambari `IMG_20221126_163700.jpg` (peso anotado 0,60 g) é excluído por lista no código. `annotations.csv` não é editado. Restam 244 peixes.
- Nenhuma escolha (atributo, hiperparâmetro, normalização) usa dados de teste. A única exceção é a variante `e1_monografia`, que reproduz de propósito o procedimento antigo e é rotulada como tal.
- Variantes que recebem a espécie anotada ou medidas de fita são **referência**, nunca candidatas a vencedor.
- Escala preservada: máscaras são reduzidas por um fator inteiro, sem esticar.
- `random_state=0` em tudo que for aleatório.
- Identificadores em inglês, docstrings e mensagens em português, `from __future__ import annotations` e type hints.
- Não tocar em: `notebooks/`, `commons.py` e `feature_extraction.py` da raiz, `handcraft_segmentation/`, `yolo_v8_instance_seg/`, `models/`, `combined_features.csv`, `features_from_unet_predictions.csv`.

## Review Focus

1. **Espécie do teste ausente no treino** (acontece em P3 com lambari e traíra): o modelo por espécie usa os coeficientes globais em vez de falhar. Teste na Task 7.
2. **Máscara com manchas soltas ou ruído de JPEG**: só o maior componente é medido. Teste na Task 5.
3. **Divisão desatualizada** (imagem na divisão que não está mais na tabela, ou o contrário): o executor falha com mensagem clara em vez de descartar linhas em silêncio. Teste na Task 6.
4. **Comando rodado fora da raiz**: os caminhos partem da raiz do repositório, não do diretório corrente. Teste na Task 2.
5. **Máscara com resolução diferente de 4624×3472**: recusada com erro que cita o arquivo e o tamanho encontrado. Teste na Task 5.

## File Structure

| Arquivo | Responsabilidade |
|---|---|
| `pyproject.toml` | Projeto `uv`, dependências, configuração do pytest |
| `fishweight/commons.py`, `fishweight/segmentation/handcraft.py` | Utilitários herdados de `src/` |
| `fishweight/data.py` | Manifesto: uma linha por peixe, com espécie, sessão e caminhos |
| `fishweight/splits.py` | Constrói, grava e lê P1, P2 e P3 |
| `fishweight/metrics.py` | Métricas, intervalo por bootstrap, faixas de peso |
| `fishweight/features.py` | Atributos de forma da máscara na geometria nativa |
| `fishweight/experiment.py` | Executor: divisões → `fit_predict` → `predictions.csv` e `metrics.json` |
| `fishweight/allometric.py` | Regressor `W = a·Áreaᵇ` |
| `fishweight/regressors.py` | Fábrica dos sete regressores da monografia |
| `experiments/e0_allometric.py`, `experiments/e1_regionprops.py` | Scripts dos experimentos |
| `experiments/check_scale.py`, `experiments/report.py` | Conferência de escala e relatório |
| `data/splits/p1.json`, `p2.json`, `p3.json` | Divisões versionadas |
| `data/features/manual.csv` | Atributos das máscaras manuais (versionado) |

---

### Task 1: Projeto `uv` e pacote `fishweight`

**Files:**
- Create: `pyproject.toml`, `tests/test_commons.py`
- Move: `src/` → `fishweight/`; `requirements.txt` → `notebooks/requirements-legado.txt`
- Delete: `fishweight/feature_extraction.py`, `fishweight/regression/`, `fishweight/segmentation/unet.py`, `fishweight/segmentation/yolo.py` (substituídos nas Tasks 5 e 8 e no Plano 1B; as cópias da raiz e os notebooks continuam como registro)
- Modify: `fishweight/segmentation/handcraft.py:15` (`from src.commons` → `from fishweight.commons`), `.gitignore` (acrescentar `data/external/`)

**Interfaces:**
- Produces: pacote importável `fishweight`; comando `uv run pytest`.

- [ ] **Step 1: Mover e apagar**

```bash
git mv src fishweight
git mv requirements.txt notebooks/requirements-legado.txt
git rm -r fishweight/feature_extraction.py fishweight/regression fishweight/segmentation/unet.py fishweight/segmentation/yolo.py
```

- [ ] **Step 2: Criar `pyproject.toml`**

```toml
[project]
name = "fishweight"
version = "0.1.0"
description = "Estimativa de peso de peixes a partir de uma foto"
requires-python = ">=3.12,<3.13"
dependencies = [
  "numpy",
  "pandas>=2.0",
  "scikit-image>=0.22",
  "scikit-learn>=1.4",
  "scipy",
  "joblib",
  "tqdm",
  "matplotlib",
]

[dependency-groups]
dev = ["pytest"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["fishweight"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Escrever o teste**

```python
# tests/test_commons.py
import numpy as np

from fishweight.commons import find_bighest_cluster
from fishweight.segmentation import handcraft  # noqa: F401  (o import precisa funcionar)


def test_find_bighest_cluster_keeps_only_largest_blob():
    mask = np.zeros((50, 50), dtype=bool)
    mask[5:15, 5:15] = True    # 100 px
    mask[40:43, 40:43] = True  # 9 px
    result = find_bighest_cluster(mask)
    assert result.sum() == 100
    assert not result[40:43, 40:43].any()
```

- [ ] **Step 4: Corrigir o import em `handcraft.py`, acrescentar `data/external/` ao `.gitignore`, instalar e rodar**

Run: `uv sync && uv run pytest tests/test_commons.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "chore: projeto uv e pacote fishweight"
```

---

### Task 2: Manifesto de dados

**Files:**
- Create: `fishweight/data.py`, `tests/test_data.py`

**Interfaces:**
- Produces:
  - `REPO_ROOT: Path` (raiz do repositório, derivada de `__file__`)
  - `DATA_DIR: Path = REPO_ROOT / "data/coletas/combined"`
  - `EXCLUDED_IMAGES: dict[str, str] = {"IMG_20221126_163700.jpg": "peso anotado 0,60 g; os outros lambaris têm 50 e 60 g"}`
  - `load_manifest(base_path: Path = DATA_DIR) -> pd.DataFrame`, ordenado por `img`, índice 0..n-1, colunas: `img, species, species_group, weight, width, length, session, session_group, image_path, mask_path`
- Regras:
  - `species` vem da coluna `specie` do CSV.
  - `species_group`: `tambaqui`, `tilápia` e `curimatã` ficam como estão; as demais viram `outras`.
  - `session`: os 8 dígitos da data no nome (`IMG_20221126_...` → `20221126`).
  - `session_group`: `20220110` e `20220212` → `2022-inicio`; as demais no formato `2022-10-29`.
  - `ValueError` com o nome do arquivo quando há `img` duplicada, imagem ou máscara ausente, ou `weight <= 0` depois das exclusões.

- [ ] **Step 1: Escrever os testes**

```python
# tests/test_data.py
import pandas as pd
import pytest

from fishweight.data import EXCLUDED_IMAGES, load_manifest


def test_manifest_counts():
    m = load_manifest()
    assert len(m) == 244
    assert not set(EXCLUDED_IMAGES) & set(m["img"])
    assert m["species_group"].value_counts().to_dict() == {
        "tambaqui": 148, "tilápia": 61, "curimatã": 20, "outras": 15}
    assert m["session_group"].value_counts().to_dict() == {
        "2022-11-29": 79, "2022-11-26": 65, "2022-11-22": 48, "2022-10-29": 25, "2022-inicio": 27}
    assert m["img"].is_monotonic_increasing and m["weight"].min() == 50


def test_manifest_works_from_another_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert len(load_manifest()) == 244


def _dataset(tmp_path, rows):
    (tmp_path / "images").mkdir(); (tmp_path / "masks").mkdir()
    for img, *_ in rows:
        (tmp_path / "images" / img).touch(); (tmp_path / "masks" / img).touch()
    pd.DataFrame(rows, columns=["img", "specie", "weight", "width", "length"]).to_csv(
        tmp_path / "annotations.csv", index=False)
    return tmp_path


def test_duplicate_image_is_rejected(tmp_path):
    base = _dataset(tmp_path, [("IMG_20221029_000001.jpg", "panga", 100, 5, 20)] * 2)
    with pytest.raises(ValueError, match="IMG_20221029_000001.jpg"):
        load_manifest(base)


def test_missing_mask_is_rejected(tmp_path):
    base = _dataset(tmp_path, [("IMG_20221029_000001.jpg", "panga", 100, 5, 20)])
    (base / "masks" / "IMG_20221029_000001.jpg").unlink()
    with pytest.raises(ValueError, match="IMG_20221029_000001.jpg"):
        load_manifest(base)


def test_non_positive_weight_is_rejected(tmp_path):
    base = _dataset(tmp_path, [("IMG_20221029_000001.jpg", "panga", 0, 5, 20)])
    with pytest.raises(ValueError, match="IMG_20221029_000001.jpg"):
        load_manifest(base)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL com `ModuleNotFoundError: fishweight.data`

- [ ] **Step 3: Implementar `fishweight/data.py` conforme o bloco Interfaces**

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_data.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add fishweight/data.py tests/test_data.py && git commit -m "feat: manifesto de dados com 244 peixes"
```

---

### Task 3: Divisões P1, P2 e P3

**Files:**
- Create: `fishweight/splits.py`, `tests/test_splits.py`, `data/splits/p1.json`, `data/splits/p2.json`, `data/splits/p3.json`

**Interfaces:**
- Consumes: `load_manifest`, `REPO_ROOT`.
- Produces:
  - `Fold = TypedDict("Fold", {"name": str, "train": list[str], "test": list[str]})` (listas de `img`, ordenadas)
  - `PROTOCOLS = ("p1", "p2", "p3")`
  - `build_splits(manifest: pd.DataFrame) -> dict[str, list[Fold]]`
  - `load_splits(protocol: str) -> list[Fold]` (lê `data/splits/<protocol>.json`)
  - `python -m fishweight.splits` grava os três arquivos (JSON com `indent=1`, `ensure_ascii=False`)
- Regras:
  - **p1**: uma dobra `"p1"`. Teste = as imagens da coluna `label` de `features_from_unet_predictions.csv` (são as 74 do teste da monografia; nenhuma é a excluída). Treino = as outras 170.
  - **p2**: `RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)` sobre `species_group`, na ordem do manifesto. 15 dobras nomeadas `r0f0` … `r2f4`.
  - **p3**: uma dobra por `session_group`, com o nome do grupo; teste = o grupo, treino = o resto. 5 dobras.

- [ ] **Step 1: Escrever os testes**

```python
# tests/test_splits.py
from fishweight.data import load_manifest
from fishweight.splits import PROTOCOLS, build_splits, load_splits

M = load_manifest()
ALL = set(M["img"])
S = build_splits(M)


def test_no_fold_shares_images_between_train_and_test():
    for protocol in PROTOCOLS:
        for fold in S[protocol]:
            assert not set(fold["train"]) & set(fold["test"])
            assert set(fold["train"]) | set(fold["test"]) == ALL


def test_p1_is_the_monograph_test_set():
    (fold,) = S["p1"]
    assert (len(fold["train"]), len(fold["test"])) == (170, 74)
    assert "IMG_20221122_175014.jpg" in fold["test"]


def test_p2_each_repeat_tests_every_fish_once():
    assert [f["name"] for f in S["p2"]] == [f"r{r}f{k}" for r in range(3) for k in range(5)]
    for r in range(3):
        tested = [img for f in S["p2"][r * 5:(r + 1) * 5] for img in f["test"]]
        assert sorted(tested) == sorted(ALL)


def test_p3_holds_out_one_session_group_per_fold():
    session_of = dict(zip(M["img"], M["session_group"]))
    assert sorted(f["name"] for f in S["p3"]) == sorted(set(session_of.values()))
    for fold in S["p3"]:
        assert {session_of[i] for i in fold["test"]} == {fold["name"]}
        assert fold["name"] not in {session_of[i] for i in fold["train"]}


def test_committed_files_match_a_fresh_build():
    for protocol in PROTOCOLS:
        assert load_splits(protocol) == S[protocol]
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_splits.py -v`
Expected: FAIL com `ModuleNotFoundError: fishweight.splits`

- [ ] **Step 3: Implementar `fishweight/splits.py` e gerar os arquivos**

Run: `uv run python -m fishweight.splits`
Expected: imprime `p1: 1 dobras`, `p2: 15 dobras`, `p3: 5 dobras`

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_splits.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add fishweight/splits.py tests/test_splits.py data/splits && git commit -m "feat: divisões P1, P2 e P3 versionadas"
```

`data/splits/` fica fora da regra `data/coletas` do `.gitignore`, então o `git add` funciona sem `-f`.

---

### Task 4: Métricas

**Files:**
- Create: `fishweight/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `regression_metrics(y_true, y_pred) -> dict[str, float]` com chaves `n, mae, rmse, mape, r2, bias`. `mape` em %, `bias = média(y_pred − y_true)`. `ValueError` se algum `y_true <= 0` ou se os tamanhos diferem.
  - `bootstrap_ci(y_true, y_pred, metric: str, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float]` (percentis, reamostrando pares)
  - `weight_band(weights: pd.Series) -> pd.Series` com rótulos `baixo`, `medio`, `alto` (tercis via `pd.qcut`)

- [ ] **Step 1: Escrever os testes**

```python
# tests/test_metrics.py
import numpy as np
import pandas as pd
import pytest

from fishweight.metrics import bootstrap_ci, regression_metrics, weight_band

Y_TRUE, Y_PRED = [100.0, 200.0, 400.0], [110.0, 180.0, 400.0]


def test_metrics_exact_values():
    m = regression_metrics(Y_TRUE, Y_PRED)
    assert m["n"] == 3
    assert m["mae"] == pytest.approx(10.0)
    assert m["rmse"] == pytest.approx(12.909944)
    assert m["mape"] == pytest.approx(6.666667)
    assert m["bias"] == pytest.approx(-3.333333)
    assert m["r2"] == pytest.approx(0.9892857)


def test_non_positive_true_weight_is_rejected():
    with pytest.raises(ValueError):
        regression_metrics([0.0, 100.0], [1.0, 100.0])


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        regression_metrics([100.0, 200.0], [100.0])


def test_bootstrap_ci_brackets_the_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    y = rng.uniform(100, 1000, 200)
    p = y + rng.normal(0, 30, 200)
    lo, hi = bootstrap_ci(y, p, "mae")
    assert lo < regression_metrics(y, p)["mae"] < hi
    assert (lo, hi) == bootstrap_ci(y, p, "mae")


def test_weight_band_splits_into_terciles():
    bands = weight_band(pd.Series(range(1, 10)))
    assert bands.tolist() == ["baixo"] * 3 + ["medio"] * 3 + ["alto"] * 3


```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL com `ModuleNotFoundError: fishweight.metrics`

- [ ] **Step 3: Implementar `fishweight/metrics.py`**

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add fishweight/metrics.py tests/test_metrics.py && git commit -m "feat: métricas de regressão com intervalo por bootstrap"
```

---

### Task 5: Atributos da máscara na geometria nativa

**Files:**
- Create: `fishweight/features.py`, `tests/test_features.py`, `data/features/manual.csv`

**Interfaces:**
- Consumes: `load_manifest`, `REPO_ROOT`.
- Produces:
  - `EXPECTED_SHAPE = (3472, 4624)` (linhas, colunas) e `WORK_SCALE = 8` (máscara de trabalho: 434×578)
  - `FEATURE_NAMES = ["area", "area_bbox", "area_convex", "area_filled", "axis_major_length", "axis_minor_length", "eccentricity", "equivalent_diameter_area", "extent", "feret_diameter_max", "perimeter", "solidity"]`
  - `load_mask(path: Path) -> np.ndarray` (booleana, 434×578). Lê em tons de cinza, limiar `> 127`, reduz com `skimage.measure.block_reduce(mask, (8, 8), np.mean) > 0.5`. Se a forma não for `EXPECTED_SHAPE`, `ValueError(f"{path}: máscara {linhas}×{colunas}, esperado 3472×4624")`.
  - `mask_features(mask: np.ndarray) -> dict[str, float]` com as chaves de `FEATURE_NAMES`, medidas no maior componente conexo. `ValueError` se a máscara estiver vazia.
  - `build_feature_table(manifest: pd.DataFrame, cache_path: Path = REPO_ROOT / "data/features/manual.csv") -> pd.DataFrame`: o manifesto mais as 12 colunas. Usa o cache se as `img` do arquivo forem as do manifesto; senão recalcula (joblib, `n_jobs=-2`) e grava.
  - `python -m fishweight.features` gera o cache.

São 12 atributos de tamanho e forma. Posição, orientação e momentos brutos ficam de fora porque dependem de onde o peixe foi colocado na caixa.

- [ ] **Step 1: Escrever os testes**

```python
# tests/test_features.py
import numpy as np
import pytest
from skimage.io import imsave

from fishweight.data import load_manifest
from fishweight.features import FEATURE_NAMES, build_feature_table, load_mask, mask_features


def _write_mask(path, shape=(3472, 4624), boxes=((1000, 1800, 1000, 2600),)):
    full = np.zeros(shape, dtype=np.uint8)
    for r0, r1, c0, c1 in boxes:
        full[r0:r1, c0:c1] = 255
    imsave(path, full, check_contrast=False)
    return path


def test_rectangle_is_measured_at_one_eighth_scale(tmp_path):
    mask = load_mask(_write_mask(tmp_path / "m.png"))  # retângulo de 800×1600 px
    assert mask.shape == (434, 578)
    f = mask_features(mask)
    assert list(f) == FEATURE_NAMES
    assert f["area"] == 100 * 200
    assert f["area_bbox"] == 100 * 200
    assert f["extent"] == pytest.approx(1.0)
    assert f["solidity"] == pytest.approx(1.0)


def test_only_the_largest_component_is_measured(tmp_path):
    boxes = ((1000, 1800, 1000, 2600), (100, 180, 100, 180))  # peixe + mancha de 80×80
    f = mask_features(load_mask(_write_mask(tmp_path / "m.png", boxes=boxes)))
    assert f["area"] == 100 * 200


def test_wrong_resolution_is_rejected_with_file_and_size(tmp_path):
    path = _write_mask(tmp_path / "small.png", shape=(1080, 1920), boxes=((100, 200, 100, 300),))
    with pytest.raises(ValueError, match=r"small\.png.*1080.*1920"):
        load_mask(path)


def test_empty_mask_is_rejected():
    with pytest.raises(ValueError):
        mask_features(np.zeros((434, 578), dtype=bool))


def test_feature_table_covers_every_fish(tmp_path):
    table = build_feature_table(load_manifest(), cache_path=tmp_path / "manual.csv")
    assert len(table) == 244
    assert not table[FEATURE_NAMES].isna().any().any()
    assert (table["area"] > 0).all()
    cached = build_feature_table(load_manifest(), cache_path=tmp_path / "manual.csv")
    assert cached[FEATURE_NAMES].equals(table[FEATURE_NAMES])
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_features.py -v`
Expected: FAIL com `ModuleNotFoundError: fishweight.features`

- [ ] **Step 3: Implementar `fishweight/features.py` e gerar o cache**

Run: `uv run python -m fishweight.features`
Expected: imprime `244 máscaras → data/features/manual.csv`

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_features.py -v`
Expected: 5 passed (o último leva cerca de um minuto: lê 244 máscaras em resolução cheia)

- [ ] **Step 5: Commit**

```bash
git add fishweight/features.py tests/test_features.py data/features && git commit -m "feat: atributos de forma na geometria nativa"
```

---

### Task 6: Executor de experimentos

**Files:**
- Create: `fishweight/experiment.py`, `tests/test_experiment.py`

**Interfaces:**
- Consumes: `load_splits`, `Fold`, `regression_metrics`, `bootstrap_ci`, `weight_band`.
- Produces:
  - `FitPredict = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]`: recebe as linhas de treino e as de teste **sem a coluna `weight`** e devolve as predições na ordem do teste.
  - `run_experiment(name: str, protocol: str, table: pd.DataFrame, fit_predict: FitPredict, results_dir: Path = REPO_ROOT / "results", reference_only: bool = False) -> dict`
- Comportamento:
  - `table` precisa ter exatamente as `img` que aparecem nas dobras do protocolo. Qualquer diferença gera `ValueError` listando até cinco imagens que sobram ou faltam.
  - Predições com tamanho diferente do teste, ou com `NaN`, geram `ValueError` cuja mensagem começa com `dobra <nome>:`.
  - Grava `results/<name>/<protocol>/predictions.csv` com `img, fold, species, species_group, session_group, y_true, y_pred`.
  - Grava e devolve `metrics.json`:

```json
{
  "experiment": "e0_global", "protocol": "p3", "reference_only": false,
  "overall": {"n": 244, "mae": 0, "rmse": 0, "mape": 0, "r2": 0, "bias": 0,
              "mae_ci": [0, 0], "mape_ci": [0, 0]},
  "per_fold": {"2022-11-29": {"n": 79, "mae": 0, "rmse": 0, "mape": 0, "r2": 0, "bias": 0}},
  "per_species_group": {"tambaqui": {}},
  "per_weight_band": {"baixo": {}}
}
```

  - `overall` junta todas as predições de teste do protocolo (em P2 cada peixe aparece três vezes). `per_weight_band` usa os tercis do peso real da tabela inteira.

- [ ] **Step 1: Escrever os testes**

```python
# tests/test_experiment.py
import json

import numpy as np
import pandas as pd
import pytest

from fishweight import experiment
from fishweight.experiment import run_experiment

FOLDS = [{"name": "a", "train": ["3", "4", "5", "6"], "test": ["1", "2"]},
         {"name": "b", "train": ["1", "2", "5", "6"], "test": ["3", "4"]},
         {"name": "c", "train": ["1", "2", "3", "4"], "test": ["5", "6"]}]


@pytest.fixture
def table(monkeypatch):
    monkeypatch.setattr(experiment, "load_splits", lambda protocol: FOLDS)
    return pd.DataFrame({
        "img": list("123456"), "species": "tambaqui", "species_group": "tambaqui",
        "session_group": list("aabbcc"), "weight": [100.0, 200, 300, 400, 500, 600]})


def train_mean(train, test):
    return np.full(len(test), train["weight"].mean())


def test_writes_predictions_and_metrics(table, tmp_path):
    metrics = run_experiment("fake", "p3", table, train_mean, results_dir=tmp_path)
    preds = pd.read_csv(tmp_path / "fake/p3/predictions.csv")
    assert list(preds.columns) == ["img", "fold", "species", "species_group", "session_group", "y_true", "y_pred"]
    assert len(preds) == 6
    assert preds.loc[preds["fold"] == "a", "y_pred"].tolist() == [450.0, 450.0]
    assert metrics == json.loads((tmp_path / "fake/p3/metrics.json").read_text())
    assert metrics["overall"]["n"] == 6
    assert set(metrics["per_fold"]) == {"a", "b", "c"}
    assert set(metrics["per_weight_band"]) == {"baixo", "medio", "alto"}
    assert metrics["overall"]["mae_ci"][0] <= metrics["overall"]["mae"] <= metrics["overall"]["mae_ci"][1]


def test_model_cannot_see_test_weights(table, tmp_path):
    def cheats(train, test):
        return test["weight"].to_numpy()
    with pytest.raises(KeyError):
        run_experiment("cheat", "p3", table, cheats, results_dir=tmp_path)


def test_stale_split_is_rejected(table, tmp_path):
    with pytest.raises(ValueError, match="6"):
        run_experiment("stale", "p3", table[table["img"] != "6"], train_mean, results_dir=tmp_path)


def test_wrong_number_of_predictions_is_rejected(table, tmp_path):
    with pytest.raises(ValueError, match="dobra a:"):
        run_experiment("short", "p3", table, lambda train, test: np.zeros(1), results_dir=tmp_path)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_experiment.py -v`
Expected: FAIL com `ModuleNotFoundError: fishweight.experiment`

- [ ] **Step 3: Implementar `fishweight/experiment.py`**

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_experiment.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add fishweight/experiment.py tests/test_experiment.py && git commit -m "feat: executor de experimentos sobre divisões fixas"
```

---

### Task 7: E0 — piso alométrico

**Files:**
- Create: `fishweight/allometric.py`, `tests/test_allometric.py`, `experiments/__init__.py`, `experiments/e0_allometric.py`

**Interfaces:**
- Consumes: `run_experiment`, `build_feature_table`, `load_manifest`, `PROTOCOLS`.
- Produces:
  - `class AllometricRegressor(min_species_samples: int = 5)` com `fit(area, weight, species=None) -> Self` e `predict(area, species=None) -> np.ndarray`. Ajusta `log(W) = log(a) + b·log(Área)` por mínimos quadrados. Com `species`, ajusta um par `(a, b)` por espécie que tenha pelo menos `min_species_samples` no treino; as demais, e as nunca vistas, usam o par global. `ValueError` se alguma área ou peso for `<= 0`. Atributos `a_`, `b_` (globais) e `per_species_: dict[str, tuple[float, float]]`.
  - Experimentos gravados: `e0_global` e `e0_especie_oraculo` (`reference_only=True`, usa a espécie anotada).

- [ ] **Step 1: Escrever os testes**

```python
# tests/test_allometric.py
import numpy as np
import pytest

from fishweight.allometric import AllometricRegressor

AREA = np.array([1000.0, 2000, 4000, 8000, 16000, 32000])


def test_recovers_exact_power_law():
    model = AllometricRegressor().fit(AREA, 0.02 * AREA**1.5)
    assert model.a_ == pytest.approx(0.02)
    assert model.b_ == pytest.approx(1.5)
    assert model.predict(np.array([5000.0])) == pytest.approx([0.02 * 5000**1.5])


def test_per_species_coefficients():
    area = np.concatenate([AREA, AREA])
    species = np.array(["x"] * 6 + ["y"] * 6)
    weight = np.where(species == "x", 0.02, 0.05) * area**1.5
    model = AllometricRegressor().fit(area, weight, species)
    assert model.predict(np.array([5000.0, 5000.0]), np.array(["x", "y"])) == pytest.approx(
        [0.02 * 5000**1.5, 0.05 * 5000**1.5])


def test_unseen_or_rare_species_falls_back_to_global():
    area = np.concatenate([AREA, AREA[:2]])
    species = np.array(["x"] * 6 + ["rara"] * 2)
    model = AllometricRegressor().fit(area, 0.02 * area**1.5, species)
    assert "rara" not in model.per_species_
    assert model.predict(np.array([5000.0, 5000.0]), np.array(["rara", "nunca_vista"])) == pytest.approx(
        [0.02 * 5000**1.5] * 2)


def test_non_positive_area_is_rejected():
    with pytest.raises(ValueError):
        AllometricRegressor().fit(np.array([0.0, 10.0]), np.array([1.0, 2.0]))
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_allometric.py -v`
Expected: FAIL com `ModuleNotFoundError: fishweight.allometric`

- [ ] **Step 3: Implementar `fishweight/allometric.py` e `experiments/e0_allometric.py`**

O script monta a tabela com `build_feature_table(load_manifest())`, roda as duas variantes nos três protocolos e imprime uma linha por combinação: `e0_global p3 MAE=… MAPE=… R2=…`.

- [ ] **Step 4: Rodar os testes e o experimento**

Run: `uv run pytest tests/test_allometric.py -v && uv run python -m experiments.e0_allometric`
Expected: 4 passed; seis linhas impressas; existem `results/e0_global/{p1,p2,p3}/metrics.json` e `results/e0_especie_oraculo/{p1,p2,p3}/metrics.json`

- [ ] **Step 5: Commit**

```bash
git add fishweight/allometric.py tests/test_allometric.py experiments results/e0_* && git commit -m "feat: E0, piso alométrico em P1-P3"
```

---

### Task 8: E1 — atributos `regionprops`

**Files:**
- Create: `fishweight/regressors.py`, `tests/test_regressors.py`, `experiments/e1_regionprops.py`

**Interfaces:**
- Consumes: `run_experiment`, `build_feature_table`, `FEATURE_NAMES`, `load_manifest`, `load_splits`, `regression_metrics`, `REPO_ROOT`.
- Produces:
  - `make_models(random_state: int = 0, scale: bool = True) -> dict[str, sklearn estimator]` com as sete chaves da monografia: `GradientBoostingRegressor`, `LinearRegression`, `LassoRegression`, `RidgeRegression`, `RandomForestRegressor`, `PCARegression` (`PCA(n_components=2)` + `LinearRegression`), `PolyRegression` (`PolynomialFeatures(degree=3)` + `Lasso`). Cada chamada devolve instâncias novas. Com `scale=True`, cada modelo é um pipeline que começa com `StandardScaler`.
- Experimentos gravados (`<Modelo>` = chave de `make_models`):
  - `e1_nativo_<Modelo>`: os 12 `FEATURE_NAMES`, `scale=True`, P1–P3.
  - `e1_nativo_especie_oraculo_<Modelo>`: os 12 atributos mais a espécie anotada em one-hot (`OneHotEncoder(handle_unknown="ignore")`), `reference_only=True`, P1–P3.
  - `e1_monografia_<Modelo>`: só P1, `reference_only=True`, `scale=False`. Reproduz o notebook `notebooks/04_regression/regression_last_execution.ipynb`:
    1. Junta `annotations.csv` com `combined_features.csv` pela imagem (245 linhas, **com** o lambari de 0,6 g, como no original).
    2. Codifica `specie` com `LabelEncoder`.
    3. Calcula a correlação de Pearson de cada coluna numérica com `weight` no conjunto inteiro e fica com as de correlação `> 0.85`, mais `specie`.
    4. Treina nas 171 imagens fora do teste de P1 e avalia nas 74 do teste.
    Como usa 245 linhas, esta variante não passa por `run_experiment`: grava `results/e1_monografia_<Modelo>/p1/metrics.json` com `overall` de `regression_metrics`, `reference_only: true` e o campo extra `selected_features` (lista das colunas escolhidas).

- [ ] **Step 1: Escrever os testes**

```python
# tests/test_regressors.py
import numpy as np

from fishweight.regressors import make_models

NAMES = {"GradientBoostingRegressor", "LinearRegression", "LassoRegression", "RidgeRegression",
         "RandomForestRegressor", "PCARegression", "PolyRegression"}


def test_seven_models_fresh_on_every_call():
    first, second = make_models(), make_models()
    assert set(first) == NAMES
    assert all(first[name] is not second[name] for name in NAMES)


def test_models_fit_and_are_reproducible():
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 10, (60, 3))
    y = 5 * X[:, 0] + X[:, 1] ** 2 + rng.normal(0, 0.1, 60)
    for name in NAMES:
        a = make_models()[name].fit(X[:40], y[:40]).predict(X[40:])
        b = make_models()[name].fit(X[:40], y[:40]).predict(X[40:])
        assert np.allclose(a, b), name
        assert np.isfinite(a).all(), name
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_regressors.py -v`
Expected: FAIL com `ModuleNotFoundError: fishweight.regressors`

- [ ] **Step 3: Implementar `fishweight/regressors.py` e `experiments/e1_regionprops.py`**

O script imprime, para `e1_monografia`, a lista `selected_features` e uma tabela lado a lado com os valores de teste da Tabela 6 da monografia:

| Modelo | R² | MAE |
|---|---|---|
| GradientBoostingRegressor | 0,8832 | 33,79 |
| LinearRegression | 0,8594 | 41,85 |
| LassoRegression | 0,8545 | 39,65 |
| RidgeRegression | 0,8599 | 41,68 |
| RandomForestRegressor | 0,8810 | 34,98 |
| PCARegression | 0,7687 | 69,55 |
| PolyRegression | 0,9124 | 31,92 |

- [ ] **Step 4: Rodar os testes e o experimento**

Run: `uv run pytest tests/test_regressors.py -v && uv run python -m experiments.e1_regionprops`
Expected: 2 passed; existem `results/e1_nativo_*/{p1,p2,p3}/metrics.json` (7 modelos), `results/e1_nativo_especie_oraculo_*/{p1,p2,p3}/metrics.json` (7) e `results/e1_monografia_*/p1/metrics.json` (7). `LinearRegression` e `RidgeRegression` de `e1_monografia` ficam a menos de 0,01 de R² da tabela acima (têm solução fechada, então não dependem de semente nem de convergência). Se não ficarem, pare e reporte a diferença e a lista `selected_features`: a reprodução é o que liga os números novos aos antigos. Os outros cinco modelos são apenas reportados.

- [ ] **Step 5: Commit**

```bash
git add fishweight/regressors.py tests/test_regressors.py experiments/e1_regionprops.py results/e1_* && git commit -m "feat: E1, regionprops em P1-P3 e reprodução da monografia"
```

---

### Task 9: Conferência de escala, relatório e fechamento

**Files:**
- Create: `experiments/check_scale.py`, `experiments/report.py`, `tests/test_report.py`, `results/checks/escala.csv`, `docs/experiments/relatorio-fase1a.md`
- Modify: `AGENTS.md`, `docs/ROADMAP.md`

**Interfaces:**
- Consumes: `build_feature_table`, `load_manifest`, os `metrics.json` de `results/`.
- Produces:
  - `python -m experiments.check_scale` grava `results/checks/escala.csv` com `img, session_group, species, length_cm, length_px, px_per_cm, deviation_pct, flagged`. `length_px = feret_diameter_max × 8`; `px_per_cm = length_px / length`; `deviation_pct` é relativo à mediana global; `flagged = |deviation_pct| > 15`. Imprime mediana e coeficiente de variação por `session_group`.
  - `experiments.report.build_tables(results_dir: Path) -> str`: Markdown com uma tabela por protocolo (linhas = experimentos ordenados por MAE; os de referência com o sufixo ` (ref.)` no nome; colunas = MAE com IC 95% no formato `20,0 [19,0; 21,0]`, RMSE, MAPE, R², viés). Números com uma casa e vírgula decimal, R² com três casas. Depois, sob o título `### Por espécie: <experimento>`, a tabela por grupo de espécie do melhor experimento não-referência em P3. Por fim, se `results_dir/checks/escala.csv` existir, a tabela das imagens com `flagged`.
  - `python -m experiments.report` grava as tabelas em `docs/experiments/relatorio-fase1a.md`, entre os marcadores `<!-- tabelas:início -->` e `<!-- tabelas:fim -->`, preservando o texto fora deles.

- [ ] **Step 1: Escrever o teste do relatório**

```python
# tests/test_report.py
import json

from experiments.report import build_tables


def _metrics(path, name, mae, reference_only=False):
    path.mkdir(parents=True)
    base = {"n": 10, "mae": mae, "rmse": mae, "mape": 5.0, "r2": 0.9, "bias": 0.0}
    (path / "metrics.json").write_text(json.dumps({
        "experiment": name, "protocol": "p3", "reference_only": reference_only,
        "overall": {**base, "mae_ci": [mae - 1, mae + 1], "mape_ci": [4.0, 6.0]},
        "per_fold": {}, "per_species_group": {"tambaqui": base}, "per_weight_band": {}}))


def test_tables_rank_by_mae_and_mark_reference_runs(tmp_path):
    _metrics(tmp_path / "pior/p3", "pior", 40.0)
    _metrics(tmp_path / "melhor/p3", "melhor", 20.0)
    _metrics(tmp_path / "oraculo/p3", "oraculo", 10.0, reference_only=True)
    text = build_tables(tmp_path)
    assert text.index("oraculo (ref.)") < text.index("melhor") < text.index("pior")
    assert "20,0 [19,0; 21,0]" in text
    assert "Por espécie: melhor" in text
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL com `ModuleNotFoundError: experiments.report`

- [ ] **Step 3: Implementar `experiments/check_scale.py` e `experiments/report.py`**

- [ ] **Step 4: Rodar tudo**

Run: `uv run pytest -q && uv run python -m experiments.check_scale && uv run python -m experiments.report`
Expected: todos os testes passam; `results/checks/escala.csv` tem 244 linhas; `docs/experiments/relatorio-fase1a.md` tem as tabelas de P1, P2 e P3.

- [ ] **Step 5: Escrever a leitura dos resultados em `docs/experiments/relatorio-fase1a.md`, acima do marcador de início**

Com os números das tabelas, responder em texto corrido, uma resposta por parágrafo:
1. A variante `e1_monografia` reproduz a Tabela 6? Quais colunas foram selecionadas, e `length` está entre elas?
2. Quanto o melhor modelo não-referência perde de P2 para P3 em MAE e MAPE?
3. Tirando a espécie anotada e a medida de fita, quanto sobra do R² em P1?
4. O E1 nativo supera o E0 global em P3, considerando os intervalos de confiança?
5. Qual `session_group` tem o pior erro em P3, e as imagens com `flagged` explicam isso?
6. Recomendação sobre a sessão de 2022-01-10: manter, excluir imagens específicas ou excluir a sessão. A decisão é do autor; o relatório lista as imagens.

- [ ] **Step 6: Atualizar `AGENTS.md` e `docs/ROADMAP.md`**

`AGENTS.md`: trocar os comandos de desenvolvimento por `uv sync`, `uv run pytest` e `uv run python -m experiments.<nome>`; trocar `src/` por `fishweight/` com os módulos novos; remover os avisos que deixaram de valer (bug do numpy, `create_mask`, `DEFAULT_MODELS` compartilhado); acrescentar que `results/` e `data/splits/` são versionados.
`docs/ROADMAP.md`: marcar 1A, atualizar a linha **Agora** e acrescentar uma linha ao Registro.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "docs: relatório da Fase 1A e conferência de escala"
```
