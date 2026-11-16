# Fase 1B — Segmentação automática: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trocar a máscara manual pela máscara gerada por uma U-Net e medir quanto os pisos E0 e E1 perdem com isso, nas mesmas divisões P1, P2 e P3.

**Architecture:** Uma U-Net (encoder MobileNetV2, PyTorch) é treinada uma vez por dobra, só com as imagens de treino daquela dobra, e prevê a máscara das imagens de teste. As máscaras previstas são gravadas e medidas com o mesmo `mask_features`. Os regressores continuam treinando com os atributos da máscara manual e passam a ser testados com os atributos da máscara prevista.

**Tech Stack:** PyTorch (MPS), segmentation-models-pytorch, o pacote `fishweight` do Plano 1A.

**Spec:** `docs/superpowers/specs/2026-11-15-evolucao-fishweight-design.md`

## Global Constraints

- Tudo do Plano 1A continua valendo (Python 3.12 com `uv`, `random_state=0`, identificadores em inglês e texto em português, não tocar no legado).
- **Sem vazamento:** a máscara prevista de uma imagem vem sempre de uma U-Net que não viu aquela imagem. Sem parada antecipada nem escolha de época por dobra: são 20 épocas fixas.
- **Escala preservada:** a rede recebe a foto reduzida para 192×256 (mesma proporção 3:4 da original); o aumento de dados só espelha e altera brilho e contraste.
- **Resolução e épocas** foram escolhidas numa divisão interna do treino de P1 (136 peixes para treinar, 34 para conferir), sem tocar em nenhum conjunto de teste. O plano previa 288×384 e 30 épocas; no M4 isso passava de 40 s por época, contra 6 s em 192×256.
- A máscara prevista é levada para 434×578, a mesma resolução de trabalho da máscara manual, antes de medir.
- O curimatã de 1.760 g continua no conjunto: o autor ainda não confirmou se o peso está errado.
- Sem commits nesta fase, a pedido do autor.
- **Mudança durante a execução:** o treino das redes saiu do MacBook. O código fica pronto e testado aqui; a execução com os dados reais é feita em Kaggle, Colab ou PC com GPU por `remote/executar.ipynb`. Por isso os passos de execução abaixo estão abertos.

## Review Focus

1. **Máscara prevista vazia** (a rede não achou o peixe): a geração dos atributos falha citando a imagem e a dobra, em vez de gravar zeros.
2. **Retomada depois de interrupção:** o treino das 21 dobras leva minutos; rodar o script de novo pula as dobras cujos resultados já existem.
3. **Atributos previstos da dobra errada:** em P2 cada peixe tem três máscaras previstas; o teste de cada dobra usa a máscara daquela dobra.
4. **Foto com orientação EXIF:** a foto reduzida tem de estar alinhada com a máscara manual.
5. **Execução sem MPS:** o treino cai para CPU sem erro.

## File Structure

| Arquivo | Responsabilidade |
|---|---|
| `fishweight/segmentation/unet.py` | Rede, treino, predição, Dice e IoU |
| `fishweight/segmentation/cache.py` | Fotos e máscaras manuais reduzidas para 192×256, em `data/cache/` (fora do Git) |
| `experiments/segmentation.py` | Treina por dobra, grava máscaras previstas, Dice/IoU e atributos |
| `fishweight/features.py` | `load_predicted_features`, `with_predicted_test_features` |
| `fishweight/experiment.py` | Coluna `fold` nas linhas de teste; campo `mask` no `metrics.json` |
| `experiments/e0_allometric.py`, `experiments/e1_regionprops.py` | Variantes `*_mascara_predita` |
| `experiments/report.py` | `comparacao.md` com coluna de máscara, seção de segmentação e tabelas por protocolo |
| `results/segmentation/<protocolo>/` | `masks/<dobra>/<img>.png` e `metrics.csv` |
| `data/features/predita_<protocolo>.csv` | Atributos das máscaras previstas: `fold, img` + 12 colunas |

---

### Task 1: Rede, treino e métricas de segmentação

**Files:** Create `fishweight/segmentation/unet.py`, `tests/test_unet.py`. Modify `pyproject.toml` (torch, torchvision, segmentation-models-pytorch).

**Interfaces — Produces:**
- `IMAGE_SIZE = (192, 256)` (linhas, colunas)
- `dice_iou(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]` (máscaras booleanas; duas vazias dão `(1.0, 1.0)`)
- `build_unet(pretrained: bool = True) -> torch.nn.Module` (`smp.Unet("mobilenet_v2", classes=1)`)
- `train_unet(images: np.ndarray, masks: np.ndarray, epochs: int = 20, batch_size: int = 16, seed: int = 0, pretrained: bool = True, device: str | None = None) -> torch.nn.Module` (`images` uint8 N×H×W×3, `masks` bool N×H×W; Adam 1e-3; perda BCE + Dice; espelhamento e brilho/contraste)
- `predict_masks(model, images: np.ndarray, device: str | None = None) -> np.ndarray` (probabilidades float32 N×H×W)
- `pick_device() -> str` (`"mps"` se disponível, senão `"cpu"`)

- [x] **Step 1: Testes**
  - `test_dice_iou_exact_values`: `pred` com 4 pixels, `true` com 4, interseção 2 → Dice 0,5, IoU 1/3; duas vazias → (1, 1).
  - `test_unet_learns_to_segment_a_bright_rectangle`: 24 imagens sintéticas 32×64 (retângulo claro em fundo escuro com ruído), `pretrained=False`, CPU, treina em 16 e testa em 8 → Dice médio > 0,9.
  - `test_predict_returns_probabilities_with_input_shape`: saída N×H×W em [0, 1].
- [x] **Step 2: Ver falhar** — `uv run pytest tests/test_unet.py -q`
- [x] **Step 3: Implementar**
- [x] **Step 4: Ver passar**

### Task 2: Cache de fotos reduzidas

**Files:** Create `fishweight/segmentation/cache.py`, `tests/test_cache.py`. Modify `.gitignore` (`data/cache/`).

**Interfaces — Produces:**
- `load_image_small(path: Path) -> np.ndarray` (uint8 192×256×3; aplica a orientação EXIF; `ValueError(f"{path}: foto {l}×{c}, esperado 3472×4624")` se a resolução for outra)
- `resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray` (booleana; usada para levar a máscara manual a 192×256)
- `load_small_dataset(manifest: pd.DataFrame, cache_dir: Path = REPO_ROOT / "data/cache") -> tuple[np.ndarray, np.ndarray, np.ndarray]` (fotos reduzidas, máscaras manuais reduzidas e máscaras manuais em 434×578, na ordem do manifesto; cacheia em `.npz` e refaz se a lista de imagens mudar)

- [x] **Step 1: Testes**
  - `test_small_image_keeps_layout`: foto sintética 3472×4624 com o quadrante superior esquerdo vermelho → saída 192×256×3 com esse quadrante vermelho.
  - `test_exif_orientation_is_applied`: foto gravada deitada com orientação EXIF 6 sai na posição correta.
  - `test_wrong_photo_resolution_is_rejected`: 1080×1920 → `ValueError` com arquivo e tamanho.
  - `test_resize_mask_preserves_area_fraction`: retângulo que ocupa 1/8 da imagem continua ocupando 1/8 (±1%) em 192×256 e em 434×578.
  - `test_photos_and_manual_masks_are_aligned` (dados reais): em 20 peixes, a cor média dentro da máscara difere da cor média fora dela em mais de 10 níveis.
- [x] **Step 2–4:** ver falhar, implementar, ver passar.

### Task 3: Máscaras previstas fora da amostra

**Files:** Create `experiments/segmentation.py`, `tests/test_segmentation_run.py`.

**Interfaces — Produces:**
- `run_fold(fold: Fold, images, masks, image_names, reference_masks, out_dir: Path, epochs: int = 20, pretrained: bool = True, device: str | None = None) -> pd.DataFrame`: treina nas imagens de `fold["train"]`, prevê as de `fold["test"]`, grava `out_dir/masks/<dobra>/<img>.png` (na resolução de `reference_masks`, 0/255) e devolve `img, fold, dice, iou` medidos contra `reference_masks`. Se `out_dir/masks/<dobra>/` já tiver todas as máscaras do teste, não treina: relê e mede.
- `predicted_features(out_dir: Path, fold: Fold) -> pd.DataFrame` (`fold, img` + `FEATURE_NAMES`). Máscara prevista vazia gera `ValueError` citando dobra e imagem.
- `python -m experiments.segmentation` roda todas as dobras de P1, P2 e P3 e grava `results/segmentation/<protocolo>/metrics.csv` e `data/features/predita_<protocolo>.csv` (`fold, img` + `FEATURE_NAMES`). Máscara prevista vazia gera `ValueError` citando imagem e dobra.

- [x] **Step 1: Testes** (sintéticos, 32×64, `pretrained=False`)
  - `test_run_fold_writes_one_mask_per_test_image_and_scores_it`
  - `test_run_fold_skips_training_when_masks_exist` (segunda chamada com `epochs=0` devolve as mesmas métricas)
  - `test_empty_predicted_mask_is_rejected`
- [x] **Step 2–4:** ver falhar, implementar, ver passar.
- [ ] **Step 5: Rodar de verdade, em GPU** — `python -m experiments.segmentation` (21 dobras). Esperado: 74 máscaras em P1, 732 em P2 e 244 em P3; nenhuma vazia. **Feito até aqui, localmente:** P1 inteiro (74 máscaras) e a dobra `r0f0` de P2. O script retoma de onde parou.

### Task 4: Experimentos com máscara prevista

**Files:** Modify `fishweight/experiment.py`, `fishweight/features.py`, `experiments/e0_allometric.py`, `experiments/e1_regionprops.py`, testes correspondentes.

**Interfaces — Produces:**
- `run_experiment(..., mask: str = "manual")`: grava `"mask"` no `metrics.json`; as linhas de teste passadas a `fit_predict` ganham a coluna `fold`.
- `load_predicted_features(protocol: str) -> pd.DataFrame` (índice `(fold, img)`, colunas `FEATURE_NAMES`)
- `with_predicted_test_features(fit_predict: FitPredict, predicted: pd.DataFrame) -> FitPredict`: troca os `FEATURE_NAMES` das linhas de teste pelos da máscara prevista daquela dobra; `ValueError` se faltar algum par `(fold, img)`. O treino continua com a máscara manual.
- Experimentos novos: `e0_global_mascara_predita`, `e0_especie_oraculo_mascara_predita` (ref.), `e1_nativo_<Modelo>_mascara_predita`, `e1_nativo_especie_oraculo_<Modelo>_mascara_predita` (ref.), todos com `mask="predita"`.

- [x] **Step 1: Testes**
  - `test_test_rows_carry_their_fold_name` (executor)
  - `test_metrics_record_the_mask_source` (executor)
  - `test_predicted_features_replace_only_the_test_rows_of_that_fold`
  - `test_missing_predicted_mask_is_rejected`
- [x] **Step 2–4:** ver falhar, implementar, ver passar.
- [ ] **Step 5:** `uv run python -m experiments.e0_allometric && uv run python -m experiments.e1_regionprops`. **Feito para P1**; P2 e P3 são pulados com aviso até as máscaras previstas chegarem.

### Task 5: Comparação, documentação e fechamento

**Files:** Modify `experiments/report.py`, `tests/test_report.py`, `docs/experiments/tecnicas.md`, `docs/ROADMAP.md`, `AGENTS.md`. Create `docs/experiments/relatorio-fase1b.md`.

- `comparacao.md` passa a ser o único arquivo gerado: tabela-resumo com a coluna **Máscara**, seção **Segmentação** (Dice e IoU médios por protocolo, lidos de `results/segmentation/*/metrics.csv`), as tabelas por protocolo e o detalhamento (espécie, sessão, faixa de peso) do melhor candidato. Candidato = não-referência **com máscara prevista**; enquanto não houver nenhum, vale o melhor não-referência.
- `relatorio-fase1a.md` fica como está, sem ser regenerado.

- [x] **Step 1: Testes** — `test_comparison_shows_mask_source`, `test_breakdown_prefers_predicted_mask_runs`, `test_segmentation_summary_per_protocol`
- [x] **Step 2–4:** ver falhar, implementar, ver passar.
- [x] **Step 5:** `uv run pytest -q && uv run python -m experiments.report`
- [ ] **Step 6:** escrever `relatorio-fase1b.md` (quanto custa a máscara automática em P3; Dice/IoU contra a monografia; pior sessão; o que muda para a ferramenta) quando P2 e P3 estiverem completos. `tecnicas.md`, `ROADMAP.md` e `AGENTS.md` já estão atualizados.

### Task 6: Preparar a execução em outro ambiente (acrescentada)

**Files:** Create `remote/executar.ipynb`. Modify `pyproject.toml`, `fishweight/segmentation/unet.py`, `experiments/common.py`, testes de treino.

- [x] `pick_device()` escolhe `cuda`, depois `mps`, depois `cpu`.
- [x] `pyproject.toml`: Python `>=3.11` e pisos baixos de PyTorch, para `pip install -e .` aproveitar o que Kaggle e Colab já trazem.
- [x] Testes que treinam rede marcados como `slow`; `uv run pytest -q -m "not slow"` é a suíte do dia a dia.
- [x] `run_both_masks` pula, com aviso, o protocolo que ainda não tem máscaras previstas.
- [x] `remote/executar.ipynb`: clona o branch, instala, confere a GPU, roda os testes, a segmentação, os experimentos leves e empacota `results/`, `data/features/` e `comparacao.md` num zip.
- [ ] Notebook executado de ponta a ponta num ambiente com GPU. Ainda não foi rodado em lugar nenhum.
