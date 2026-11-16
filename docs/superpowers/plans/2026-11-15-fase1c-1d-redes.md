# Fases 1C e 1D — Regressão direta, multitarefa e híbrido: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deixar prontos, testados e executáveis em GPU os experimentos E2 (regressão direta da foto), E3 (multitarefa), E4 e E5 (híbrido).

**Architecture:** Um módulo único, `fishweight/direct.py`, cobre os três usos de um backbone do `timm`: gerar vetores com a rede congelada, ajuste fino para prever o peso e fusão com os atributos da máscara. Cada experimento é um script fino que monta o `fit_predict` e o entrega ao executor do Plano 1A. Os scripts caros pulam o que já tem resultado, para sobreviver a sessões de GPU interrompidas.

**Tech Stack:** PyTorch, timm, scikit-learn, o pacote `fishweight`.

**Spec:** `docs/superpowers/specs/2026-11-15-evolucao-fishweight-design.md`

## Global Constraints

- Nada de treino com dados reais no MacBook: aqui só código, testes com redes minúsculas e experimentos leves. A execução é por `remote/executar.ipynb`.
- Sem vazamento: épocas fixas, sem parada antecipada; padronizações e codificações ajustadas só no treino; espécie e medidas de fita entram apenas na perda.
- Escala preservada: entradas 3:4 (288×384 nas convolucionais, 294×392 no ViT); aumento só com espelhamento, brilho e contraste.
- Pesos pré-treinados fixados por nome em `BACKBONES`.
- Sem commits, a pedido do autor.

## Review Focus

1. **Sessão de GPU interrompida:** rodar o script de novo não repete o que já tem `metrics.json`.
2. **Anotações do peixe de teste:** espécie e medidas do teste não podem influenciar a predição.
3. **Espécie só no teste** (lambari e traíra em P3): a saída auxiliar não pode quebrar.
4. **E3 e E4 antes de E2:** sem resultado de ajuste fino em P3, o script diz o que rodar primeiro.
5. **Híbrido sem máscaras previstas ou sem vetores:** avisa e pula, sem falhar.

---

### Task 1: Organização e utilitários comuns

- [x] `fishweight/torch_utils.py`: `pick_device`, `to_tensor`, `augment`, `shuffled_batches`.
- [x] `fishweight/photos.py` (ex-`segmentation/cache.py`): `load_image_small(path, size)`, `resize_mask`, `load_photos`, `load_masks`, com cache por tamanho.
- [x] `fishweight/unet.py` (ex-`segmentation/unet.py`) sobre os utilitários comuns.
- [x] Testes: `tests/test_photos.py`, `tests/test_unet.py`, `tests/test_segmentation_run.py` passando.

### Task 2: Regressão direta (`fishweight/direct.py`)

- [x] `BACKBONES`, `input_size`, `embed`, `embedding_file`, `add_embeddings`.
- [x] `WeightRegressor`, `train_regressor`, `FittedRegressor.predict`, `direct_fit_predict`.
- [x] Testes (`tests/test_direct.py`): proporção 3:4 de toda entrada; vetores independentes do tamanho do lote; a rede aprende o peso a partir da foto; alvo em log dá pesos positivos; predição ignora as anotações do peixe de teste; o híbrido usa os atributos do teste; junção de vetores por nome de imagem.
- [x] Os quatro backbones foram construídos no tamanho de entrada previsto; o DINOv2 carregou os pesos pré-treinados em 294×392.

### Task 3: Scripts dos experimentos

- [x] `experiments/common.py`: `run_once`, `run_and_report`, `best_finetuned`, `sklearn_fit_predict`, `training_arguments`. Testes em `tests/test_common.py`.
- [x] `experiments/e2_direct.py`: vetores e regressores sobre rede congelada (P1–P3); ajuste fino em gramas e em log (P1 e P3 por padrão).
- [x] `experiments/e3_multitask.py`: saídas auxiliares de espécie e de medidas sobre o melhor ajuste fino.
- [x] `experiments/e4_hybrid.py`: fusão com rede congelada (máscara manual e prevista), fusão dentro da rede e E5.
- [x] `experiments/report.py`: concorrem as rodadas com máscara prevista ou sem máscara.
- [x] Fumaça local: os três scripts rodaram de ponta a ponta com uma rede minúscula, uma época e só P1; as saídas foram apagadas.

### Task 4: Execução em GPU

- [ ] `python -m experiments.segmentation` completo (P2 e P3).
- [ ] `python -m experiments.e2_direct` (vetores das quatro redes; ajuste fino em P1 e P3).
- [ ] `python -m experiments.e3_multitask`.
- [ ] `python -m experiments.e4_hybrid`.
- [ ] Resultados trazidos de volta e `experiments.report` rodado.
- [ ] `docs/experiments/relatorio-fase1b.md` e relatório de 1C/1D escritos a partir da comparação.

## Decisões tomadas sem medir (rever com os resultados)

- 30 épocas, AdamW com taxa 0,0001 e decaimento cosseno, lote 16, perda Huber, peso 0,3 nas perdas auxiliares. São valores usuais, não ajustados.
- O ajuste fino roda por padrão só em P1 e P3. P2 soma 15 dobras e não decide o vencedor; inclua com `--protocols p1 p2 p3`.
- O DINOv2 entra só congelado, como na pesquisa.
- As fusões dentro da rede rodam apenas com máscara prevista, para não dobrar o custo.
