# Roadmap

**Agora:** código dos Planos 1B, 1C, 1D e 1E pronto e testado. O próximo passo é seu: enviar o branch e executar `remote/executar.ipynb` num ambiente com GPU. Enquanto isso, o que pode avançar aqui é o Plano 3A (referências do artigo).

Desenho aprovado: [`docs/superpowers/specs/2026-11-15-evolucao-fishweight-design.md`](superpowers/specs/2026-11-15-evolucao-fishweight-design.md)

## Como acompanhar

- Este arquivo é o painel. Cada linha é um plano; a caixa é marcada quando a última tarefa do plano termina e os testes passam.
- Cada plano em `docs/superpowers/plans/` tem suas próprias caixas, uma por passo. É lá que se vê o andamento do dia.
- Um plano por vez. O seguinte só é escrito quando o anterior termina, porque depende dos resultados dele.
- A linha **Agora**, no topo, diz onde o trabalho está. Ela é atualizada ao fim de cada plano.
- Resultados numéricos ficam em `results/`; a leitura deles, em `docs/experiments/`.
- [`docs/experiments/tecnicas.md`](experiments/tecnicas.md) descreve cada técnica implementada; [`docs/experiments/comparacao.md`](experiments/comparacao.md) põe as métricas de todas lado a lado. Os dois são atualizados a cada plano.

## Onde cada coisa roda

| O quê | Onde | Como |
|---|---|---|
| Escrever código, testes, experimentos leves (E0, E1, escala, comparação) | Neste MacBook | `uv run pytest -q -m "not slow"` e `uv run python -m experiments.<nome>` |
| Treinar redes nos dados reais (segmentação e, adiante, E2 a F2) | Kaggle, Colab ou PC com GPU | Enviar o branch (`git push`), abrir `remote/executar.ipynb`, rodar, baixar o zip e descompactar na raiz do repositório |

Cada plano daqui em diante tem duas caixas: **código** (pronto e testado aqui) e **execução** (rodado em GPU e resultados trazidos de volta).

## Fase 1 — Experimentos

| | Plano | Entrega | Documento |
|---|---|---|---|
| [x] | **1A** Base e pisos | Projeto `uv`, pacote `fishweight`, manifesto de 244 peixes, divisões P1–P3, métricas, E0 e E1 com máscara manual, conferência de escala | [plano](superpowers/plans/2026-11-15-fase1a-base-e-pisos.md) · [relatório](experiments/relatorio-fase1a.md) |
| [ ] | **1B** Segmentação | U-Net em PyTorch com máscaras fora da amostra; E0 e E1 com máscara predita. **Código: pronto. Execução: P1 feito (74 máscaras); P2 e P3 pendentes em GPU** | [plano](superpowers/plans/2026-11-15-fase1b-segmentacao.md) |
| [ ] | **1C** Regressão direta | E2 (ResNet50, EfficientNet-B3, ConvNeXt-Tiny, DINOv2), E3 (espécie e medidas auxiliares). **Código: pronto. Execução: pendente em GPU** | [plano](superpowers/plans/2026-11-15-fase1c-1d-redes.md) |
| [ ] | **1D** Híbrido | E4 e E5. **Código: pronto. Execução: pendente em GPU** | [plano](superpowers/plans/2026-11-15-fase1c-1d-redes.md) |
| [ ] | **1E** FDWE | Leitor do dataset, recortes, F1 (pré-treino) e F2 (domínio cruzado). **Código: pronto. Execução: pendente em GPU** | [plano](superpowers/plans/2026-11-15-fase1e-fdwe.md) |
| [ ] | **1F** Fechamento | Intervalos de predição, comparação estatística, relatório final, exportação ONNX | a escrever |

**Parada:** você lê `docs/experiments/relatorio-fase1.md` e confirma o modelo vencedor.

## Fase 2 — Ferramenta FishWeight

| | Plano | Entrega | Documento |
|---|---|---|---|
| [ ] | **2A** Motor e API | `fishweight.inference.Predictor`, FastAPI, PostgreSQL, MinIO, teste de paridade | a escrever |
| [ ] | **2B** Interface | Next.js: captura, resultado, histórico | a escrever |
| [ ] | **2C** Integração | Docker Compose, teste de ponta a ponta, meta de 1 s no p95 | a escrever |

**Parada:** você usa a ferramenta com o celular e a caixa.

## Fase 3 — Artigo

| | Plano | Entrega | Documento |
|---|---|---|---|
| [ ] | **3A** Referências | `.bib` conferido na fonte e seção de trabalhos relacionados (pode andar junto com a Fase 1) | a escrever |
| [ ] | **3B** Métodos e resultados | Tabelas e figuras geradas de `results/`, seções de métodos, experimentos e resultados | a escrever |
| [ ] | **3C** Fechamento | Seção da ferramenta, limitações, suplementar, PDFs compilados | a escrever |

**Parada:** você e o orientador revisam o PDF.

## Pendências suas

- [ ] **Executar em GPU:** fazer commit e `git push` do branch `fase-1a-base-e-pisos`, abrir `remote/executar.ipynb` no Kaggle ou no Colab, rodar tudo, baixar `resultados_fishweight.zip` e descompactar na raiz do repositório. Sem isso não há máscara prevista em P2 e P3 nem resultado de E2 a E5.

## Decididas por recomendação

Resolvidas em 2026-11-15 pela opção recomendada, a seu pedido. Evidências e como reverter em [`docs/experiments/decisoes-dados.md`](experiments/decisoes-dados.md).

- [x] Curimatã de 1.760 g: fica no conjunto; a comparação mostra o erro com e sem ele.
- [x] Sessão de 29/10/2022: fica como está; a ferramenta ganha uma referência de escala na caixa.
- [x] Traíras com comprimento de 8 e 7 cm: o peso é usado, a medida de fita não.
- [x] Datas de coleta: valem as seis datas gravadas nas fotos.
- [x] Evento alvo: SBIAgro como primeira opção; conferir a chamada antes de submeter.

## Registro

| Data | O que aconteceu |
|---|---|
| 2026-11-15 | `AGENTS.md` criado; pesquisa salva em `docs/research/`; desenho escrito; Plano 1A escrito |
| 2026-11-15 | Plano 1A executado e revisado: 35 testes; E0 e E1 em P1–P3; monografia reproduzida; o melhor modelo só de máscara tem MAE de 38,3 g e R² 0,925 com uma sessão de fora |
| 2026-11-15 | Criados `tecnicas.md` e `comparacao.md`. Plano 1B implementado: U-Net em PyTorch, máscaras fora da amostra, experimentos com máscara prevista. Treino local interrompido a pedido: as redes passam a rodar em Kaggle, Colab ou PC com GPU. Em P1, a U-Net atinge Dice 0,981 e a máscara prevista custa menos de 1 g de MAE |
| 2026-11-15 | Repositório reorganizado: código original da monografia (notebooks, scripts, modelo Keras) reunido em `monografia/`; PDF e figuras em `docs/monografia/`; tabelas antigas em `data/monografia/`; removidos só duplicados (`combined_no_29_11_22/`, `ESTRUTURA.md`, utilitários sem uso). Implementados E2, E3, E4 e E5 (`fishweight/direct.py` e scripts), testados com redes minúsculas. 64 testes |
| 2026-11-15 | FDWE copiado para `data/external/fdwe/` e inspecionado (1.086 imagens, 2.216 peixes, CC BY 4.0). Implementados o leitor (`fishweight/fdwe.py`) e os experimentos F1 e F2, testados com uma amostra de 50 imagens e rede minúscula |
| 2026-11-15 | Cinco perguntas sobre os dados resolvidas pela opção recomendada (`decisoes-dados.md`). Novas variantes de referência `*_escala_sessao`, tabela de sensibilidade a pesos suspeitos e medidas de fita duvidosas fora da saída auxiliar. 82 testes |
