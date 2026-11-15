# Roadmap

**Agora:** Plano 1A escrito, aguardando sua revisão para começar.

Desenho aprovado: [`docs/superpowers/specs/2026-11-15-evolucao-fishweight-design.md`](superpowers/specs/2026-11-15-evolucao-fishweight-design.md)

## Como acompanhar

- Este arquivo é o painel. Cada linha é um plano; a caixa é marcada quando a última tarefa do plano termina e os testes passam.
- Cada plano em `docs/superpowers/plans/` tem suas próprias caixas, uma por passo. É lá que se vê o andamento do dia.
- Um plano por vez. O seguinte só é escrito quando o anterior termina, porque depende dos resultados dele.
- A linha **Agora**, no topo, diz onde o trabalho está. Ela é atualizada ao fim de cada plano.
- Resultados numéricos ficam em `results/`; a leitura deles, em `docs/experiments/`.

## Fase 1 — Experimentos

| | Plano | Entrega | Documento |
|---|---|---|---|
| [ ] | **1A** Base e pisos | Projeto `uv`, pacote `fishweight`, manifesto de 244 peixes, divisões P1–P3, métricas, E0 e E1 com máscara manual, conferência de escala | [plano](superpowers/plans/2026-11-15-fase1a-base-e-pisos.md) |
| [ ] | **1B** Segmentação | U-Net em PyTorch com máscaras fora da amostra; E0 e E1 com máscara predita | a escrever |
| [ ] | **1C** Regressão direta | E2 (ResNet50, EfficientNet-B3, ConvNeXt-Tiny, DINOv2), E3 (espécie e medidas auxiliares) | a escrever |
| [ ] | **1D** Híbrido | E4 e E5 | a escrever |
| [ ] | **1E** FDWE | Download, recortes, F1 (pré-treino) e F2 (domínio cruzado) | a escrever |
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

- [ ] Datas de coleta: a monografia fala em cinco dias entre 29/10 e 10/12/2022; os nomes dos arquivos mostram seis datas, duas no início de 2022. Qual vale para o artigo?
- [ ] Evento ou revista alvo do artigo.

## Registro

| Data | O que aconteceu |
|---|---|
| 2026-11-15 | `AGENTS.md` criado; pesquisa salva em `docs/research/`; desenho escrito; Plano 1A escrito |
