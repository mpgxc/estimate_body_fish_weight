# Evolução do projeto: experimentos, ferramenta FishWeight e artigo

Data: 2026-11-15 · Estado: aguardando aprovação

## Por quê

A monografia (2023) mostrou que dá para estimar o peso de peixes a partir de uma foto: U-Net segmenta o corpo, `regionprops` mede a forma e um regressor prevê o peso, com R² ≈ 0,90 no teste. O trabalho parou em notebooks. Três coisas faltam para ele virar ciência publicável e algo que um produtor use:

1. **Saber se o resultado se sustenta.** O R² foi medido com divisão aleatória. Espécie e dia de coleta estão misturados nos dados (75 das 79 fotos de 29/11/2022 são tambaqui), então o número pode estar otimista.
2. **Testar as técnicas que a literatura recente usa** (regressão direta da imagem, modelos híbridos, multiespécie), levantadas na pesquisa salva em `docs/research/2026-11-pesquisa-fish-weight-chatgpt.md`.
3. **Entregar uma ferramenta e um artigo** baseados no modelo que vencer a comparação.

São três projetos encadeados, nesta ordem: **experimentos → ferramenta → artigo**. A ferramenta e o artigo dependem do modelo vencedor. Cada fase recebe seu próprio plano de implementação e só começa depois que a anterior for revisada.

## O que existe hoje

- **Dados** (`data/coletas/combined/`, versionados no Git apesar da regra em `.gitignore`): 245 peixes, uma foto cada (4624×3472), máscara manual do corpo (sem cauda e nadadeiras) e `annotations.csv` com `img, specie, weight, width, length`.
- **Espécies**: tambaqui 148, tilápia 61, curimatã 20, panga 8, traíra 5, lambari 3.
- **Sessões** (data no nome do arquivo): 2022-01-10 (21), 2022-02-12 (6), 2022-10-29 (25), 2022-11-22 (48), 2022-11-26 (66), 2022-11-29 (79).
- **Captura** (monografia, seção 4.1): peixe dentro de uma caixa de isopor com tampa, câmera do celular encaixada num orifício da tampa. A distância entre câmera e peixe é fixa, logo a escala em pixels é praticamente constante entre as fotos.
- **Resultados da monografia** (teste 70/30, 74 peixes): melhor R² 0,912 com máscara manual e 0,904 com máscara da U-Net (regressão polinomial); MAE de 32 g e 39 g.
- **Escala conferida nos dados**: dividindo o maior lado da caixa delimitadora da máscara pelo comprimento medido com fita, a mediana é 125 px/cm, com variação de 1% a 5% dentro de cada sessão. A exceção é a sessão de 2022-01-10 (21 fotos), com variação de 33%: ali a escala ou as medidas não são consistentes, e isso precisa ser investigado no início da Fase 1.
- **Código**: `src/` (pacote refatorado que nenhum notebook importa), cópias na raiz usadas pelos notebooks, `models/unet_model.h5` (Keras). Problemas conhecidos estão em `AGENTS.md`.

## Objetivos

1. Uma comparação justa entre o método atual e as novas técnicas, com os mesmos conjuntos de treino e teste para todos, incluindo um protocolo que separa dias de coleta.
2. Um motor de inferência único, usado tanto na avaliação quanto na ferramenta, que recebe uma foto e devolve peso, intervalo, espécie provável e qualidade da imagem.
3. Uma aplicação web (FishWeight) em que a pessoa fotografa ou envia a imagem, vê a estimativa em cerca de um segundo, registra o peso real da balança e consulta o histórico.
4. Um artigo-base completo em português, no modelo SBC, com todas as tabelas e figuras geradas a partir dos arquivos de resultado e fácil de transpor depois para o modelo de uma revista.

## Fora do escopo

- Peixes vivos, dentro d'água, ou mais de um peixe por foto.
- Câmera estéreo, profundidade ou qualquer sensor além de uma foto RGB.
- Vídeo contínuo. A primeira versão trabalha com foto capturada; o modo ao vivo fica para depois.
- Login, multiusuário e implantação em nuvem. A ferramenta roda localmente com Docker Compose. O armazenamento usa a API do S3 para que migrar para a AWS depois seja troca de configuração.
- Monitoramento (Prometheus, Grafana) e re-treino automático a partir do feedback. O peso real fica guardado para um re-treino manual futuro.
- Coleta de novos peixes. O desenho assume que só existem os dados atuais.

## O que seria um fracasso mesmo funcionando

- **Vazamento entre treino e teste**: escolher hiperparâmetros, parar o treino ou calibrar intervalos olhando o conjunto de teste; usar máscara predita por uma U-Net que viu aquela imagem no treino.
- **Número bonito e enganoso**: publicar só o resultado da divisão aleatória, ou esconder que três espécies têm oito peixes ou menos.
- **Pipeline diferente na ferramenta**: pré-processamento da API divergindo do usado no treino.
- **Falsa precisão**: mostrar "842 g" sem intervalo, ou estimar com confiança uma foto fora das condições validadas.
- **Referência inventada no artigo**: citar trabalho ou número que não foi conferido na fonte.

## Restrições

- Máquina: Apple M4, 16 GB. Treino em PyTorch com aceleração MPS.
- Python 3.12 gerenciado com `uv` e `pyproject.toml`. O `requirements.txt` atual sai.
- 244 peixes úteis: o lambari anotado com 0,6 g é excluído por uma lista explícita no código, sem editar o CSV original.
- Dados próprios já estão versionados no repositório e assim continuam.
- FDWE (Wageningen, 1.086 imagens, 2.216 peixes com peso, 9 espécies marinhas em esteira) é baixado para `data/external/fdwe/`, fora do Git. A licença é conferida no download e registrada no artigo.

## Fase 1 — Experimentos

A fase é executada em seis planos (1A a 1F), listados em `docs/ROADMAP.md`.

**Como o método da monografia está no notebook:** a espécie anotada entra como atributo de entrada, e os atributos foram escolhidos pela correlação com o peso (> 0,85) calculada no conjunto inteiro, teste incluído. Refazendo essa seleção, o comprimento medido com fita (`length`, correlação 0,868) também passa no corte e entra no modelo. O R² 0,90 da monografia, portanto, não vem só da imagem. E1 é rodado de três formas: reprodução fiel (só em P1), só com atributos da máscara, e com a espécie anotada como teto de referência.

### Protocolos de avaliação

Os índices de cada divisão ficam em `data/splits/*.json`, versionados, e todos os experimentos leem os mesmos arquivos.

| Protocolo | Divisão | Para quê |
|---|---|---|
| P1 reprodução | 70/30 aleatória, `random_state=0`, como na monografia | Ligar os novos números aos antigos |
| P2 aleatório | 5 partes estratificadas por espécie, repetido 3 vezes | Estimativa estável no cenário favorável |
| P3 sessão | Deixa uma sessão de fora por vez; as duas sessões pequenas do início de 2022 formam um grupo (5 rodadas) | Medir generalização para um dia de coleta não visto |

Dentro de cada rodada, qualquer escolha (hiperparâmetro, parada antecipada, seleção de atributos, calibração) usa só uma fatia de validação tirada do treino.

**P3 com máscara predita é o que decide o modelo vencedor**, porque é isso que a ferramenta terá em mãos.

Só concorre a vencedor o modelo que funciona a partir da foto, sem informação digitada. Variantes "por espécie" usam a espécie prevista por um classificador; a versão com a espécie anotada aparece nas tabelas apenas como teto de referência. F1 só concorre se a licença do FDWE permitir distribuir pesos derivados dele.

### Experimentos

| Id | Entrada | Modelo | Papel |
|---|---|---|---|
| E0 | Área da máscara | Alométrico `W = a·Áreaᵇ`, global e por espécie | Piso interpretável |
| E1 | 53 atributos `regionprops` | Regressores atuais (`DEFAULT_MODELS`) | Método da monografia |
| E2 | Foto RGB | ResNet50, EfficientNet-B3, ConvNeXt-Tiny com ajuste fino; DINOv2 congelado + Ridge/MLP | Regressão direta |
| E3 | Foto RGB | Melhor E2 + saídas auxiliares: (a) espécie, (b) comprimento e largura medidos | Multitarefa |
| E4 | Foto RGB + atributos da máscara | Fusão: (a) embeddings congelados + atributos → Ridge/GBR, (b) MLP treinado ponta a ponta | Híbrido |
| E5 | Igual a E4 | Melhor E4 + saída auxiliar de espécie | Híbrido multiespécie |
| F1 | Foto RGB | Melhor E2 pré-treinado no FDWE e depois ajustado nos dados próprios | O FDWE ajuda? |
| F2 | Foto RGB | Modelo treinado num conjunto e testado no outro, sem ajuste | Tamanho da diferença de domínio |

Decisões que valem para todos:

- **Escala preservada.** Como a distância da câmera é fixa, a foto é reduzida mantendo a proporção (sem esticar para quadrado) e o aumento de dados não usa zoom nem recorte redimensionado. Só espelhamento, rotação pequena, brilho e contraste. Os atributos de E0/E1/E4 são recalculados nessa geometria; os CSVs antigos (224×224 esticado) servem apenas para P1.
- **Máscara manual e máscara predita.** E0, E1, E4 e E5 rodam nas duas. A predita vem de uma U-Net com encoder MobileNetV2 treinada de novo em PyTorch, sempre por um modelo que não viu aquela imagem. Dice e IoU são reportados por rodada.
- **Alvo.** Peso em gramas; `log(peso)` entra como variação. Perda Huber nas redes.
- **FDWE.** As imagens têm vários peixes, então cada peixe é recortado pela caixa anotada. A escala lá é outra; a hipótese de que o pré-treino ajuda pode falhar e o resultado é reportado de qualquer forma.

### Métricas

- MAE, RMSE (gramas), MAPE (%), R² e viés (erro médio com sinal), com intervalo de confiança de 95% por bootstrap.
- Por espécie: tambaqui, tilápia e curimatã separadas; panga, traíra e lambari agrupadas como "outras", com o aviso de amostra pequena.
- Por faixa de peso (tercis).
- Comparação pareada entre modelos (Wilcoxon sobre o erro absoluto por peixe).
- Latência p50 e p95 da inferência completa em CPU.

### Intervalo de predição

Predição conforme usando os resíduos relativos fora da amostra, com alvo de 90% de cobertura. A cobertura real é medida em P3 e reportada.

### Saídas da fase

- `results/<experimento>/<protocolo>/predictions.csv` (uma linha por peixe e rodada) e `metrics.json`.
- `docs/experiments/relatorio-fase1.md` com as tabelas comparativas e a recomendação do modelo vencedor: menor MAE em P3 com máscara predita; em empate estatístico, o mais simples.
- O modelo vencedor e a U-Net exportados para ONNX em `artifacts/`, com um arquivo de metadados (versão, pré-processamento, métricas, divisão usada).

**Parada para revisão:** você lê o relatório e confirma o modelo antes da Fase 2.

## Fase 2 — Ferramenta FishWeight

### Motor de inferência

Uma classe no pacote Python (`fishweight.inference.Predictor`) carrega os modelos ONNX uma vez e executa: validar imagem → segmentar → medir qualidade → estimar peso → calcular intervalo. A avaliação da Fase 1 e a API chamam essa mesma classe.

Resposta:

```json
{
  "weight_grams": 842.5,
  "prediction_interval": { "lower": 760, "upper": 925, "coverage": 0.9 },
  "species": { "name": "tilapia", "confidence": 0.94 },
  "quality": { "is_valid": true, "segmentation_score": 0.97, "reasons": [] },
  "model": { "name": "fishweight-hybrid", "version": "1.0.0" },
  "latency_ms": 85
}
```

Valores ilustrativos. `species` só existe se o modelo vencedor tiver saída de espécie.

`quality.is_valid` vira `false`, com o motivo em `reasons`, quando: a foto não tem a resolução e a proporção das fotos da coleta (4624×3472), não há peixe, há mais de uma região grande, a máscara encosta na borda, ou a área está fora da faixa vista no treino. Nesses casos a interface mostra o aviso em vez de um número confiante.

**Condição de uso (decisão sua):** a ferramenta será usada sempre com o mesmo celular e a mesma caixa da coleta; fotos de outra resolução são recusadas. A tela de captura explica o arranjo.

**Referência de escala (acrescentada em 2026-11-15):** um marcador de tamanho conhecido fica fixo no fundo da caixa e é lido em cada foto. Se a escala medida fugir da usada no treino, a ferramenta corrige as medidas ou recusa a foto. Motivo: numa das sessões a razão entre pixels e centímetros é 4% maior, e os dados não dizem se foi a câmera ou a fita; ver `docs/experiments/decisoes-dados.md`.

### Componentes

| Parte | Tecnologia | Responsabilidade |
|---|---|---|
| `apps/api` | FastAPI + ONNX Runtime | Recebe a imagem, chama o motor, grava resultado |
| `apps/web` | Next.js + TypeScript | Captura, resultado, histórico |
| Banco | PostgreSQL | Predições e pesos reais |
| Arquivos | S3 (MinIO no Compose) | Foto original e máscara |
| Empacotamento | Docker Compose | Sobe os quatro serviços com um comando |

API:

- `POST /predictions`: imagem (multipart) → resposta do motor + `id`.
- `GET /predictions`: lista paginada, com filtro por data e espécie.
- `GET /predictions/{id}`: detalhe, com URLs da foto e da máscara.
- `PATCH /predictions/{id}`: registra o peso real em gramas.
- `GET /predictions/export.csv`: histórico completo.
- `GET /health`: versão dos modelos carregados.

Tabela `predictions`: `id`, `created_at`, `image_key`, `mask_key`, `weight_grams`, `interval_lower`, `interval_upper`, `species`, `species_confidence`, `quality_valid`, `quality_reasons`, `model_name`, `model_version`, `latency_ms`, `real_weight_grams` (nulo até ser informado), `real_weight_at`.

Telas (em português):

1. **Captura**: abre a câmera nativa do celular (campo de arquivo com `capture`), que entrega a foto em resolução cheia, ou aceita um arquivo já salvo. Não usa quadros de vídeo do navegador, que teriam outra resolução e outro enquadramento. Funciona pelo celular na rede local, sem HTTPS.
2. **Resultado**: foto com a máscara sobreposta, peso estimado e intervalo, espécie provável, qualidade, versão do modelo, campo para o peso real.
3. **Histórico**: tabela das predições, erro médio onde houver peso real, exportação em CSV.

Meta de desempenho: p95 abaixo de 1 segundo entre o envio da foto e a resposta, na máquina local.

### Testes

- Paridade: a API devolve, para uma imagem fixa, o mesmo peso que o pipeline de avaliação.
- Integração da API com banco e armazenamento reais do Compose.
- Um teste de ponta a ponta da interface: enviar imagem, ver resultado, registrar peso real, ver no histórico.
- Uma foto com resolução diferente da coleta é recusada com o motivo na tela.

## Fase 3 — Artigo

- **Formato**: LaTeX, modelo de artigos da SBC, em português, em `paper/`, com até 12 páginas no texto principal. A primeira opção de destino é o SBIAgro (Congresso Brasileiro de Agroinformática); a chamada da próxima edição ainda precisa ser conferida. O conteúdo fica separado do modelo: um arquivo por seção, tabelas e figuras como arquivos gerados, referências em BibTeX. Trocar de modelo depois é trocar o arquivo principal.
- **Coleta**: seis dias em 2022 (10/01, 12/02, 29/10, 22/11, 26/11 e 29/11), conforme a data gravada nas fotos.
- **Material suplementar**: as tabelas que não couberem nas 12 páginas (todos os protocolos, as duas máscaras, por espécie) vão para `paper/suplementar/`.
- **Autores**: Mateus Pinto Garcia e Romuere Rodrigues Veloso e Silva (UFPI).
- **Tese**: comparação de métodos de estimativa de peso por imagem única em espécies de piscicultura do Piauí, com avaliação separada por dia de coleta, e a ferramenta resultante.
- **Seções**: Introdução; Trabalhos Relacionados; Materiais e Métodos (coleta, anotação, protocolos); Experimentos; Resultados e Discussão; Ferramenta; Limitações; Conclusão.
- **Reprodutibilidade**: um script lê `results/` e gera todas as tabelas e figuras. Nenhum número é digitado à mão.
- **Referências**: cada entrada do `.bib` é conferida pelo DOI ou pela página do periódico antes de entrar. Os links da pesquisa do ChatGPT são ponto de partida, não fonte; parte deles não sobreviveu ao compartilhamento.
- **Limitações declaradas**: 244 peixes, uma propriedade, uma câmera, desbalanceamento de espécies, sem amostra independente nova, escala não conferida por referência nas fotos da coleta, um peso e duas medidas de fita suspeitos.
- **Entrega**: PDF do artigo e PDF do suplementar compilados, completos, prontos para sua revisão e a do orientador.

O trabalho relacionado pode ser escrito em paralelo à Fase 1. Resultados e Ferramenta dependem das fases anteriores.

## Organização do repositório

```
fishweight/        pacote Python: dados, atributos, modelos, avaliação, inferência (vem de src/)
experiments/       scripts executáveis E0–F2 e suas configurações
results/           predições e métricas por experimento
artifacts/         modelos exportados + metadados
apps/api/          FastAPI
apps/web/          Next.js
paper/             LaTeX do artigo
docs/              pesquisa, relatórios, este desenho
tests/
docker-compose.yml
pyproject.toml
```

`src/` é renomeado para `fishweight/` e corrigido (nada o importa hoje). `AGENTS.md` é atualizado ao fim de cada fase.

## Decisões minhas (não suas) — conteste se discordar

1. **PyTorch em vez de TensorFlow** para tudo que for novo, com a U-Net treinada de novo. Motivo: um só framework, exportação ONNX direta, e os modelos da pesquisa (DINOv2, ConvNeXt) estão nesse ecossistema.
2. **O código original da monografia fica intocado**, reunido em `monografia/` (notebooks, `commons.py`, `feature_extraction.py`, `handcraft_segmentation/`, `yolo_v8_instance_seg/`, `models/unet_model.h5`). O PDF e as figuras ficam em `docs/monografia/` e as duas tabelas de atributos usadas na reprodução, em `data/monografia/`. O trabalho novo é feito em scripts.
3. **E0 (alométrico) foi acrescentado** à lista E1–E5 da pesquisa, por ser o piso que os revisores esperam.
4. **Saída auxiliar de comprimento e largura** (E3b) foi acrescentada porque você já tem essas medidas.
5. **Sem vídeo ao vivo e sem Prometheus/Grafana** na primeira versão.
6. **Câmera nativa em vez de vídeo do navegador** na tela de captura, para manter a mesma resolução das fotos de treino.
7. **Só concorre a vencedor o modelo que parte apenas da foto**, avaliado em P3 com máscara predita.
8. **Decidido pelo autor durante a execução:** o treino das redes não roda no MacBook. O código é escrito e testado localmente e executado em Kaggle, Colab ou PC com GPU por `remote/executar.ipynb`. A restrição "Treino em PyTorch com aceleração MPS" acima deixa de valer.

## Deixado para quem implementar

- Hiperparâmetros, resolução de entrada das redes, número de épocas.
- Biblioteca de componentes e visual da interface.
- Estrutura interna dos módulos e nomes de funções.
- Ferramenta de migração do banco e cliente S3.
- Como compilar o LaTeX localmente.
