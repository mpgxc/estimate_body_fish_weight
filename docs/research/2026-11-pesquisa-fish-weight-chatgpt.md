# Pesquisa: Fish Weight Prediction por imagem

Transcrição limpa da conversa compartilhada em https://chatgpt.com/share/6afa3247-1e78-83e8-b256-11d81307c043, salva em 2026-11-15. É material de apoio gerado por IA: títulos, números e links precisam ser conferidos na fonte antes de serem citados. Links marcados como "link não preservado" vinham de referências internas da conversa que não sobrevivem ao compartilhamento.

## Pergunta

pesquisa para mim artigos e projetos github e kaggle sobre

# Fish Weight Prediction

Porém quero que as tecnicas sejam apartir de imagens de peixes.

## Resposta

Encontrei uma linha de pesquisa bastante interessante: **Fish Weight Estimation using Computer Vision**, na qual o peso do peixe é estimado diretamente a partir de fotografias ou vídeos, utilizando visão computacional e modelos de Machine Learning.

Um detalhe importante: existem duas abordagens diferentes:

1. **Image-to-Weight:** a rede neural recebe a imagem do peixe e estima diretamente seu peso.
2. **Image-to-Measurements-to-Weight:** o sistema detecta e segmenta o peixe, extrai medidas corporais (comprimento, largura, altura) e utiliza regressão para estimar o peso.

A segunda abordagem tem bastante relevância para aplicações reais de aquicultura, porque permite interpretar melhor o resultado e calibrar o modelo com medições reais.

Encontrei pesquisas científicas, implementações no GitHub e projetos que podem servir como ponto de partida.

## 1. Artigos científicos — os mais relevantes

Separei os trabalhos que efetivamente utilizam imagens como entrada ou extraem características visuais para estimar peso.

      **1. Multi-stage image-based approach for fish detection and weight estimation**

Prioridade máxima
2025 · Biosystems Engineering
      - Pipeline de detecção, classificação e regressão de peso.
      - Dataset com 1.086 imagens e 2.216 peixes.
      - MAPE-macro de 18,06%.
      - Compara modelos de regressão genéricos e específicos por espécie.

      **Por que estudar:** é um trabalho muito próximo do problema proposto e apresenta uma arquitetura modular.

Artigo e PDF gratuito no Zenodo (link não preservado) · [DOI](https://doi.org/10.1016/j.biosystemseng.2025.104239)

      **2. Deep Learning image segmentation for extraction of fish body measurements and prediction of body weight and carcass traits in Nile tilapia**

Muito relevante
2020 · Computers and Electronics in Agriculture
      - Segmentação semântica de imagens.
      - Extração automática de medidas corporais.
      - Predição de peso e características de carcaça.
      - Aplicação em tilápia-do-Nilo.

      **Por que estudar:** especialmente interessante se o objetivo for trabalhar com tilápias e estimativa de peso a partir da morfologia.

Acessar artigo científico (link não preservado) · [DOI](https://doi.org/10.1016/j.compag.2020.105274)

      **3. Applications of deep convolutional neural networks to predict length, circumference, and weight from mostly dewatered images of fish**

2020 · Ecology and Evolution
      - CNN para regressão direta de comprimento, circunferência e peso.
      - 22 espécies de peixes.
      - Avalia imagens individuais e conjuntos de múltiplas imagens.
      - Erro percentual absoluto médio de peso: 28,6% em imagem individual e 26,9% usando ensemble.

      **Por que estudar:** é uma referência importante para comparar regressão direta de imagens contra extração de medidas.

Artigo completo gratuito (link não preservado) · [DOI](https://doi.org/10.1002/ece3.6618)

      **4. An affordable and easy-to-use tool for automatic fish length and weight estimation in mariculture**

2022 · Aquaculture / maricultura
      - Estimativa automática de comprimento e peso.
      - Detecção de pontos anatômicos.
      - Calibração por câmeras estéreo.
      - Uso de imagens reais e aumento de dados.

      **Por que estudar:** apresenta uma alternativa prática para medição não invasiva em ambientes de aquicultura.

Ler artigo completo (link não preservado)

      **5. AquaFishNet: A Binocular Vision-Based Method for Fish Body Mass Estimation**

2026 · Fishes / MDPI
      - Visão binocular.
      - Segmentação do corpo.
      - Estimativa de comprimento por visão estéreo.
      - Regressão de massa corporal.
      - \(R^2 = 0,9851\), MAE de 12,92 g.

      **Por que estudar:** é uma referência recente para estimativa de massa usando geometria 3D e redes neurais.

Artigo completo (link não preservado)

## 2. Projetos GitHub — implementações práticas

Aqui estão os repositórios que considero mais interessantes para analisar código, arquitetura, modelos e pipelines.

    **1. AI-Powered Precision Measurement of Fish Size and Weight**

Um dos melhores pontos de partida

Python · CNN · XGBoost · Segmentação
    - Aproximadamente 900 imagens de salmões com comprimento e peso medidos.
    - Extrai comprimento, altura e área do peixe.
    - Compara Feedforward Neural Network e XGBoost.
    - Possui scripts de treinamento, inferência e interface web.
    - Utiliza calibração visual.

    **Atenção:** o dataset não está diretamente disponível para download; o autor solicita contato.

GitHub — código-fonte (link não preservado)

    **2. fish-weight-estimation**

Arquitetura interessante

Python · YOLOv12 · UNet · Gradient Boosting
    - Câmeras com duas perspectivas: superior e frontal.
    - YOLO para localização.
    - UNet com encoder MiT-B0 para segmentação.
    - Extração de descritores geométricos.
    - Modelo de regressão para estimar massa.
    - Dataset próprio com 16 peixes e 13.780 imagens.

O repositório declara MAE de 3,54 g. Entretanto, a amostra é pequena e o resultado precisa ser validado independentemente.

GitHub — implementação (link não preservado)

    **3. fish-length-estimation**

Python · YOLOv8 · OpenCV · Segmentação
    - Detecta o peixe.
    - Mede comprimento utilizando máscara ou bounding box.
    - Utiliza objeto de referência para converter pixels em centímetros.
    - Aplica relação comprimento-peso:

\[
W = a \cdot L^b
\]

    - Permite calibrar os parâmetros com peixes pesados manualmente.

    **Por que estudar:** implementação simples, modular e boa para construir um baseline.

GitHub — implementação (link não preservado)

    **4. Fish Length Predictor**

Python · DINOv2 · XGBoost · AutoML
    - Utiliza embeddings visuais do DINOv2.
    - Treina regressor XGBoost.
    - Predição de comprimento diretamente das imagens.
    - Permite treinar com imagens próprias e CSV de medidas.

    **Por que estudar:** uma abordagem moderna de transfer learning, sem precisar começar treinando uma CNN do zero. Pode ser adaptada para peso.

GitHub — implementação (link não preservado)

    **5. Fish Metrics Estimation Pipeline**

Python · YOLOv8/YOLOv12 · OpenCV · Regressão polinomial
    - Detecção de peixes.
    - Estimativa de comprimento e largura.
    - Calibração com marcador ArUco.
    - Regressão polinomial para peso.
    - Processamento de vídeos.

GitHub — implementação (link não preservado)

**Minha avaliação:** os projetos 1 e 2 são os mais interessantes para estudar a relação entre características visuais e peso. O projeto 3 é um bom baseline para começar uma implementação própria.

## 3. Kaggle — datasets e notebooks

Aqui existe uma ressalva importante: nas buscas realizadas, não encontrei um notebook Kaggle suficientemente consistente que faça exatamente a estimativa de peso a partir de fotografias de peixes com peso real associado a cada imagem.

É importante não confundir dois problemas:

- **Fish Market Dataset:** possui comprimento, altura, largura, espécie e peso, mas é essencialmente um dataset tabular. Não é uma solução de estimativa baseada em imagens.
- **Fish Image Classification:** possui imagens e classes de espécies, mas normalmente não contém o peso real de cada exemplar.

Para o seu problema, precisamos de datasets com uma associação semelhante a esta:

| image | species | length_cm | width_cm | weight_g |
|---|---|---:|---:|---:|
| fish_001.jpg | Tilapia | 22.4 | 6.2 | 320 |
| fish_002.jpg | Tilapia | 25.1 | 7.0 | 410 |
| fish_003.jpg | Tilapia | 19.8 | 5.4 | 245 |

Esses dados permitem treinar e avaliar regressão visual.

Links para continuar a exploração:

- [Kaggle — Fish Images](https://www.kaggle.com/datasets?search=fish+images)
- [Kaggle — Fish Weight](https://www.kaggle.com/datasets?search=fish+weight)
- [Kaggle — Fish Weight Prediction notebooks](https://www.kaggle.com/code?search=fish+weight+prediction)
- [Kaggle — Fish Classification](https://www.kaggle.com/datasets?search=fish+classification)

## 4. Datasets científicos que merecem atenção

Na minha avaliação, vale ampliar a busca para além do Kaggle. Alguns dos datasets mais interessantes estão associados diretamente a publicações científicas.

| Dataset / fonte | Conteúdo | Aplicação |
|---|---|---|
| Fish Detection and Weight Estimation | 1.086 imagens, 2.216 exemplares | Detecção e peso |
| USGS Great Lakes Fish | Imagens e dados biométricos | Regressão visual |
| Salmon dataset — projeto Anran | Aproximadamente 900 imagens | Comprimento e peso |
| Dataset do projeto YOLO-UNet | 13.780 imagens, 16 peixes | Segmentação e massa |

**Links:**

1. [Fish Detection and Weight Estimation — Zenodo](https://zenodo.org/records/22096222).
2. [USGS ScienceBase](https://www.sciencebase.gov/catalog/) — procure pelo título *Image and biometric data for fish from Great Lakes tributaries collected during spring 2019*.
3. [Salmon dataset — GitHub](https://github.com/anran-github/AI-Powered-Precision-Measurement-of-Fish-Size-and-Weight).
4. [Fish Weight Estimation — GitHub](https://github.com/nivedkm/fish-weight-estimation).

O primeiro dataset é particularmente interessante porque acompanha uma publicação de 2025 com avaliação quantitativa de detecção e regressão.

## 5. Técnicas de Machine Learning que eu investigaria

Além dos projetos encontrados, recomendo comparar cinco abordagens.

  **Abordagem A — CNN com regressão direta**

Imagem → EfficientNet / ResNet / ConvNeXt → Fully Connected → Peso (g).

  - Transfer learning com ImageNet.
  - EfficientNet-B0/B3 como baseline.
  - Loss: MAE ou Huber Loss.

Baseline obrigatório

  **Abordagem B — Vision Transformers**

Imagem → DINOv2 / ViT → Embeddings → MLP ou XGBoost → Peso.

  - Extrai características visuais sem treinamento completo da arquitetura.
  - Permite comparar embeddings com regressão tradicional.
  - Interessante para datasets pequenos.

  **Abordagem C — Object Detection + Segmentation**

Imagem → YOLO → Segmentação → Características morfológicas → Regressão.

  - YOLO para detecção.
  - U-Net ou Mask R-CNN para segmentação.
  - Extração de área, comprimento, largura e proporções.
  - XGBoost, CatBoost ou MLP para peso.
Minha recomendação

  **Abordagem D — Visão estéreo e reconstrução 3D**

Câmeras sincronizadas → Stereo Vision → Reconstrução geométrica → Volume corporal → Peso.

  - Duas câmeras em ângulos diferentes.
  - Estimativa de comprimento, largura e altura.
  - Reconstrução volumétrica.
  - Regressão de massa.

  **Abordagem E — Multimodal**

Imagem + espécie + medidas conhecidas → Encoder visual + atributos tabulares → Regressão de peso.

É interessante quando a espécie influencia significativamente a relação entre formato corporal e peso.

## 6. Como eu estruturaria um experimento

Para não depender de uma única técnica, faria uma comparação controlada:

| Experimento | Entrada | Modelo |
|---|---|---|
| E1 | Imagem RGB | EfficientNet |
| E2 | Imagem RGB | ResNet50 |
| E3 | Embeddings da imagem | DINOv2 + XGBoost |
| E4 | Máscara + medidas | CatBoost |
| E5 | Duas perspectivas | YOLO + U-Net + XGBoost |
| E6 | Imagem + espécie | ViT + MLP |

Métricas que utilizaria:

- MAE (g): erro médio absoluto em gramas.
- RMSE (g): penaliza erros maiores.
- MAPE (%): erro percentual médio.
- R²: capacidade explicativa do modelo.
- Bias: tendência de superestimar ou subestimar o peso.

**Um cuidado metodológico fundamental:** as imagens do mesmo peixe não podem aparecer simultaneamente no treino e no teste. Caso contrário, o modelo pode memorizar características individuais e apresentar resultados artificialmente bons. Utilize GroupKFold ou Leave-One-Fish-Out.

## 7. Minha conclusão

Se eu fosse iniciar esse projeto, seguiria esta ordem:

1. Estudaria o artigo de 2025 da Wageningen University.
2. Analisaria o projeto AI-Powered Precision Measurement.
3. Reproduziria o baseline com DINOv2 + XGBoost.
4. Implementaria segmentação com YOLO + U-Net.
5. Compararia regressão direta contra regressão baseada em características morfológicas.
6. Avaliaria generalização entre espécies e diferentes condições de captura.

Um ponto importante: **comprimento não é suficiente para determinar peso com precisão**. Dois peixes da mesma espécie e comprimento podem ter massas diferentes por causa da condição corporal. Área, largura, altura e volume são características que merecem investigação.

---

## 8. Uma informação que pode mudar completamente a pesquisa

Quero entender o cenário de aplicação, porque isso determina quais artigos, datasets e arquiteturas são realmente úteis.

{@body const [scenario,setScenario] = DIL.useState("aquaculture")}
{@body const [camera,setCamera] = DIL.useState("single")}
{@body const [species,setSpecies] = DIL.useState("tilapia")}
{@body const [goal,setGoal] = DIL.useState("both")}

  **1. Onde os peixes serão fotografados?**

- Aquicultura — peixes em tanques ou viveiros
- Peixes fora da água, sobre uma superfície
- Debaixo d'água, em ambiente natural
- Esteira industrial / processamento de pescado
- Imagens genéricas da internet

  **2. Qual será a configuração de captura?**

- Uma fotografia / uma câmera
- Duas ou mais câmeras
- Vídeo / câmera contínua
- Ainda não definido

  **3. Qual espécie de peixe?**

- Tilápia
- Salmão
- Tambaqui
- Múltiplas espécies
- Ainda não definido

  **4. Qual é o objetivo da pesquisa?**

- Predizer peso diretamente pela imagem
- Extrair medidas e calcular peso
- Comparar as duas abordagens

Com essas respostas, consigo direcionar a próxima investigação para encontrar trabalhos realmente reproduzíveis, com datasets de imagens e pesos reais, evitando projetos que apenas fazem classificação de espécies.

Também posso acompanhar semanalmente novos artigos, datasets e repositórios sobre estimativa de peso de peixes por visão computacional, trazendo uma atualização com os achados relevantes.

## Pergunta

Refine a pesquisa Fish Weight Prediction com estas condições: cenário market, captura single, espécies multiple, objetivo direct. Busque artigos, datasets disponíveis, notebooks Kaggle e implementações GitHub, priorizando trabalhos reproduzíveis.

## Resposta

# Pesquisa refinada: Fish Weight Prediction por imagem

Pesquisa direcionada   Atualizada em novembro de 2026

Com as condições definidas, o problema fica bem mais específico:

  **Objetivo do projeto**

Estimar o peso de um peixe diretamente a partir de uma única fotografia RGB, considerando diferentes espécies e condições reais de apresentação em mercados de pescado.

Ambiente
      **Mercado de peixes**

Entrada
      **Uma fotografia RGB**

Espécies
      **Múltiplas**

Saída
      **Peso em gramas**

Não utilizar balanças conectadas, câmeras estéreo, sensores de profundidade ou medições externas como entrada da inferência.

A pesquisa encontrou algo importante: existem trabalhos que já fazem exatamente a estimativa de peso por imagem RGB, inclusive com múltiplas espécies. Também encontrei datasets de mercado que podem ajudar a construir um modelo mais robusto.

Minha principal recomendação é combinar os datasets de peso real com datasets de imagens de mercado, em vez de depender exclusivamente de um único conjunto.

## 1. Artigos científicos — prioridade de leitura

### 1. Automatic Weight Estimation of Harvested Fish from Images (2019)

Correspondência muito alta

**Autores:** Konovalov, Saleh, Efremova, Domingos e Jerry.

- Aproximadamente 2.500 fotografias de peixes.
- Peso real associado às imagens.
- Espécie: Asian seabass / Barramundi.
- Utiliza LinkNet-34 para segmentação.
- Compara estimativa baseada na área do peixe com regressão direta da imagem.
- A regressão direta por imagem alcançou MAPE de 4,28% no teste.

**O que interessa:** é uma das referências mais próximas do seu objetivo, pois testa diretamente CNN para regressão de peso a partir de imagens.

O resultado de 4,28% não deve ser generalizado para outras espécies: a pesquisa foi realizada em uma única espécie e com uma distribuição de dados específica.

- Artigo completo no arXiv (link não preservado)
- [Publicação DOI](https://doi.org/10.1109/DICTA47822.2019.8945971)
- James Cook University — publicação e informações (link não preservado)

### 2. An integrated end-to-end deep neural network for automated detection of discarded fish species and their weight estimation (2023)

Correspondência muito alta

Este é provavelmente o artigo mais importante para começar.

**Características:**
- Uma única imagem RGB como entrada.
- Detecção dos peixes.
- Classificação de espécies.
- Estimativa de peso individual.
- Rede neural end-to-end.
- Nove espécies.
- Dataset FDWE com 1.086 imagens e 2.216 exemplares.

Resultados no teste:
- MAE: 29,74 g.
- RMSE: 44,69 g.
- MAPE: 23,78%.

**Por que é relevante:** não exige uma arquitetura de câmeras especiais. Trabalha com fotografia colorida e associa o peso ao peixe detectado.

Também é uma boa referência para o caso em que uma fotografia contém mais de um peixe.

Links:
- Artigo científico — ICES Journal of Marine Science (link não preservado)
- Dataset FDWE — 4TU.ResearchData (link não preservado)
- Wageningen University — descrição dos dados e arquivos (link não preservado)

### 3. Multi-stage image-based approach for fish detection and weight estimation (2025)

Prioridade máxima

**Autores:** Córdova, Sokolova, van Helmond, Mencarelli e Kootstra.

É a evolução metodológica da linha anterior.

Em vez de obrigar uma única rede a resolver tudo, separa:

1. Detecção.
2. Classificação.
3. Estimativa de peso.

Uma das coisas mais interessantes é a comparação entre:

- Regressor genérico: aprende a estimar peso independentemente da espécie.
- Regressores específicos: cada espécie possui seu próprio modelo de peso.

Resultados:
- F1-macro: 92,72%.
- MAPE-macro: 18,06%.
- Redução de aproximadamente 60% no MAPE em comparação com a melhor abordagem single-stage.

O estudo também observa queda significativa de desempenho quando a oclusão ultrapassa 60%.

**Minha recomendação:** esse trabalho deve orientar a arquitetura experimental, principalmente a comparação entre modelo universal e modelos especializados.

Links:
- Página oficial da Wageningen University (link não preservado)
- PDF e registro no Zenodo (link não preservado)
- [DOI — Biosystems Engineering](https://doi.org/10.1016/j.biosystemseng.2025.104239)

### 4. Applications of deep convolutional neural networks to predict length, circumference, and weight from mostly dewatered images of fish (2020)

**Características:**
- 694 peixes.
- 22 espécies.
- Imagens de peixes fora d'água.
- Predição de peso, comprimento e circunferência.
- Comparação entre regressão de imagem individual e ensemble de imagens.

Resultado de peso:
- Imagem individual: MAPE de 28,6%.
- Ensemble: MAPE de 26,9%.
- Modelo multi-target: erro percentual médio de 24%.

**Por que estudar:** oferece uma referência para generalização entre espécies e para avaliar a dificuldade de estimar massa com uma única imagem.

Artigo completo — Ecology and Evolution (link não preservado).

---

## 2. Datasets disponíveis — seleção para o seu cenário

Aqui está a parte mais importante da pesquisa.

Separei os datasets em duas categorias: aqueles que possuem peso real associado à imagem e aqueles que fornecem fotografias de mercado para melhorar a generalização visual.

### A. Datasets com peso real

      **FDWE — Fish Detection and Weight Estimation**
Prioridade 1

Wageningen University · 2023
    | Característica | Informação |
    |---|---|
    | Imagens | 1.086 |
    | Exemplares | 2.216 |
    | Espécies | 9 |
    | Peso real | Sim |
    | Anotações | Classe, bounding box, peso e oclusão |
    | Formato | YOLO / COCO |

Download público

Arquivo principal de aproximadamente 2,44 GB compactado.

Download do dataset e pesos do modelo (link não preservado).

    **É o melhor ponto de partida para treinamento supervisionado direto.**

      **FishL — Great Lakes**
Prioridade 2

Great Lakes Fishery Commission
    | Característica | Informação |
    |---|---|
    | Imagens | 6.246 |
    | Espécies | Múltiplas |
    | Peso real | Sim, para os exemplares medidos |
    | Comprimento | Disponível |
    | Circunferência | Disponível |
    | Formato | Imagens e scripts |

    **Por que é interessante:** amplia o volume de imagens e permite estudar generalização entre espécies.

O estudo de 2020 utilizou um conjunto de 694 peixes e 22 espécies proveniente dessa linha de aquisição.

Dataset oficial — GLFC (link não preservado) · [Acessar repositório OSF](https://doi.org/10.17605/OSF.IO/KQVG8).

      **Barramundi — Harvested Fish**
Acesso a verificar

James Cook University · 2018–2019
    - Aproximadamente 2.500 imagens com pesos correspondentes no estudo de 2019.
    - Mais de 1.000 exemplares na pesquisa anterior.
    - Possui experimentos de regressão direta e segmentação.
    - O conjunto completo de imagens não está claramente disponibilizado para download público nos registros consultados.

Artigo de 2018 (link não preservado) · Artigo de 2019 (link não preservado).

### B. Datasets de mercado para imagens

Esses conjuntos não possuem peso individual associado a cada fotografia, mas são muito úteis para treinamento visual, fine-tuning, detecção e avaliação de robustez.

      **SylFishBD — 2026**
Prioridade 1

Scientific Data · Junho de 2026
Disponível no Kaggle
    - 9.075 imagens.
    - Nove espécies de peixes de água doce.
    - Uma instância de peixe por imagem.
    - Fotografias reais em mercados.
    - Iluminação variável, gelo, bandejas e fundos desorganizados.
    - Bounding boxes e máscaras de segmentação.
    - Licença CC0.

    **Excelente correspondência com seu cenário de captura.**

Não contém peso real individual: não pode ser usado sozinho para treinar a regressão de peso.

[Kaggle — SylFishBD](https://www.kaggle.com/datasets/shakibabsar42/sylfishbd) · Artigo científico (link não preservado) · [Código de preparação do dataset](https://github.com/sabsar42/sylfishbd-dataset-codes).

      **DeepFish — Fish Market**
Prioridade 2

    - 1.291 imagens.
    - 7.339 exemplares.
    - 59 espécies.
    - Fotografias de bandejas de mercado.
    - Máscaras pixel a pixel.
    - Informações de espécie e medidas de tamanho.

Não possui peso individual para todos os peixes.

    **Interesse:** excelente para treinar o encoder visual a reconhecer morfologias diferentes e segmentar exemplares em ambientes de mercado.

Download no Zenodo (link não preservado) · Artigo Scientific Data (link não preservado).

      **BengalDeltaFish — 2025**
Complementar

    - 4.560 imagens.
    - 33 espécies.
    - Fotografias reais em sete mercados de Bangladesh.
    - Variações de iluminação, ângulo, gelo, bandejas e fundos.
    - Anotações para detecção.

Licença CC BY 4.0.

Não contém peso individual como variável supervisionada.

Download no Mendeley Data (link não preservado) · [GitHub oficial](https://github.com/281096alif/BengalDeltaFish).

    **DeepFish2 — 2022**

    - Imagens de mercados espanhóis.
    - Espécies, máscaras e bounding boxes.
    - Medidas de comprimento dos exemplares.

Download no Zenodo (link não preservado).

## 3. Kaggle — notebooks e projetos

Encontrei notebooks de Fish Weight Prediction, mas existe uma limitação: os notebooks mais populares trabalham com o dataset clássico Fish Market, que contém peso e medidas corporais, não imagens.

| Notebook | Técnica | Serve para imagem? |
|---|---|---|
| Multiple Linear Regression — Melih Akdag (link não preservado) | Regressão linear |  Não  |
| Fish Weight Prediction — Satish Gunjal (link não preservado) | Regressão linear |  Não  |
| Polynomial Linear Regression (link não preservado) | Regressão polinomial |  Não  |
| Aqualytics — AI Fish Weight (link não preservado) | Machine Learning |  Verificar dados  |
| [SylFishBD](https://www.kaggle.com/datasets/shakibabsar42/sylfishbd) | Dataset de imagens |  Sim, sem peso  |

Minha sugestão é não utilizar os notebooks clássicos como solução principal. Eles podem servir para comparar regressões tabulares, mas não resolvem a predição visual.

### Kaggle para experimentos próprios

Eu utilizaria o SylFishBD como base para criar um notebook de fine-tuning de modelos visuais, e o FDWE como dataset supervisionado de peso.

Assim, você teria dois experimentos independentes:

- Modelo treinado exclusivamente com imagens que possuem peso real.
- Modelo com pré-treinamento visual adicional em fotografias de mercado.

---

## 4. GitHub — implementações que valem estudar

Aqui estão os projetos que consegui identificar e verificar.

    **1. AI-Powered Precision Measurement of Fish Size and Weight**

Mais próximo da regressão visual

Abrir repositório GitHub (link não preservado)

    - Cerca de 900 imagens de salmões.
    - Comprimento e peso medidos manualmente.
    - Compara Feedforward Neural Network e XGBoost.
    - Possui aplicação Python e JavaScript.
    - Utiliza calibração visual.

Limitação: o dataset é disponibilizado mediante solicitação ao autor, e a implementação utiliza características geométricas intermediárias. Não é um benchmark de regressão RGB pura.

    **2. Fish Weight Estimation — YOLO + U-Net**

Abrir repositório GitHub (link não preservado)

    - YOLOv12.
    - U-Net com encoder MiT-B0.
    - Extração de características geométricas.
    - Gradient Boosting.
    - Dataset de 16 peixes e 13.780 imagens.

Não corresponde à captura single

Utiliza câmeras superior e frontal. É útil para estudar segmentação e regressão, mas não atende à sua restrição de uma única fotografia.

    **3. SylFishBD Dataset Codes**

Útil para começar o dataset visual

[Abrir repositório GitHub](https://github.com/sabsar42/sylfishbd-dataset-codes)

    - Preparação de imagens.
    - Organização de anotações.
    - Máscaras de segmentação.
    - Estruturação do dataset.

Não implementa estimativa de peso, mas oferece uma base reproduzível para preparação dos dados.

    **4. Fish Metrics Estimation Pipeline**

Abrir repositório GitHub (link não preservado)

    - YOLOv12.
    - OpenCV.
    - Regressão polinomial.
    - Estimativa de comprimento e peso.

Limitação: o projeto é orientado a vídeo e trabalha com camarão, não com múltiplas espécies de peixes em fotografia única.

**Conclusão sobre GitHub:** ainda não encontrei uma implementação pública que reúna todas as condições simultaneamente: fotografia RGB única, múltiplas espécies, peso individual real e cenário de mercado. O FDWE é a melhor base para reproduzir a pesquisa científica, enquanto SylFishBD fornece a distribuição visual mais próxima do ambiente pretendido.

---

## 5. Arquitetura que eu investigaria

Para o seu problema, eu faria um experimento com regressão direta, sem depender de comprimento ou largura como entrada.

  **Pipeline de inferência**

    **Fotografia RGB**
Uma imagem capturada em mercado

    **Visual Encoder**
EfficientNet / ConvNeXt / DINOv2
Transfer Learning

    **Feature Representation**
Características visuais aprendidas diretamente da imagem

    **Regression Head**
MLP com saída contínua

425 g
    **Peso estimado**
Exemplo ilustrativo

### Modelos que compararia

| Modelo | Papel no experimento |
|---|---|
| ResNet50 | Baseline CNN |
| EfficientNet-B3 | CNN eficiente |
| ConvNeXt-Tiny | Representação visual moderna |
| ViT-B/16 | Vision Transformer |
| DINOv2 + MLP | Embeddings visuais pré-treinados |

Minha escolha inicial seria EfficientNet-B3, ConvNeXt-Tiny e DINOv2.

### Uma questão importante: espécie conhecida ou desconhecida?

Eu testaria duas variantes:

- **Species-agnostic:** imagem → peso. O modelo aprende a estimar peso sem receber a espécie.
- **Species-aware:** imagem → representação visual → peso, com uma tarefa auxiliar de classificação de espécie.

A segunda pode ajudar a rede a aprender diferenças morfológicas entre espécies, sem exigir que a aplicação receba manualmente a espécie.

## 6. Estratégia experimental e validação

Eu estruturaria os experimentos assim:

| Experimento | Dataset | Objetivo |
|---|---|---|
| E1 | FDWE | Reproduzir regressão direta |
| E2 | FishL | Generalização entre espécies |
| E3 | FDWE + FishL | Aumentar diversidade |
| E4 | FDWE + SylFishBD | Pré-treinamento visual e fine-tuning |
| E5 | FDWE + DeepFish | Robustez visual de mercado |

Importante: SylFishBD e DeepFish não fornecem peso individual para a supervisão. Não devemos inventar rótulos de peso para esses exemplares. Eles podem ser usados em pré-treinamento de classificação, segmentação ou representação visual.

**Validação obrigatória:**

- Separar imagens por indivíduo, não apenas aleatoriamente por fotografia.
- Testar espécies vistas e não vistas durante o treinamento.
- Avaliar separadamente diferentes ambientes e condições de iluminação.
- Medir MAE, RMSE, MAPE e R².
- Comparar erro por espécie e faixa de peso.

Um modelo que alcança bom MAE em peixes de 100–300 g pode falhar completamente em peixes de 2 kg. Por isso, a distribuição de peso é tão importante quanto a quantidade de imagens.

---

## 7. Minha seleção final

Ordem de prioridade

      **FDWE**

Dataset com peso real, imagens e múltiplas espécies. Base principal de treinamento.

      **Artigo end-to-end de 2023**

Referência para regressão direta de peso por imagem RGB.

      **SylFishBD**

Melhor complemento encontrado para fotografias reais de mercados, com uma instância por imagem.

      **FishL**

Expansão para diferentes espécies e avaliação de generalização.

      **Artigo multi-stage de 2025**

Referência para comparar modelos universais e específicos por espécie.

**Minha conclusão:** existe material científico suficiente para construir uma prova de conceito séria. O maior gargalo não será a arquitetura neural, mas conseguir um dataset com fotografias de mercado e peso real medido para uma variedade grande de espécies.

Eu começaria reproduzindo o FDWE e, em paralelo, estruturaria uma base própria de fotografias de mercado com pesagem individual.

---

## 8. Próximo passo: transformar a pesquisa em um projeto reproduzível

Posso estruturar um projeto experimental em Python/PyTorch com:

- Download e preparação dos datasets públicos.
- Padronização de imagens e labels.
- Treinamento de EfficientNet, ConvNeXt e DINOv2.
- Regressão direta de peso.
- Comparação species-agnostic versus species-aware.
- Avaliação com MAE, RMSE, MAPE e R².
- Relatórios e gráficos comparativos.
- Inferência sobre uma fotografia enviada.

Minha sugestão é começar pelo FDWE + SylFishBD, mantendo os conjuntos separados para evitar vazamento de dados.

Também posso acompanhar semanalmente novos datasets, artigos e implementações reproduzíveis nessa área, destacando apenas descobertas que acrescentem algo à pesquisa.

## Pergunta

Monta um documento com todos os resultados das buscas, ao final adiciona uma planilha com tabela de trabalhos, com titulo, link, tecnica aplicada.

## Resposta

Preparei os dois arquivos, consolidando os resultados das pesquisas e organizando o material para servir como referência técnica do projeto.

      **Documento completo — Fish Weight Prediction**
DOCX · Artigos, datasets, GitHub, Kaggle, técnicas e arquitetura experimental.
Inclui tabela consolidada no anexo

[ Baixar documento Word](sandbox:/mnt/data/pesquisa_fish_weight_prediction_por_imagens.docx)

      **Catálogo de trabalhos**
XLSX · 25 registros, links clicáveis, técnicas, datasets e limitações.
Filtros e tabela estruturada

[ Baixar planilha Excel](sandbox:/mnt/data/catalogo_trabalhos_fish_weight_prediction.xlsx)

O documento também destaca a principal oportunidade identificada: combinar **FDWE (peso real)** com **SylFishBD e DeepFish (imagens de mercados reais)** para desenvolver um modelo de regressão direta por imagem.

Incluí as ressalvas sobre disponibilidade dos datasets, reprodutibilidade e generalização entre espécies.

## Pergunta

Eu tenho minha propria pesquisa [https://github.com/mpgxc/estimate_body_fish_weight](https://github.com/mpgxc/estimate_body_fish_weight)

como poderiamos melhorarla com base nos conhecimentos desses docuemntos pesquisado, e como formatar isso em uma ferramenta completa que execute em realtime essa inferencia.

## Resposta

Eu estruturaria isso como a evolução de um experimento de visão computacional para um **produto de inferência de peso de peixes em tempo real**.

A mudança mais importante é separar três problemas que costumam ficar misturados:

1. **Precisão do modelo:** estimar o peso real a partir de uma única imagem.
2. **Robustez operacional:** funcionar com diferentes espécies, câmeras, iluminação, fundos e condições de captura.
3. **Experiência de uso:** permitir que alguém fotografe um peixe e receba a estimativa imediatamente, com confiança, histórico e possibilidade de correção por pesagem real.

A ideia não é simplesmente colocar uma API em volta do modelo. É construir um fluxo completo de captura, inferência, calibração e evolução do modelo.

Vou olhar a estrutura da sua implementação para propor melhorias concretas, em vez de ficar apenas em uma arquitetura genérica.

## 1. O que você já construiu

Analisei o README, a estrutura proposta e os módulos centrais do repositório.

Seu trabalho já tem uma base interessante e, principalmente, um diferencial que eu preservaria: **você possui uma coleta própria de 245 peixes, com imagens e pesos reais, obtida em uma propriedade no interior do Piauí**. O README registra um resultado de \(R^2 = 0{,}90\). Isso é um ponto de partida valioso, embora não seja suficiente, isoladamente, para afirmar que o modelo generaliza para outras espécies ou ambientes. [Repositório](https://github.com/mpgxc/estimate_body_fish_weight)

O pipeline atual é, essencialmente:

    **Imagem do peixe**

    **U-Net + MobileNetV2**

Segmentação do corpo do peixe

    **Extração de características**

Área, perímetro, eixos, excentricidade, solidez etc.

    **Regressão tradicional**

Gradient Boosting, Random Forest, Ridge, Lasso e outros

    **Peso estimado**

Isso está bem alinhado com uma abordagem clássica de visão computacional: segmentar, extrair descritores morfológicos e aplicar regressão.

Eu não descartaria essa abordagem. Ela deve continuar sendo seu baseline experimental e uma das alternativas de inferência.

Alguns pontos que merecem atenção no código atual:

- A função `extract_features_from_images` em `src/feature_extraction.py` utiliza `np.asarray`, mas o módulo não importa NumPy (`import numpy as np`).
- A extração de propriedades depende de máscaras redimensionadas para 224 × 224. Isso torna as medidas geométricas relativas à imagem, não dimensões físicas reais.
- O `load_dataset` une anotações e características por índices derivados dos nomes dos arquivos. É importante validar explicitamente a correspondência e detectar registros ausentes ou duplicados.
- O `KFold` aleatório pode colocar imagens do mesmo peixe ou da mesma sessão de captura em treino e teste. Se houver essa relação, o resultado fica otimista. A validação deve separar indivíduos e, idealmente, sessões ou lotes.
- O MAPE atual divide pelo peso real sem tratar valores zero ou próximos de zero.

Esses pontos não invalidam o experimento, mas precisam ser resolvidos antes de transformar o resultado em uma ferramenta confiável.

## 2. Como eu evoluiria o modelo

A principal mudança seria adicionar uma segunda linha de pesquisa: **regressão direta de peso a partir dos pixels RGB**.

O estudo de Konovalov et al. demonstrou essa possibilidade em barramundi: a CNN de regressão direta alcançou MAPE de 4,28% no conjunto de teste, enquanto também foram estudadas abordagens baseadas em segmentação e área. É um resultado de uma espécie e de condições específicas, não uma promessa de desempenho para seu cenário.

Outro estudo, com 694 peixes de 22 espécies, avaliou regressão de peso a partir de uma única imagem. O MAPE reportado para a estimativa de peso em imagem individual foi de 28,6%, mostrando como a generalização entre espécies e condições é difícil.

Minha proposta é comparar três abordagens.

### Abordagem A — manter seu modelo atual

Baseline

U-Net → features geométricas → Gradient Boosting / Random Forest.

Vantagens:
- Mais interpretável.
- Mais simples de depurar.
- Permite analisar quais características morfológicas influenciam a predição.
- Pode funcionar muito bem em um cenário de captura controlado.

Eu manteria esse pipeline e salvaria o melhor modelo treinado com todos os pré-processamentos.

### Abordagem B — regressão visual direta

Nova linha experimental

    **Imagem RGB — 224 × 224 ou 384 × 384**

    **Visual Encoder**

EfficientNet / ConvNeXt / DINOv2

    **Regression Head — MLP**

    **Peso estimado em gramas**

Eu começaria com:

| Modelo | Papel |
|---|---|
| EfficientNet-B3 | Primeiro modelo de regressão direta |
| ConvNeXt-Tiny | Comparação de arquitetura |
| DINOv2 + MLP | Transfer learning com representação visual pré-treinada |
| ResNet50 | Baseline simples para comparação |

Não começaria treinando um Transformer do zero. Com apenas 245 indivíduos, transfer learning e regularização são muito mais razoáveis.

O treinamento deve usar:
- Augmentation moderada: rotação, brilho, contraste, pequenas translações e flip horizontal quando biologicamente aceitável.
- Loss Huber ou MAE, comparando também MSE.
- Transformação logarítmica do peso como experimento, especialmente se a distribuição tiver uma cauda longa.
- Early stopping e weight decay.
- Divisão dos dados por indivíduo, não por imagem aleatória.

### Abordagem C — modelo híbrido

Minha recomendação

Essa é a abordagem que eu mais gostaria de testar no seu projeto.

Em vez de escolher entre visão profunda e características geométricas, combinar as duas.

    **Imagem RGB**

        **Branch visual**
Encoder CNN / DINOv2

        **Branch geométrica**
U-Net + features morfológicas

    **Feature Fusion**
Concatenação e projeção das representações

    **Regression Head → Peso**

O modelo visual aprende padrões que talvez não estejam representados nos descritores geométricos. O ramo geométrico preserva características que já se mostraram úteis na sua pesquisa.

É importante comparar o híbrido com os outros dois, pois a fusão também pode aumentar overfitting quando os dados são poucos.

## 3. Como aproveitar as pesquisas anteriores

Eu incorporaria quatro referências diretamente ao roadmap:

| Trabalho | O que aproveitar |
|---|---|
| [Konovalov et al. (2019)](https://researchonline.jcu.edu.au/61472/) | Comparar regressão direta por imagem com estimativa baseada em segmentação |
| [Bravata et al. (2020)](https://doi.org/10.1002/ece3.6618) | Regressão visual com várias espécies e avaliação de imagem individual |
| [FDWE / ICES (2023)](https://academic.oup.com/icesjms/article/80/7/1911/7240285) | Dataset com peso individual, espécie e oclusão |
| [Abordagem multi-stage (2025)](https://www.sciencedirect.com/science/article/pii/S1537511025001758) | Separação entre localização, classificação e estimativa de peso |

O FDWE é particularmente interessante: possui 1.086 imagens, 2.216 indivíduos e nove espécies com pesos individuais anotados. O trabalho de 2025 também demonstra que separar as tarefas pode melhorar o resultado em relação a abordagens integradas.

Há uma distinção importante para o seu objetivo:

- **Dataset de segmentação/classificação:** pode ajudar a reconhecer peixes e adaptar o encoder visual.
- **Dataset de regressão:** precisa conter o peso real associado à imagem ou ao indivíduo.

Não podemos treinar regressão de peso com imagens do DeepFish ou SylFishBD como se elas tivessem pesos conhecidos. Esses conjuntos são úteis para segmentação e adaptação visual, mas não substituem os rótulos de peso.

### O desafio das múltiplas espécies

Eu não começaria com um único modelo universal sem testar o efeito da espécie.

Faria dois experimentos:

1. *Species-agnostic:* imagem → peso.
2. *Species-aware:* imagem → representação visual + classificação auxiliar de espécie → peso.

Na segunda opção, a classificação de espécie é uma saída auxiliar do modelo, não uma informação que o operador precisa fornecer. A regressão continua recebendo a imagem como entrada.

Também testaria cabeças de regressão específicas por espécie, selecionadas pelo classificador, mas somente depois de ter amostras suficientes por espécie.

Se o sistema não reconhecer uma espécie ou encontrar uma espécie fora do conjunto de treinamento, deve sinalizar que a estimativa está fora do domínio validado, em vez de inventar uma confiança alta.

## 4. Transformando em uma ferramenta completa

Eu chamaria a aplicação provisoriamente de **FishWeight AI**.

O produto teria uma interface de captura, uma API de inferência e um serviço responsável por carregar e executar os modelos.

FishWeight AI
Proposta de experiência da ferramenta

      **Captura**
Câmera ou upload de uma fotografia

      **Segmentação**
Visualização do peixe identificado

      **Estimativa**
Peso, intervalo e qualidade da imagem

      **Histórico**
Predições, pesagens e relatórios

### Fluxo de inferência

1. O usuário abre a câmera ou seleciona uma imagem.
2. O sistema valida resolução, iluminação, enquadramento e presença de um peixe.
3. A segmentação identifica o corpo.
4. O modelo de regressão estima o peso.
5. O sistema apresenta o resultado e sua faixa de incerteza.
6. O usuário pode informar o peso real aferido em uma balança.
7. A medição real é armazenada como dado de calibração para futuras versões do modelo.

Uma tela de resultado poderia ser assim:

    **Estimativa de peso**
Inferência concluída

842 g
Peso estimado

Espécie provável
      **Tilápia**

Qualidade da imagem
      **Boa**

Modelo
      **Hybrid v1**

Tempo de inferência
      **Exemplo: 85 ms**

Valores ilustrativos para demonstrar a interface, não são uma predição real.

  **Peso real (balança)**

O peso real deve ser opcional para o fluxo de inferência, mas muito importante para a evolução do produto.

## 5. Arquitetura técnica proposta

Eu manteria o núcleo de machine learning em Python. Não tentaria migrar a inferência para TypeScript ou NestJS: TensorFlow/PyTorch, OpenCV e bibliotecas de visão computacional estão muito mais maduras nesse ecossistema.

Arquitetura de execução

    **Web App — Next.js / React**
Camera API · Upload · Preview · Resultados

HTTPS / WebSocket quando necessário

    **FastAPI — Python**
Validação · Orquestração · API de inferência

        **Image Pipeline**
Resize · Normalize · Quality checks

        **Model Runtime**
U-Net · Encoder · Regressor

    **Prediction Response**
Peso · Espécie · Incerteza · Latência · Versão do modelo

      **PostgreSQL**
Pesagens e histórico

      **S3**
Imagens e artefatos

      **Observability**
Métricas e erros

### Stack sugerida

| Camada | Tecnologia |
|---|---|
| Frontend | Next.js + TypeScript |
| Captura | Web MediaDevices API |
| API | FastAPI |
| Inferência | PyTorch |
| Segmentação | U-Net atual |
| Regressão | EfficientNet / ConvNeXt / híbrido |
| Processamento | OpenCV + NumPy |
| Runtime otimizado | ONNX Runtime |
| Persistência | PostgreSQL |
| Arquivos | S3 |
| Monitoramento | Prometheus + Grafana |
| Empacotamento | Docker |

Para a primeira versão, não usaria microserviços. Um backend FastAPI modular e um processo de inferência com modelos carregados em memória são suficientes.

### Como executar em realtime

Há duas experiências diferentes:

- **Foto em tempo real:** o usuário captura uma imagem e recebe a estimativa rapidamente. É a primeira funcionalidade que eu implementaria.
- **Vídeo contínuo:** a câmera envia frames periodicamente, o sistema detecta o peixe, faz inferência e atualiza o resultado na tela.

No vídeo, não é necessário executar a rede em todos os frames. Podemos selecionar um frame a cada intervalo, detectar mudanças relevantes e reutilizar resultados enquanto o peixe permanecer estável.

Se houver mais de um peixe no quadro, a ferramenta deverá detectar e selecionar cada indivíduo separadamente. Para o seu primeiro escopo, manteria a restrição de um peixe por captura.

**Importante:** uma única imagem RGB não revela toda a profundidade ou espessura do peixe. O modelo pode aprender correlações visuais, mas não medir diretamente o volume real. Por isso, a posição, a perspectiva, o fundo e a apresentação do peixe precisam fazer parte da validação. Uma câmera fixa e uma área de captura padronizada podem melhorar bastante a confiabilidade sem exigir duas câmeras.

## 6. O que eu mudaria primeiro na pesquisa

Antes de desenvolver a interface, eu faria uma rodada de experimentos para descobrir qual modelo realmente merece ir para produção.

| Experimento | Entrada | Saída |
|---|---|---|
| E1 — Atual | Features da U-Net | Peso |
| E2 — CNN | RGB | Peso |
| E3 — CNN + classificação | RGB | Peso + espécie |
| E4 — Híbrido | RGB + features da máscara | Peso |
| E5 — Híbrido multi-espécie | RGB + features + classificação auxiliar | Peso |

Todos devem utilizar exatamente os mesmos conjuntos de treino, validação e teste.

**Métricas obrigatórias:**

- MAE em gramas: erro médio absoluto.
- RMSE em gramas: penaliza erros grandes.
- MAPE: erro percentual, com cuidado para pesos pequenos.
- \(R^2\): capacidade explicativa.
- Bias: tendência de superestimar ou subestimar.
- Métricas por espécie e faixa de peso.
- Latência p50 e p95 de inferência.

Também adicionaria intervalos de predição calibrados com dados de validação. Uma estimativa de 800 g não deveria ser apresentada como se fosse uma pesagem exata.

### O que fazer com os 245 peixes?

Eu trataria sua coleta original como um dataset de referência, sem misturar indiscriminadamente com os datasets públicos.

Criaria uma identificação única por peixe:

```text
fish_id
image_path
weight_grams
species
capture_session
capture_location
capture_date
camera_id
```

Se você tiver várias fotos do mesmo peixe, todas precisam permanecer no mesmo grupo de treino ou teste.

Depois, coletaria uma nova amostra independente para avaliar generalização. Esse é um dos pontos mais importantes para transformar o \(R^2 = 0{,}90\) inicial em uma evidência científica mais forte.

## 7. Roadmap de implementação

Eu dividiria o projeto em quatro entregas:

    **Fase 1 — Evolução científica**

    - Revisar os notebooks e consolidar o pipeline.
    - Corrigir problemas de pré-processamento.
    - Criar validação por indivíduo.
    - Implementar CNN de regressão direta.
    - Comparar os cinco experimentos.
    - Gerar relatório comparativo.

    **Fase 2 — Motor de inferência**

    - Extrair os modelos dos notebooks.
    - Criar classes independentes de preprocessing, segmentation e regression.
    - Implementar carregamento único dos modelos.
    - Criar serviço FastAPI.
    - Exportar e comparar PyTorch com ONNX Runtime.
    - Criar testes de consistência entre treinamento e inferência.

    **Fase 3 — Aplicação realtime**

    - Interface responsiva.
    - Captura pela câmera.
    - Upload de imagens.
    - Visualização da máscara.
    - Estimativa e faixa de incerteza.
    - Histórico e exportação dos resultados.

    **Fase 4 — Evolução do produto**

    - Cadastro de espécies.
    - Registro de pesagens reais.
    - Monitoramento de erro por espécie.
    - Versionamento de modelos.
    - Dataset de feedback.
    - Re-treinamento controlado e avaliação antes de publicar uma nova versão.

## 8. Como eu organizaria o repositório

Eu evoluiria o projeto sem perder a parte científica. Os notebooks continuam sendo o ambiente de pesquisa, enquanto a aplicação de produção fica separada.

```text
estimate_body_fish_weight/
│
├── research/
│   ├── notebooks/
│   ├── experiments/
│   └── reports/
│
├── ml/
│   ├── datasets/
│   ├── preprocessing/
│   ├── segmentation/
│   ├── regression/
│   ├── evaluation/
│   └── training/
│
├── artifacts/
│   ├── segmentation/
│   ├── regression/
│   └── model_registry/
│
├── inference/
│   ├── api/
│   ├── pipelines/
│   ├── runtime/
│   └── schemas/
│
├── apps/
│   └── web/
│
├── infrastructure/
│   ├── docker/
│   └── deployment/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── model/
│
└── docs/
├── architecture/
├── experiments/
└── model-cards/
```

O ponto central é que o modelo treinado precisa ser um artefato reproduzível: pesos, configuração, versão do preprocessing, métricas e identificação do dataset. Não podemos ter um pipeline no treinamento e outro discretamente diferente na API.

---

## Minha recomendação final

Eu não começaria pelo frontend.

Começaria construindo um **Fish Weight Inference Engine**, independente da interface, com um contrato simples:

```json
{
"weight_grams": 842.5,
"species": {
"name": "tilapia",
"confidence": 0.94
},
"prediction_interval": {
"lower": 760,
"upper": 925
},
"quality": {
"is_valid": true,
"segmentation_score": 0.97
},
"model": {
"name": "fishweight-hybrid",
"version": "1.0.0"
}
}
```

Valores ilustrativos.

Depois, qualquer cliente poderá consumir esse motor: web, aplicativo mobile, câmera instalada em um criatório ou até uma futura integração com sistemas de gestão aquícola.

**A visão de produto que eu perseguiria:** uma ferramenta na qual o produtor aponta a câmera para um peixe, recebe a estimativa de peso e consegue acompanhar a biomassa do criatório sem precisar pesar manualmente cada indivíduo.

O próximo passo mais produtivo seria transformar essa proposta em um plano técnico executável, começando pela auditoria dos notebooks e pela implementação dos experimentos E1–E5 no próprio repositório.

Posso conduzir essa evolução em etapas, começando por uma revisão mais profunda do código e uma proposta de implementação com tarefas, arquivos, modelos e critérios de aceite.
