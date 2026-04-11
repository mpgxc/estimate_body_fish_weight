# Plano de Estruturação do Projeto

## Estrutura Atual

```
estimate_body_fish_weight/
├── commons.py                          # Utilitários compartilhados (raiz)
├── feature_extraction.py               # Extração de características das máscaras
├── segmentation.ipynb                  # Treinamento do modelo U-Net
├── features_from_unet_predictions.ipynb # Extração de features via U-Net
├── regression.ipynb                    # Experimentos de regressão
├── regression-last-execution.ipynb     # Última execução de regressão
├── regression_cleaner_data_syntetic_adjust.ipynb
├── combined_features.csv
├── features_from_unet_predictions.csv
├── unet_model.h5                       # Modelo U-Net treinado (Keras)
├── data/
│   ├── coletas/                        # Imagens e anotações coletadas
│   │   ├── combined/
│   │   └── combined_no_29_11_22/
│   ├── convert_polygon2mask.ipynb      # Conversão de polígonos para máscaras
│   ├── sorted_frames.ipynb             # Ordenação de frames por data EXIF
│   └── unsupervised_seg_encoder-decoder.ipynb
├── handcraft_segmentation/
│   ├── commons.py
│   ├── kmeans.ipynb
│   └── segmentation.ipynb
├── yolo_v8_instance_seg/
│   ├── commons.py
│   ├── main.py
│   ├── main.ipynb
│   ├── binary_masks.ipynb
│   └── yolo_segmentation.py
└── figures/                            # Figuras geradas (PDF)
```

---

## Estrutura Proposta

A estrutura abaixo reorganiza o projeto em módulos coesos, mantendo
compatibilidade com os notebooks existentes via imports relativos.

```
estimate_body_fish_weight/
│
├── src/                                  # Pacote principal (código-fonte)
│   ├── __init__.py
│   ├── commons.py                        # Utilitários de imagem compartilhados
│   ├── feature_extraction.py             # Extração de features das máscaras
│   │
│   ├── segmentation/                     # Abordagens de segmentação
│   │   ├── __init__.py
│   │   ├── unet.py                       # Definição e treinamento da U-Net
│   │   ├── handcraft.py                  # Segmentação clássica (K-means, threshold)
│   │   └── yolo.py                       # Segmentação via YOLOv8
│   │
│   └── regression/                       # Pipeline de regressão
│       ├── __init__.py
│       ├── models.py                     # Definição e avaliação dos modelos
│       └── dataset.py                    # Carregamento e pré-processamento de dados
│
├── notebooks/                            # Notebooks organizados por etapa
│   ├── 01_data_preparation/
│   │   ├── convert_polygon2mask.ipynb
│   │   └── sorted_frames.ipynb
│   ├── 02_segmentation/
│   │   ├── unet_training.ipynb           # (ex-segmentation.ipynb)
│   │   ├── handcraft_segmentation.ipynb
│   │   ├── kmeans_segmentation.ipynb
│   │   ├── yolo_segmentation.ipynb
│   │   └── unsupervised_autoencoder.ipynb
│   ├── 03_feature_extraction/
│   │   └── features_from_unet.ipynb
│   └── 04_regression/
│       ├── regression_experiments.ipynb
│       └── regression_cleaner_data.ipynb
│
├── data/                                 # Dados (não versionados no Git)
│   └── coletas/
│       ├── combined/
│       │   ├── images/
│       │   ├── masks/
│       │   └── annotations.csv
│       └── combined_no_29_11_22/
│
├── models/                               # Modelos treinados salvos
│   └── unet_model.h5
│
├── figures/                              # Figuras e visualizações geradas
│
├── requirements.txt                      # Dependências do projeto
├── README.md                             # Descrição do projeto
└── ESTRUTURA.md                          # Este arquivo
```

---

## Pipeline do Projeto

```
Coleta de Imagens
      │
      ▼
01. Preparação dos Dados
      ├── Ordenar frames por data EXIF (sorted_frames.ipynb)
      └── Converter anotações polígono → máscara binária (convert_polygon2mask.ipynb)
      │
      ▼
02. Segmentação (isolar o corpo do peixe)
      ├── Abordagem Clássica: K-means + threshold adaptativo (handcraft_segmentation/)
      ├── Deep Learning: U-Net MobileNetV2 (segmentation.ipynb → src/segmentation/unet.py)
      └── YOLOv8 Instance Segmentation (yolo_v8_instance_seg/ → src/segmentation/yolo.py)
      │
      ▼
03. Extração de Características (features_from_unet_predictions.ipynb)
      └── regionprops: área, perímetro, eixos, excentricidade, solidez, etc.
      │
      ▼
04. Regressão de Peso (regression.ipynb)
      ├── GradientBoostingRegressor
      ├── RandomForestRegressor
      ├── Regressão Polinomial (PolynomialFeatures + Lasso)
      └── PCA + LinearRegression
      │
      ▼
Peso Estimado (R² ≈ 0.90)
```

---

## Dependências Principais

| Biblioteca        | Uso                                           |
|-------------------|-----------------------------------------------|
| TensorFlow/Keras  | Treinamento e inferência da U-Net             |
| Ultralytics       | Segmentação YOLOv8                            |
| scikit-image      | Processamento de imagens, regionprops         |
| scikit-learn      | Modelos de regressão, métricas, pipelines     |
| OpenCV (cv2)      | Operações morfológicas, visualização          |
| pandas / NumPy    | Manipulação de dados e arrays                 |
| Joblib            | Paralelização do carregamento de imagens      |
| yellowbrick       | Visualização de modelos de regressão          |
| XGBoost           | Modelo de regressão baseado em gradient boost |

---

## Melhorias Implementadas nos Arquivos Python

### `commons.py` (raiz)
- Adicionadas type hints em todas as funções
- Adicionadas docstrings nas funções sem documentação
- Removed import redundante de `tqdm` (não usado diretamente)

### `feature_extraction.py`
- Substituído `DataFrame.append()` (depreciado/removido no pandas ≥ 2.0) por `pd.concat()`
- Corrigida guarda de `None` em `images_arr`
- Adicionadas type hints

### `handcraft_segmentation/commons.py`
- Removidos imports duplicados (numpy, matplotlib, skimage, statistics importados duas vezes)
- Adicionadas type hints e docstrings

### `yolo_v8_instance_seg/yolo_segmentation.py`
- Substituído `except:` sem tipo por `except Exception:`

### `yolo_v8_instance_seg/main.py`
- Adicionado guarda `if __name__ == "__main__":` para execução como script
