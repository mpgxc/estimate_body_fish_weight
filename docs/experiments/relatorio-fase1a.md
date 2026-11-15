# Relatório da Fase 1A

Pisos E0 (alométrico) e E1 (`regionprops`) com máscara manual, 244 peixes, nos protocolos P1, P2 e P3. As tabelas abaixo são geradas por `uv run python -m experiments.report`; o texto desta seção foi escrito à mão a partir delas.

## Leitura

**O resultado da monografia se sustenta quando um dia de coleta inteiro fica de fora.** O melhor modelo que usa só a máscara (`e1_nativo_RidgeRegression`) tem MAE de 33,7 g [30,2; 37,9] em P2 e 38,3 g [31,7; 46,1] em P3: 4,6 g a mais (14%), com intervalos que se sobrepõem. O MAPE vai de 7,8% para 8,6% e o R² de 0,934 para 0,925.

**1. A reprodução confere.** `e1_monografia` repete a Tabela 6 em Ridge (0,8599 / 41,68), Lasso (0,8546 / 39,66), PCA (0,7688 / 69,55) e Polinomial (0,9125 / 31,92). Gradient Boosting e Random Forest diferem na terceira casa porque o original não fixou semente. A regressão linear sem normalização dá 0,834 em vez de 0,859: a matriz tem posto 19 para 26 colunas (a área aparece quatro vezes, como `area`, `filled_area`, `moments-0-0` e `moments_central-0-0`) e a solução depende do solver; com normalização ela dá 0,8594 / 41,86, como na monografia. Foram selecionadas 26 colunas, entre elas o comprimento medido com fita (`length`) e a espécie anotada; 18 são momentos brutos ou centrais.

**2. Sem a espécie e sem a fita, pouco se perde.** Em P1, o melhor modelo só de máscara (`e1_nativo_LinearRegression`) chega a MAE de 33,6 g [20,6; 54,8] e R² 0,869, contra 31,9 g [20,1; 48,6] e R² 0,912 da regressão polinomial da monografia. A diferença de MAE (1,7 g) é muito menor que a largura dos intervalos. P1 tem só 74 peixes, e um deles pesa na conta: o curimatã `IMG_20220110_152533.jpg`, anotado com 1.760 g e previsto em 1.046 g.

**3. A área sozinha explica quase tudo.** Em P3, o piso alométrico `e0_global` (um atributo, dois coeficientes) tem MAE de 41,3 g [34,5; 49,5], contra 38,3 g [31,7; 46,1] do melhor E1. Os intervalos se sobrepõem quase por inteiro; os 11 atributos extras não mostram ganho claro. O teste pareado fica para o Plano 1F.

**4. A regressão polinomial de grau 3, a melhor da monografia, é a pior fora do sorteio original.** Com os atributos novos ela cai para R² 0,478 em P2 e 0,602 em P3. O resultado de 0,912 depende daquela divisão específica.

**5. O erro não é uniforme entre espécies.** Em P3, tambaqui tem MAE de 24,6 g (MAPE 7,4%); tilápia, 48,3 g (10,1%); curimatã, 99,6 g (8,7%), com viés de −50,9 g. Dentro de tilápia o R² é 0,506: o modelo ordena mal peixes de tamanho parecido.

**6. A pior sessão é 2022-10-29, e a causa provável é a escala.** O MAE é 74,6 g e o viés também é +74,6 g: todas as 25 predições ficam acima do peso real. Nessa sessão a mediana é 130,7 px/cm, contra 125,4 no conjunto (4,2% a mais), o que infla a área em 8,6%. A câmera parece ter ficado mais perto do peixe naquele dia. As duas imagens sinalizadas não explicam isso: estão em outra sessão.

**7. A sessão de 2022-01-10 pode ficar.** A mediana do grupo é 125,1 px/cm, igual à global. A variação alta vinha de duas traíras cujo comprimento anotado (8 e 7 cm) não bate com a máscara, que indica cerca de 18 e 16 cm; as outras três traíras, de peso próximo, têm 22 a 26 cm. O erro está na anotação de `length`, não na foto, e não afeta os modelos que preveem peso. Ele importa no Plano 1C, que usa o comprimento como saída auxiliar. O MAE alto do grupo (73,9 g, mediana do erro absoluto de 39,3 g) vem do curimatã de 1.760 g citado acima.

## Decisões para o autor

- **Curimatã `IMG_20220110_152533.jpg`, 1.760 g, 33 cm.** Os outros curimatãs de 34 a 37 cm pesam entre 1.135 e 1.385 g. Conferir o peso anotado; se for erro, corrigir ou excluir.
- **Traíras `IMG_20220110_150640.jpg` e `IMG_20220110_150758.jpg`.** Conferir os comprimentos de 8 e 7 cm.
- **Sessão 2022-10-29.** Houve mudança de caixa, de altura ou de celular nesse dia? Se a distância variou, a ferramenta precisa de uma referência de escala, ao contrário do que o desenho assume hoje.

## Tabelas

<!-- tabelas:início -->

### P1 — reprodução (70/30 da monografia)

| Experimento | MAE (g) [IC 95%] | RMSE (g) | MAPE (%) | R² | Viés (g) |
|---|---|---|---|---|---|
| e1_nativo_especie_oraculo_RidgeRegression (ref.) | 31,8 [19,9; 51,0] | 81,7 | 7,8 | 0,885 | -8,1 |
| e1_monografia_PolyRegression (ref.) | 31,9 [20,1; 48,6] | 71,3 | 7,9 | 0,912 | -1,5 |
| e1_nativo_LinearRegression | 33,6 [20,6; 54,8] | 87,4 | 7,6 | 0,869 | -12,4 |
| e1_nativo_especie_oraculo_LinearRegression (ref.) | 33,7 [20,8; 53,2] | 82,8 | 9,4 | 0,882 | -3,7 |
| e0_especie_oraculo (ref.) | 34,5 [23,2; 53,6] | 79,8 | 8,6 | 0,890 | -13,2 |
| e1_nativo_especie_oraculo_LassoRegression (ref.) | 34,5 [22,2; 54,6] | 85,0 | 8,9 | 0,876 | -13,1 |
| e1_monografia_GradientBoostingRegressor (ref.) | 34,9 [22,4; 55,0] | 83,4 | 8,2 | 0,880 | -12,7 |
| e1_nativo_RidgeRegression | 35,7 [22,6; 57,8] | 90,2 | 8,3 | 0,860 | -18,8 |
| e1_monografia_RandomForestRegressor (ref.) | 35,8 [22,6; 55,7] | 84,1 | 7,7 | 0,878 | -17,9 |
| e1_nativo_GradientBoostingRegressor | 35,9 [23,1; 55,8] | 84,7 | 8,5 | 0,877 | -12,1 |
| e1_nativo_especie_oraculo_GradientBoostingRegressor (ref.) | 36,0 [23,0; 56,2] | 84,7 | 8,9 | 0,876 | -11,3 |
| e1_nativo_LassoRegression | 37,4 [24,2; 59,9] | 92,6 | 9,9 | 0,852 | -20,0 |
| e1_nativo_especie_oraculo_RandomForestRegressor (ref.) | 37,9 [25,2; 57,0] | 83,1 | 10,4 | 0,881 | -12,5 |
| e1_nativo_RandomForestRegressor | 38,5 [25,9; 57,4] | 82,2 | 10,5 | 0,884 | -12,9 |
| e1_nativo_PolyRegression | 38,7 [25,4; 59,4] | 89,6 | 10,1 | 0,862 | -18,4 |
| e1_monografia_LassoRegression (ref.) | 39,7 [25,0; 61,4] | 91,9 | 13,0 | 0,855 | -17,7 |
| e1_monografia_RidgeRegression (ref.) | 41,7 [27,3; 62,7] | 90,2 | 13,4 | 0,860 | -17,1 |
| e1_nativo_especie_oraculo_PolyRegression (ref.) | 42,9 [25,6; 65,1] | 98,0 | 16,8 | 0,835 | -14,1 |
| e0_global | 43,8 [29,5; 66,4] | 95,3 | 10,3 | 0,844 | -22,9 |
| e1_monografia_LinearRegression (ref.) | 45,1 [30,2; 68,4] | 98,1 | 10,7 | 0,834 | -20,7 |
| e1_nativo_especie_oraculo_PCARegression (ref.) | 46,2 [29,7; 70,1] | 101,2 | 19,0 | 0,824 | -17,4 |
| e1_nativo_PCARegression | 48,0 [31,2; 73,0] | 104,4 | 19,7 | 0,812 | -23,5 |
| e1_monografia_PCARegression (ref.) | 69,6 [51,7; 93,1] | 115,9 | 22,3 | 0,769 | -11,9 |

### P2 — aleatório estratificado (5 partes, 3 repetições)

| Experimento | MAE (g) [IC 95%] | RMSE (g) | MAPE (%) | R² | Viés (g) |
|---|---|---|---|---|---|
| e1_nativo_especie_oraculo_RidgeRegression (ref.) | 29,6 [26,3; 33,4] | 57,7 | 7,0 | 0,947 | 0,5 |
| e1_nativo_especie_oraculo_LassoRegression (ref.) | 30,6 [27,3; 34,6] | 59,4 | 7,0 | 0,944 | -0,0 |
| e0_especie_oraculo (ref.) | 31,3 [28,0; 35,2] | 59,5 | 7,1 | 0,944 | -1,2 |
| e1_nativo_especie_oraculo_LinearRegression (ref.) | 32,2 [28,8; 36,3] | 60,7 | 7,9 | 0,941 | 0,3 |
| e1_nativo_RidgeRegression | 33,7 [30,2; 37,9] | 64,5 | 7,8 | 0,934 | 0,7 |
| e1_nativo_LinearRegression | 34,4 [30,8; 38,8] | 65,1 | 8,4 | 0,933 | 0,0 |
| e1_nativo_especie_oraculo_RandomForestRegressor (ref.) | 34,6 [30,6; 38,8] | 68,3 | 7,6 | 0,926 | 0,2 |
| e1_nativo_LassoRegression | 35,2 [31,7; 39,3] | 66,0 | 8,8 | 0,931 | 0,2 |
| e1_nativo_especie_oraculo_GradientBoostingRegressor (ref.) | 35,3 [30,8; 40,1] | 75,3 | 7,3 | 0,910 | -0,4 |
| e1_nativo_RandomForestRegressor | 35,4 [31,3; 39,6] | 68,6 | 7,8 | 0,925 | -0,2 |
| e1_nativo_GradientBoostingRegressor | 36,1 [31,7; 40,8] | 74,9 | 7,8 | 0,911 | -0,3 |
| e0_global | 39,7 [35,8; 44,3] | 72,5 | 9,0 | 0,917 | -6,8 |
| e1_nativo_especie_oraculo_PCARegression (ref.) | 40,4 [36,1; 45,2] | 74,8 | 14,2 | 0,911 | 0,1 |
| e1_nativo_especie_oraculo_PolyRegression (ref.) | 42,3 [36,2; 48,9] | 98,2 | 10,3 | 0,847 | -0,4 |
| e1_nativo_PCARegression | 43,6 [39,4; 48,5] | 77,1 | 15,3 | 0,905 | 0,1 |
| e1_nativo_PolyRegression | 50,5 [39,3; 64,4] | 181,3 | 10,6 | 0,478 | -12,2 |

### P3 — uma sessão de coleta de fora por vez

| Experimento | MAE (g) [IC 95%] | RMSE (g) | MAPE (%) | R² | Viés (g) |
|---|---|---|---|---|---|
| e1_nativo_RidgeRegression | 38,3 [31,7; 46,1] | 68,8 | 8,6 | 0,925 | -3,6 |
| e0_especie_oraculo (ref.) | 38,7 [32,4; 45,9] | 66,6 | 8,9 | 0,930 | -7,3 |
| e1_nativo_especie_oraculo_RidgeRegression (ref.) | 38,8 [33,0; 45,9] | 65,0 | 9,4 | 0,933 | -5,9 |
| e1_nativo_especie_oraculo_PCARegression (ref.) | 39,8 [33,0; 47,8] | 72,3 | 12,4 | 0,917 | 3,2 |
| e1_nativo_LassoRegression | 40,6 [33,9; 48,3] | 70,9 | 10,5 | 0,920 | -5,1 |
| e1_nativo_especie_oraculo_LassoRegression (ref.) | 40,7 [34,8; 48,0] | 67,4 | 10,4 | 0,928 | -7,5 |
| e1_nativo_LinearRegression | 41,2 [34,3; 49,2] | 73,3 | 9,4 | 0,915 | -5,0 |
| e0_global | 41,3 [34,5; 49,5] | 72,1 | 9,5 | 0,917 | -4,1 |
| e1_nativo_especie_oraculo_LinearRegression (ref.) | 42,7 [35,8; 50,4] | 72,5 | 10,5 | 0,916 | -4,0 |
| e1_nativo_RandomForestRegressor | 43,4 [36,2; 51,3] | 74,5 | 12,0 | 0,912 | -3,4 |
| e1_nativo_especie_oraculo_RandomForestRegressor (ref.) | 43,8 [36,8; 52,0] | 75,1 | 12,1 | 0,910 | -5,0 |
| e1_nativo_PCARegression | 44,7 [37,4; 52,9] | 77,0 | 15,0 | 0,906 | 2,8 |
| e1_nativo_GradientBoostingRegressor | 46,8 [39,2; 55,6] | 82,6 | 13,0 | 0,892 | -4,4 |
| e1_nativo_especie_oraculo_GradientBoostingRegressor (ref.) | 47,3 [39,7; 56,1] | 82,5 | 12,9 | 0,892 | -4,5 |
| e1_nativo_especie_oraculo_PolyRegression (ref.) | 65,8 [50,9; 83,5] | 151,2 | 27,9 | 0,637 | -29,1 |
| e1_nativo_PolyRegression | 72,8 [56,8; 91,5] | 158,3 | 26,1 | 0,602 | -27,2 |

### Por espécie: e1_nativo_RidgeRegression

| Grupo | n | MAE (g) | RMSE (g) | MAPE (%) | R² | Viés (g) |
|---|---|---|---|---|---|---|
| curimatã | 20 | 99,6 | 180,2 | 8,7 | 0,635 | -50,9 |
| outras | 15 | 50,5 | 74,5 | 14,6 | 0,954 | -7,8 |
| tambaqui | 148 | 24,6 | 39,1 | 7,4 | 0,940 | 5,0 |
| tilápia | 61 | 48,3 | 56,5 | 10,1 | 0,506 | -7,8 |

### Por sessão: e1_nativo_RidgeRegression

| Grupo | n | MAE (g) | RMSE (g) | MAPE (%) | R² | Viés (g) |
|---|---|---|---|---|---|---|
| 2022-10-29 | 25 | 74,6 | 94,7 | 11,9 | 0,895 | 74,6 |
| 2022-11-22 | 48 | 27,4 | 42,1 | 8,2 | 0,962 | 1,5 |
| 2022-11-26 | 65 | 40,4 | 47,8 | 10,2 | 0,903 | -16,0 |
| 2022-11-29 | 79 | 19,5 | 24,9 | 5,9 | 0,964 | -12,6 |
| 2022-inicio | 27 | 73,9 | 154,8 | 10,5 | 0,869 | -28,6 |

### Por faixa de peso: e1_nativo_RidgeRegression

| Grupo | n | MAE (g) | RMSE (g) | MAPE (%) | R² | Viés (g) |
|---|---|---|---|---|---|---|
| alto | 81 | 66,1 | 109,8 | 9,4 | 0,858 | -8,1 |
| baixo | 82 | 23,0 | 31,7 | 9,5 | 0,581 | -0,1 |
| medio | 81 | 26,0 | 34,2 | 7,0 | 0,112 | -2,6 |

### Imagens com escala fora do esperado (desvio acima de 15%)

| Imagem | Sessão | Espécie | Comprimento (cm) | px/cm | Desvio (%) |
|---|---|---|---|---|---|
| IMG_20220110_150640.jpg | 2022-inicio | traíra | 8,0 | 280,6 | 123,8 |
| IMG_20220110_150758.jpg | 2022-inicio | traíra | 7,0 | 286,2 | 128,2 |

<!-- tabelas:fim -->
