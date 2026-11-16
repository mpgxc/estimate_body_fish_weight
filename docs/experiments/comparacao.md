# Comparação de métricas

Arquivo gerado por `uv run python -m experiments.report`. Não edite à mão.

Uma linha por técnica, ordenada pelo MAE em P3 (uma sessão de coleta fora do
treino), que é o protocolo que decide. A coluna **Máscara** diz de onde vêm as
medidas do teste: `manual` (anotada à mão), `predita` (gerada pela U-Net) ou
`nenhuma` (o modelo usa só a foto). A ferramenta não terá a máscara manual,
então só as duas últimas concorrem. `(ref.)` marca variantes que usam informação que a
ferramenta não terá; elas não concorrem. A descrição de cada técnica está em
[`tecnicas.md`](tecnicas.md).

| Técnica | Máscara | P1: MAE (g) | P2: MAE (g) | P3: MAE (g) [IC 95%] | P3: MAPE (%) | P3: R² | P3: viés (g) |
|---|---|---|---|---|---|---|---|
| e1_nativo_RidgeRegression_escala_sessao (ref.) | manual | 35,2 | 33,2 | 35,9 [30,3; 42,9] | 8,8 | 0,939 | -2,8 |
| e1_nativo_LassoRegression_escala_sessao (ref.) | manual | 37,5 | 34,2 | 37,0 [31,4; 44,0] | 10,0 | 0,936 | -4,9 |
| e1_nativo_LinearRegression_escala_sessao (ref.) | manual | 33,6 | 34,0 | 37,9 [31,8; 44,9] | 9,3 | 0,932 | -3,6 |
| e1_nativo_RidgeRegression | manual | 35,7 | 33,7 | 38,3 [31,7; 46,1] | 8,6 | 0,925 | -3,6 |
| e0_especie_oraculo (ref.) | manual | 34,5 | 31,3 | 38,7 [32,4; 45,9] | 8,9 | 0,930 | -7,3 |
| e1_nativo_especie_oraculo_RidgeRegression (ref.) | manual | 31,8 | 29,6 | 38,8 [33,0; 45,9] | 9,4 | 0,933 | -5,9 |
| e1_nativo_especie_oraculo_PCARegression (ref.) | manual | 46,2 | 40,4 | 39,8 [33,0; 47,8] | 12,4 | 0,917 | 3,2 |
| e1_nativo_LassoRegression | manual | 37,4 | 35,2 | 40,6 [33,9; 48,3] | 10,5 | 0,920 | -5,1 |
| e1_nativo_especie_oraculo_LassoRegression (ref.) | manual | 34,5 | 30,6 | 40,7 [34,8; 48,0] | 10,4 | 0,928 | -7,5 |
| e1_nativo_LinearRegression | manual | 33,6 | 34,4 | 41,2 [34,3; 49,2] | 9,4 | 0,915 | -5,0 |
| e0_global | manual | 43,8 | 39,7 | 41,3 [34,5; 49,5] | 9,5 | 0,917 | -4,1 |
| e1_nativo_especie_oraculo_LinearRegression (ref.) | manual | 33,7 | 32,2 | 42,7 [35,8; 50,4] | 10,5 | 0,916 | -4,0 |
| e1_nativo_RandomForestRegressor_escala_sessao (ref.) | manual | 37,5 | 34,4 | 42,9 [36,0; 51,1] | 12,0 | 0,916 | -2,5 |
| e1_nativo_RandomForestRegressor | manual | 38,5 | 35,4 | 43,4 [36,2; 51,3] | 12,0 | 0,912 | -3,4 |
| e1_nativo_especie_oraculo_RandomForestRegressor (ref.) | manual | 37,9 | 34,6 | 43,8 [36,8; 52,0] | 12,1 | 0,910 | -5,0 |
| e1_nativo_PCARegression | manual | 48,0 | 43,6 | 44,7 [37,4; 52,9] | 15,0 | 0,906 | 2,8 |
| e1_nativo_GradientBoostingRegressor_escala_sessao (ref.) | manual | 34,9 | 37,0 | 45,5 [37,9; 54,5] | 12,8 | 0,898 | -1,7 |
| e0_global_escala_sessao (ref.) | manual | 45,7 | 41,9 | 46,0 [38,6; 54,3] | 10,7 | 0,906 | -3,8 |
| e1_nativo_PCARegression_escala_sessao (ref.) | manual | 49,9 | 44,7 | 46,5 [39,3; 54,4] | 16,6 | 0,907 | 3,0 |
| e1_nativo_GradientBoostingRegressor | manual | 35,9 | 36,1 | 46,8 [39,2; 55,6] | 13,0 | 0,892 | -4,4 |
| e1_nativo_especie_oraculo_GradientBoostingRegressor (ref.) | manual | 36,0 | 35,3 | 47,3 [39,7; 56,1] | 12,9 | 0,892 | -4,5 |
| e1_nativo_PolyRegression_escala_sessao (ref.) | manual | 49,6 | 41,1 | 63,3 [47,4; 82,3] | 19,4 | 0,638 | -22,5 |
| e1_nativo_especie_oraculo_PolyRegression (ref.) | manual | 42,9 | 42,3 | 65,8 [50,9; 83,5] | 27,9 | 0,637 | -29,1 |
| e1_nativo_PolyRegression | manual | 38,7 | 50,5 | 72,8 [56,8; 91,5] | 26,1 | 0,602 | -27,2 |
| e0_especie_oraculo_mascara_predita (ref.) | predita | 34,0 | – | – | – | – | – |
| e0_global_mascara_predita | predita | 42,2 | – | – | – | – | – |
| e1_monografia_GradientBoostingRegressor (ref.) | manual | 34,9 | – | – | – | – | – |
| e1_monografia_LassoRegression (ref.) | manual | 39,7 | – | – | – | – | – |
| e1_monografia_LinearRegression (ref.) | manual | 45,1 | – | – | – | – | – |
| e1_monografia_LinearRegression_normalizada (ref.) | manual | 41,9 | – | – | – | – | – |
| e1_monografia_PCARegression (ref.) | manual | 69,6 | – | – | – | – | – |
| e1_monografia_PolyRegression (ref.) | manual | 31,9 | – | – | – | – | – |
| e1_monografia_RandomForestRegressor (ref.) | manual | 35,8 | – | – | – | – | – |
| e1_monografia_RidgeRegression (ref.) | manual | 41,7 | – | – | – | – | – |
| e1_nativo_GradientBoostingRegressor_mascara_predita | predita | 37,8 | – | – | – | – | – |
| e1_nativo_LassoRegression_mascara_predita | predita | 38,3 | – | – | – | – | – |
| e1_nativo_LinearRegression_mascara_predita | predita | 39,1 | – | – | – | – | – |
| e1_nativo_PCARegression_mascara_predita | predita | 48,4 | – | – | – | – | – |
| e1_nativo_PolyRegression_mascara_predita | predita | 44,3 | – | – | – | – | – |
| e1_nativo_RandomForestRegressor_mascara_predita | predita | 39,2 | – | – | – | – | – |
| e1_nativo_RidgeRegression_mascara_predita | predita | 36,4 | – | – | – | – | – |
| e1_nativo_especie_oraculo_GradientBoostingRegressor_mascara_predita (ref.) | predita | 37,6 | – | – | – | – | – |
| e1_nativo_especie_oraculo_LassoRegression_mascara_predita (ref.) | predita | 35,7 | – | – | – | – | – |
| e1_nativo_especie_oraculo_LinearRegression_mascara_predita (ref.) | predita | 37,3 | – | – | – | – | – |
| e1_nativo_especie_oraculo_PCARegression_mascara_predita (ref.) | predita | 47,2 | – | – | – | – | – |
| e1_nativo_especie_oraculo_PolyRegression_mascara_predita (ref.) | predita | 48,5 | – | – | – | – | – |
| e1_nativo_especie_oraculo_RandomForestRegressor_mascara_predita (ref.) | predita | 37,5 | – | – | – | – | – |
| e1_nativo_especie_oraculo_RidgeRegression_mascara_predita (ref.) | predita | 34,5 | – | – | – | – | – |

## Segmentação

Máscara prevista contra a máscara manual, sempre por uma U-Net que não viu a foto.

| Protocolo | Máscaras | Dice médio | IoU médio | Dice mínimo |
|---|---|---|---|---|
| P1 | 74 | 0,981 | 0,963 | 0,941 |

## Domínio cruzado

O modelo treina num conjunto (a coleta ou o FDWE) e é testado no outro, sem ajuste.

| Experimento | Treino | Teste | n | MAE (g) [IC 95%] | MAPE (%) | R² | Viés (g) |
|---|---|---|---|---|---|---|---|

## Sensibilidade a pesos suspeitos

O melhor candidato em P3, com todos os peixes e sem os que têm peso anotado suspeito (`SUSPECT_WEIGHTS`). Os peixes continuam no treino; só saem da conta do erro.

| Experimento | Peixes | n | MAE (g) | RMSE (g) | MAPE (%) | R² | Viés (g) |
|---|---|---|---|---|---|---|---|
| e1_nativo_RidgeRegression | todos | 244 | 38,3 | 68,8 | 8,6 | 0,925 | -3,6 |
| e1_nativo_RidgeRegression | sem pesos suspeitos | 243 | 35,4 | 50,7 | 8,5 | 0,954 | -0,6 |

## Tabelas por protocolo

### P1 — reprodução (70/30 da monografia)

| Experimento | MAE (g) [IC 95%] | RMSE (g) | MAPE (%) | R² | Viés (g) |
|---|---|---|---|---|---|
| e1_nativo_especie_oraculo_RidgeRegression (ref.) | 31,8 [19,9; 51,0] | 81,7 | 7,8 | 0,885 | -8,1 |
| e1_monografia_PolyRegression (ref.) | 31,9 [20,1; 48,6] | 71,3 | 7,9 | 0,912 | -1,5 |
| e1_nativo_LinearRegression_escala_sessao (ref.) | 33,6 [21,3; 53,3] | 84,2 | 8,0 | 0,878 | -10,9 |
| e1_nativo_LinearRegression | 33,6 [20,6; 54,8] | 87,4 | 7,6 | 0,869 | -12,4 |
| e1_nativo_especie_oraculo_LinearRegression (ref.) | 33,7 [20,8; 53,2] | 82,8 | 9,4 | 0,882 | -3,7 |
| e0_especie_oraculo_mascara_predita (ref.) | 34,0 [22,2; 53,1] | 79,4 | 8,5 | 0,892 | -7,7 |
| e0_especie_oraculo (ref.) | 34,5 [23,2; 53,6] | 79,8 | 8,6 | 0,890 | -13,2 |
| e1_nativo_especie_oraculo_LassoRegression (ref.) | 34,5 [22,2; 54,6] | 85,0 | 8,9 | 0,876 | -13,1 |
| e1_nativo_especie_oraculo_RidgeRegression_mascara_predita (ref.) | 34,5 [22,2; 53,6] | 81,8 | 8,8 | 0,885 | -0,2 |
| e1_nativo_GradientBoostingRegressor_escala_sessao (ref.) | 34,9 [22,6; 54,4] | 81,0 | 8,0 | 0,887 | -11,1 |
| e1_monografia_GradientBoostingRegressor (ref.) | 34,9 [22,4; 55,0] | 83,4 | 8,2 | 0,880 | -12,7 |
| e1_nativo_RidgeRegression_escala_sessao (ref.) | 35,2 [22,5; 56,0] | 86,4 | 8,6 | 0,871 | -17,4 |
| e1_nativo_RidgeRegression | 35,7 [22,6; 57,8] | 90,2 | 8,3 | 0,860 | -18,8 |
| e1_nativo_especie_oraculo_LassoRegression_mascara_predita (ref.) | 35,7 [23,0; 55,9] | 84,5 | 8,7 | 0,877 | -6,9 |
| e1_monografia_RandomForestRegressor (ref.) | 35,8 [22,6; 55,7] | 84,1 | 7,7 | 0,878 | -17,9 |
| e1_nativo_GradientBoostingRegressor | 35,9 [23,1; 55,8] | 84,7 | 8,5 | 0,877 | -12,1 |
| e1_nativo_especie_oraculo_GradientBoostingRegressor (ref.) | 36,0 [23,0; 56,2] | 84,7 | 8,9 | 0,876 | -11,3 |
| e1_nativo_RidgeRegression_mascara_predita | 36,4 [23,0; 58,1] | 89,0 | 8,5 | 0,864 | -12,1 |
| e1_nativo_especie_oraculo_LinearRegression_mascara_predita (ref.) | 37,3 [24,6; 56,8] | 83,2 | 11,1 | 0,881 | 1,2 |
| e1_nativo_LassoRegression | 37,4 [24,2; 59,9] | 92,6 | 9,9 | 0,852 | -20,0 |
| e1_nativo_LassoRegression_escala_sessao (ref.) | 37,5 [24,2; 58,9] | 89,3 | 10,3 | 0,863 | -18,3 |
| e1_nativo_RandomForestRegressor_escala_sessao (ref.) | 37,5 [24,9; 57,1] | 82,2 | 10,2 | 0,884 | -8,9 |
| e1_nativo_especie_oraculo_RandomForestRegressor_mascara_predita (ref.) | 37,5 [24,8; 57,0] | 83,8 | 10,2 | 0,879 | -8,4 |
| e1_nativo_especie_oraculo_GradientBoostingRegressor_mascara_predita (ref.) | 37,6 [24,7; 56,7] | 82,1 | 9,6 | 0,884 | -8,8 |
| e1_nativo_GradientBoostingRegressor_mascara_predita | 37,8 [25,2; 56,4] | 81,4 | 9,4 | 0,886 | -11,0 |
| e1_nativo_especie_oraculo_RandomForestRegressor (ref.) | 37,9 [25,2; 57,0] | 83,1 | 10,4 | 0,881 | -12,5 |
| e1_nativo_LassoRegression_mascara_predita | 38,3 [24,6; 60,3] | 91,4 | 10,3 | 0,856 | -16,2 |
| e1_nativo_RandomForestRegressor | 38,5 [25,9; 57,4] | 82,2 | 10,5 | 0,884 | -12,9 |
| e1_nativo_PolyRegression | 38,7 [25,4; 59,4] | 89,6 | 10,1 | 0,862 | -18,4 |
| e1_nativo_LinearRegression_mascara_predita | 39,1 [26,5; 58,9] | 85,8 | 9,7 | 0,873 | -0,4 |
| e1_nativo_RandomForestRegressor_mascara_predita | 39,2 [26,4; 58,2] | 83,4 | 10,5 | 0,880 | -9,3 |
| e1_monografia_LassoRegression (ref.) | 39,7 [25,0; 61,4] | 91,9 | 13,0 | 0,855 | -17,7 |
| e1_monografia_RidgeRegression (ref.) | 41,7 [27,3; 62,7] | 90,2 | 13,4 | 0,860 | -17,1 |
| e1_monografia_LinearRegression_normalizada (ref.) | 41,9 [27,4; 62,9] | 90,4 | 13,5 | 0,859 | -17,2 |
| e0_global_mascara_predita | 42,2 [28,0; 63,9] | 93,2 | 10,4 | 0,850 | -16,9 |
| e1_nativo_especie_oraculo_PolyRegression (ref.) | 42,9 [25,6; 65,1] | 98,0 | 16,8 | 0,835 | -14,1 |
| e0_global | 43,8 [29,5; 66,4] | 95,3 | 10,3 | 0,844 | -22,9 |
| e1_nativo_PolyRegression_mascara_predita | 44,3 [28,8; 65,1] | 95,0 | 13,2 | 0,845 | -8,1 |
| e1_monografia_LinearRegression (ref.) | 45,1 [30,2; 68,4] | 98,1 | 10,7 | 0,834 | -20,7 |
| e0_global_escala_sessao (ref.) | 45,7 [31,8; 66,6] | 92,1 | 10,9 | 0,854 | -21,9 |
| e1_nativo_especie_oraculo_PCARegression (ref.) | 46,2 [29,7; 70,1] | 101,2 | 19,0 | 0,824 | -17,4 |
| e1_nativo_especie_oraculo_PCARegression_mascara_predita (ref.) | 47,2 [30,8; 70,9] | 100,7 | 19,2 | 0,826 | -15,6 |
| e1_nativo_PCARegression | 48,0 [31,2; 73,0] | 104,4 | 19,7 | 0,812 | -23,5 |
| e1_nativo_PCARegression_mascara_predita | 48,4 [31,8; 72,6] | 104,0 | 19,8 | 0,814 | -21,0 |
| e1_nativo_especie_oraculo_PolyRegression_mascara_predita (ref.) | 48,5 [29,3; 71,6] | 106,5 | 23,3 | 0,805 | -8,7 |
| e1_nativo_PolyRegression_escala_sessao (ref.) | 49,6 [30,9; 72,4] | 105,1 | 20,3 | 0,810 | -22,1 |
| e1_nativo_PCARegression_escala_sessao (ref.) | 49,9 [32,9; 73,8] | 103,9 | 21,3 | 0,814 | -22,0 |
| e1_monografia_PCARegression (ref.) | 69,6 [51,7; 93,1] | 115,9 | 22,3 | 0,769 | -11,9 |

### P2 — aleatório estratificado (5 partes, 3 repetições)

| Experimento | MAE (g) [IC 95%] | RMSE (g) | MAPE (%) | R² | Viés (g) |
|---|---|---|---|---|---|
| e1_nativo_especie_oraculo_RidgeRegression (ref.) | 29,6 [24,3; 36,5] | 57,7 | 7,0 | 0,947 | 0,5 |
| e1_nativo_especie_oraculo_LassoRegression (ref.) | 30,6 [25,3; 37,9] | 59,4 | 7,0 | 0,944 | -0,0 |
| e0_especie_oraculo (ref.) | 31,3 [25,6; 38,2] | 59,5 | 7,1 | 0,944 | -1,2 |
| e1_nativo_especie_oraculo_LinearRegression (ref.) | 32,2 [26,7; 39,1] | 60,7 | 7,9 | 0,941 | 0,3 |
| e1_nativo_RidgeRegression_escala_sessao (ref.) | 33,2 [27,7; 40,6] | 61,7 | 8,0 | 0,940 | 0,6 |
| e1_nativo_RidgeRegression | 33,7 [27,8; 41,6] | 64,5 | 7,8 | 0,934 | 0,7 |
| e1_nativo_LinearRegression_escala_sessao (ref.) | 34,0 [28,5; 41,2] | 61,5 | 8,6 | 0,940 | 0,1 |
| e1_nativo_LassoRegression_escala_sessao (ref.) | 34,2 [28,7; 41,7] | 62,5 | 8,9 | 0,938 | -0,0 |
| e1_nativo_LinearRegression | 34,4 [28,4; 42,2] | 65,1 | 8,4 | 0,933 | 0,0 |
| e1_nativo_RandomForestRegressor_escala_sessao (ref.) | 34,4 [28,0; 42,2] | 66,3 | 7,8 | 0,930 | 0,2 |
| e1_nativo_especie_oraculo_RandomForestRegressor (ref.) | 34,6 [27,9; 42,8] | 68,3 | 7,6 | 0,926 | 0,2 |
| e1_nativo_LassoRegression | 35,2 [29,2; 43,1] | 66,0 | 8,8 | 0,931 | 0,2 |
| e1_nativo_especie_oraculo_GradientBoostingRegressor (ref.) | 35,3 [28,1; 44,5] | 75,3 | 7,3 | 0,910 | -0,4 |
| e1_nativo_RandomForestRegressor | 35,4 [28,6; 43,6] | 68,6 | 7,8 | 0,925 | -0,2 |
| e1_nativo_GradientBoostingRegressor | 36,1 [29,0; 45,0] | 74,9 | 7,8 | 0,911 | -0,3 |
| e1_nativo_GradientBoostingRegressor_escala_sessao (ref.) | 37,0 [29,4; 46,2] | 76,7 | 7,9 | 0,906 | 0,9 |
| e0_global | 39,7 [32,9; 47,8] | 72,5 | 9,0 | 0,917 | -6,8 |
| e1_nativo_especie_oraculo_PCARegression (ref.) | 40,4 [33,2; 48,8] | 74,8 | 14,2 | 0,911 | 0,1 |
| e1_nativo_PolyRegression_escala_sessao (ref.) | 41,1 [32,2; 52,2] | 95,1 | 9,7 | 0,856 | -6,8 |
| e0_global_escala_sessao (ref.) | 41,9 [35,3; 50,0] | 72,0 | 9,6 | 0,918 | -7,6 |
| e1_nativo_especie_oraculo_PolyRegression (ref.) | 42,3 [32,8; 53,6] | 98,2 | 10,3 | 0,847 | -0,4 |
| e1_nativo_PCARegression | 43,6 [36,4; 52,1] | 77,1 | 15,3 | 0,905 | 0,1 |
| e1_nativo_PCARegression_escala_sessao (ref.) | 44,7 [38,0; 53,2] | 76,5 | 16,3 | 0,907 | 0,1 |
| e1_nativo_PolyRegression | 50,5 [35,0; 74,8] | 181,3 | 10,6 | 0,478 | -12,2 |

### P3 — uma sessão de coleta de fora por vez

| Experimento | MAE (g) [IC 95%] | RMSE (g) | MAPE (%) | R² | Viés (g) |
|---|---|---|---|---|---|
| e1_nativo_RidgeRegression_escala_sessao (ref.) | 35,9 [30,3; 42,9] | 61,9 | 8,8 | 0,939 | -2,8 |
| e1_nativo_LassoRegression_escala_sessao (ref.) | 37,0 [31,4; 44,0] | 63,3 | 10,0 | 0,936 | -4,9 |
| e1_nativo_LinearRegression_escala_sessao (ref.) | 37,9 [31,8; 44,9] | 65,4 | 9,3 | 0,932 | -3,6 |
| e1_nativo_RidgeRegression | 38,3 [31,7; 46,1] | 68,8 | 8,6 | 0,925 | -3,6 |
| e0_especie_oraculo (ref.) | 38,7 [32,4; 45,9] | 66,6 | 8,9 | 0,930 | -7,3 |
| e1_nativo_especie_oraculo_RidgeRegression (ref.) | 38,8 [33,0; 45,9] | 65,0 | 9,4 | 0,933 | -5,9 |
| e1_nativo_especie_oraculo_PCARegression (ref.) | 39,8 [33,0; 47,8] | 72,3 | 12,4 | 0,917 | 3,2 |
| e1_nativo_LassoRegression | 40,6 [33,9; 48,3] | 70,9 | 10,5 | 0,920 | -5,1 |
| e1_nativo_especie_oraculo_LassoRegression (ref.) | 40,7 [34,8; 48,0] | 67,4 | 10,4 | 0,928 | -7,5 |
| e1_nativo_LinearRegression | 41,2 [34,3; 49,2] | 73,3 | 9,4 | 0,915 | -5,0 |
| e0_global | 41,3 [34,5; 49,5] | 72,1 | 9,5 | 0,917 | -4,1 |
| e1_nativo_especie_oraculo_LinearRegression (ref.) | 42,7 [35,8; 50,4] | 72,5 | 10,5 | 0,916 | -4,0 |
| e1_nativo_RandomForestRegressor_escala_sessao (ref.) | 42,9 [36,0; 51,1] | 72,9 | 12,0 | 0,916 | -2,5 |
| e1_nativo_RandomForestRegressor | 43,4 [36,2; 51,3] | 74,5 | 12,0 | 0,912 | -3,4 |
| e1_nativo_especie_oraculo_RandomForestRegressor (ref.) | 43,8 [36,8; 52,0] | 75,1 | 12,1 | 0,910 | -5,0 |
| e1_nativo_PCARegression | 44,7 [37,4; 52,9] | 77,0 | 15,0 | 0,906 | 2,8 |
| e1_nativo_GradientBoostingRegressor_escala_sessao (ref.) | 45,5 [37,9; 54,5] | 80,0 | 12,8 | 0,898 | -1,7 |
| e0_global_escala_sessao (ref.) | 46,0 [38,6; 54,3] | 76,8 | 10,7 | 0,906 | -3,8 |
| e1_nativo_PCARegression_escala_sessao (ref.) | 46,5 [39,3; 54,4] | 76,5 | 16,6 | 0,907 | 3,0 |
| e1_nativo_GradientBoostingRegressor | 46,8 [39,2; 55,6] | 82,6 | 13,0 | 0,892 | -4,4 |
| e1_nativo_especie_oraculo_GradientBoostingRegressor (ref.) | 47,3 [39,7; 56,1] | 82,5 | 12,9 | 0,892 | -4,5 |
| e1_nativo_PolyRegression_escala_sessao (ref.) | 63,3 [47,4; 82,3] | 150,9 | 19,4 | 0,638 | -22,5 |
| e1_nativo_especie_oraculo_PolyRegression (ref.) | 65,8 [50,9; 83,5] | 151,2 | 27,9 | 0,637 | -29,1 |
| e1_nativo_PolyRegression | 72,8 [56,8; 91,5] | 158,3 | 26,1 | 0,602 | -27,2 |

### Por espécie: e1_nativo_RidgeRegression

| Grupo | n | MAE (g) | RMSE (g) | MAPE (%) | R² | Viés (g) |
|---|---|---|---|---|---|---|
| curimatã | 20 | 99,6 | 180,2 | 8,7 | 0,635 | -50,9 |
| outras | 15 | 50,5 | 74,5 | 14,6 | 0,954 | -7,8 |
| tambaqui | 148 | 24,6 | 39,1 | 7,4 | 0,940 | 5,0 |
| tilápia | 61 | 48,3 | 56,5 | 10,1 | 0,506 | -7,8 |

`outras` reúne panga, traíra e lambari: amostra pequena, leitura apenas indicativa.

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
