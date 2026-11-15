# Comparação de métricas

Arquivo gerado por `uv run python -m experiments.report`. Não edite à mão.

Uma linha por técnica, ordenada pelo MAE em P3 (uma sessão de coleta fora do
treino), que é o protocolo que decide. `(ref.)` marca variantes que usam
informação que a ferramenta não terá; elas não concorrem. A descrição de cada
técnica está em [`tecnicas.md`](tecnicas.md).

| Técnica | P1: MAE (g) | P2: MAE (g) | P3: MAE (g) [IC 95%] | P3: MAPE (%) | P3: R² | P3: viés (g) |
|---|---|---|---|---|---|---|
| e1_nativo_RidgeRegression | 35,7 | 33,7 | 38,3 [31,7; 46,1] | 8,6 | 0,925 | -3,6 |
| e0_especie_oraculo (ref.) | 34,5 | 31,3 | 38,7 [32,4; 45,9] | 8,9 | 0,930 | -7,3 |
| e1_nativo_especie_oraculo_RidgeRegression (ref.) | 31,8 | 29,6 | 38,8 [33,0; 45,9] | 9,4 | 0,933 | -5,9 |
| e1_nativo_especie_oraculo_PCARegression (ref.) | 46,2 | 40,4 | 39,8 [33,0; 47,8] | 12,4 | 0,917 | 3,2 |
| e1_nativo_LassoRegression | 37,4 | 35,2 | 40,6 [33,9; 48,3] | 10,5 | 0,920 | -5,1 |
| e1_nativo_especie_oraculo_LassoRegression (ref.) | 34,5 | 30,6 | 40,7 [34,8; 48,0] | 10,4 | 0,928 | -7,5 |
| e1_nativo_LinearRegression | 33,6 | 34,4 | 41,2 [34,3; 49,2] | 9,4 | 0,915 | -5,0 |
| e0_global | 43,8 | 39,7 | 41,3 [34,5; 49,5] | 9,5 | 0,917 | -4,1 |
| e1_nativo_especie_oraculo_LinearRegression (ref.) | 33,7 | 32,2 | 42,7 [35,8; 50,4] | 10,5 | 0,916 | -4,0 |
| e1_nativo_RandomForestRegressor | 38,5 | 35,4 | 43,4 [36,2; 51,3] | 12,0 | 0,912 | -3,4 |
| e1_nativo_especie_oraculo_RandomForestRegressor (ref.) | 37,9 | 34,6 | 43,8 [36,8; 52,0] | 12,1 | 0,910 | -5,0 |
| e1_nativo_PCARegression | 48,0 | 43,6 | 44,7 [37,4; 52,9] | 15,0 | 0,906 | 2,8 |
| e1_nativo_GradientBoostingRegressor | 35,9 | 36,1 | 46,8 [39,2; 55,6] | 13,0 | 0,892 | -4,4 |
| e1_nativo_especie_oraculo_GradientBoostingRegressor (ref.) | 36,0 | 35,3 | 47,3 [39,7; 56,1] | 12,9 | 0,892 | -4,5 |
| e1_nativo_especie_oraculo_PolyRegression (ref.) | 42,9 | 42,3 | 65,8 [50,9; 83,5] | 27,9 | 0,637 | -29,1 |
| e1_nativo_PolyRegression | 38,7 | 50,5 | 72,8 [56,8; 91,5] | 26,1 | 0,602 | -27,2 |
| e1_monografia_GradientBoostingRegressor (ref.) | 34,9 | – | – | – | – | – |
| e1_monografia_LassoRegression (ref.) | 39,7 | – | – | – | – | – |
| e1_monografia_LinearRegression (ref.) | 45,1 | – | – | – | – | – |
| e1_monografia_LinearRegression_normalizada (ref.) | 41,9 | – | – | – | – | – |
| e1_monografia_PCARegression (ref.) | 69,6 | – | – | – | – | – |
| e1_monografia_PolyRegression (ref.) | 31,9 | – | – | – | – | – |
| e1_monografia_RandomForestRegressor (ref.) | 35,8 | – | – | – | – | – |
| e1_monografia_RidgeRegression (ref.) | 41,7 | – | – | – | – | – |
