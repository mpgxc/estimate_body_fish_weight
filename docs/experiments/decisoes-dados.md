# Decisões sobre os dados

Data: 2026-11-15. O autor pediu que as cinco perguntas em aberto fossem resolvidas pela opção recomendada. Cada decisão abaixo pode ser revertida se a planilha ou o caderno da coleta disser outra coisa.

Os números vêm de `results/` e de [`comparacao.md`](comparacao.md), salvo onde está escrito "conferência avulsa".

## 1. Curimatã de 1.760 g: fica no conjunto

- **Decisão:** `IMG_20220110_152533.jpg` continua no treino e no teste. Ele está registrado em `SUSPECT_WEIGHTS` (`fishweight/data.py`), e a comparação ganhou a tabela "Sensibilidade a pesos suspeitos", que mostra o erro com e sem ele.
- **Por quê:** não há prova de erro de anotação. A foto mostra um peixe alto e robusto, na mesma escala das outras. Tirar um peixe só porque o modelo erra nele seria escolher o resultado.
- **Efeito medido** (melhor modelo de máscara, P3): com ele, MAE de 38,3 g, RMSE de 68,8 g e R² 0,925; sem ele na conta, 35,4 g, 50,7 g e 0,954. Um único peixe responde por cerca de um quarto do RMSE.
- **Para reverter:** se o peso estiver errado, mova a entrada para `EXCLUDED_IMAGES` e rode de novo `fishweight.splits`, `fishweight.features` e os experimentos.

## 2. Sessão de 29/10/2022: fica como está; a ferramenta ganha uma referência de escala

- **Decisão sobre os dados:** a sessão não é corrigida nem excluída.
- **Decisão sobre a ferramenta:** o desenho passa a prever uma referência de tamanho conhecido fixa no fundo da caixa, lida em cada foto para conferir a escala. Antes ele assumia escala constante, sem conferência.
- **O que foi medido:** criei as variantes `*_escala_sessao`, que convertem as medidas da máscara para centímetros usando a escala de cada sessão estimada pela fita. São de referência, porque usam a fita.
  - No E1 (Ridge), a correção ajuda: MAE em P3 de 38,3 g para 35,9 g e, na sessão de 29/10, de 74,6 g para 43,0 g, com o viés caindo de +74,6 g para −13,8 g.
  - No E0 (só a área), a correção piora: MAE em P3 de 41,3 g para 46,0 g e, na sessão de 29/10, de 49,9 g para 78,6 g.
- **Leitura:** a evidência é mista. Nessa sessão a razão entre pixels e centímetros de fita é maior em todas as espécies (130 a 134 px/cm, contra 124 a 126 nas outras), mas isso tanto pode ser a câmera mais perto quanto a fita medida de outro jeito naquele dia. Os dados não separam as duas causas. Eu tinha escrito no relatório da Fase 1A que a causa provável era a escala; essa afirmação fica rebaixada a hipótese.
- **Por que a referência de escala mesmo assim:** custa pouco (um marcador impresso) e elimina a dúvida daqui para frente. Sem ela, um erro de 4% na distância vira cerca de 13% no peso sem nenhum aviso.

## 3. Traíras com comprimento de 8 e 7 cm: o peso vale, a medida não

- **Decisão:** `IMG_20220110_150640.jpg` e `IMG_20220110_150758.jpg` estão em `UNRELIABLE_MEASURES`. O peso delas é usado normalmente. O comprimento e a largura ficam fora da saída auxiliar do E3 e do cálculo de escala por sessão.
- **Por quê:** a máscara indica cerca de 18 e 16 cm, e as outras três traíras têm de 22 a 26 cm. O peso (120 e 95 g) é coerente com um peixe desse tamanho; o erro está na fita.
- **Garantia:** um teste automatizado confirma que mudar o comprimento anotado desses dois peixes não altera o treino.

## 4. Datas de coleta: valem as seis datas das fotos

- **Decisão:** o artigo vai descrever seis dias de coleta, em 10/01, 12/02, 29/10, 22/11, 26/11 e 29/11 de 2022, com 21, 6, 25, 48, 66 e 79 fotos.
- **Por quê:** a data gravada pela câmera em cada uma das 245 fotos (EXIF) coincide com a do nome do arquivo, sem exceção (conferência avulsa). A monografia fala em cinco dias entre 29/10 e 10/12/2022, o que não bate com os arquivos.
- **Ressalva:** se o relógio do celular estava errado em janeiro e fevereiro, as duas primeiras datas estão erradas nos metadados também. Só a memória da coleta resolve isso.

## 5. Evento alvo: SBIAgro como primeira opção, sem travar o formato

- **Decisão:** o artigo-base continua no modelo SBC, em português, com até 12 páginas e material suplementar. A primeira opção de destino é o Congresso Brasileiro de Agroinformática (SBIAgro).
- **Por quê:** o tema é agroinformática aplicada, o evento publica artigos completos em português e os anais saem na biblioteca digital da SBC.
- **O que não foi conferido:** a data, o limite de páginas e as regras da próxima edição. Isso precisa ser visto na chamada de trabalhos antes de submeter.

## O que continua pendente e só você resolve

- Fazer commit e `git push` do branch e rodar `remote/executar.ipynb` num ambiente com GPU.
