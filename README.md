# Resumo

A aquisição regular do peso dos peixes é uma necessidade comum para gestores
e donos de criatórios para otimizar a alimentação diária, controlar as densidades de
estocagem e determinar o momento ideal para a pesca. Contudo, é difícil estimar o
peso dos peixes, uma vez que esses animais se movem livremente em um ambiente
onde a visibilidade e iluminação são incontroláveis. Até os dias atuais, a estimativa
da biomassa de peixes tem sido baseada em amostragem manual, o que é geralmente
demorado e trabalhoso. Há várias abordagens na literatura para predizer o peso de
animais a partir do uso de técnicas de processamento de imagens, visão computacional e redes neurais. Neste estudo, foi desenvolvido um método utilizando visão
computacional e redes neurais para prever o peso a partir de imagens de peixes.
Os animais utilizados neste estudo são provenientes de uma propriedade privada
localizada no interior do estado do Piauí. Foram adquiridas imagens e pesos reais
de 245 peixes para medir a precisão da predição em relação ao peso real. Após a
aquisição das imagens, foi aplicado o processo de segmentação de imagens, utilizando uma rede neural convolucional com arquitetura U-net, com o intuito de isolar
o corpo do animal, ignorando áreas desnecessárias. Em seguida, as características
das imagens foram extraídas, criando-se um vetor de características das imagens
segmentadas. As características extraídas são características relacionadas à forma,
como por exemplo, área, perímetro, comprimento do eixo maior e comprimento do
eixo menor. Foram realizados experimentos com algoritmos de regressão para predizer o peso dos animais, utilizando a pontuação 𝑅2 para aferir o desempenho das
predições. A metodologia aplicada neste trabalho se mostrou viável nos resultados,
onde os testes alcançaram uma pontuação 𝑅2 de 0.90.

**Palavras-chaves:** visão computacional, segmentação de imagens, processamento
de imagens, predição de peso
