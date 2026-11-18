# COPILOT.md — Comparar dois métodos de homografia (imagem → campo em metros)

## Objetivo

Trabalhe em uma **branch nova** (sugestão: `feat/homografia-segmentacao`) e me ajude a **entender e documentar a diferença** entre dois métodos de homografia para um frame de transmissão de futebol. No final, quero um relatório comparativo com números, não só texto.

**Não modifique nem apague o método original.** Ele deve continuar funcionando como baseline.

## Contexto

- Entrada: um frame BGR (`image`, 1920×1080) e a saída de um modelo de segmentação (`predictions`, lista de dicts com `class`, `confidence`, `points` = polígono em pixels). Classes usadas: `Second Half Field`, `Second Half Central Circle`, `18Yard`, `18Yard Circle`.
- Saída desejada: uma matriz `H_img2m` que leva pixels da imagem para coordenadas do campo em **metros** (campo 105 × 68 m, origem no canto superior esquerdo, x = comprimento, y = largura).
- Uso final: converter o ponto dos pés de cada jogador (centro da base da bbox) em metros com `cv2.perspectiveTransform`. A homografia só é válida para o plano do gramado, então nunca aplicar em pontos acima do chão.
- A câmera está parada neste frame. Se ela se mover entre frames, a homografia precisa ser recalculada por frame.

## Método A — baseline (já existe no repositório)

Mapeia os **4 cantos do polígono do campo detectado** na imagem para os 4 cantos de um retângulo 105 × 68 m (`cv2.getPerspectiveTransform`).

**Por que está errado geometricamente:** os lados esquerdo e direito do polígono são as **bordas do enquadramento da câmera**, não linhas do campo. O frame mostra só ~60–80 m do comprimento, mas o método força isso em 105 m. Efeitos observados:
- a linha do meio aparece em x ≈ 70 m em vez de 52,5 m;
- o círculo central sai elíptico (≈ 15 m × 8 m de raio em vez de 9,15 m nos dois eixos);
- o eixo x fica esticado ~1,7×.

## Método B — novo (arquivo `homografia_campo.py`, anexado a este prompt)

Usa **restrições geométricas reais do campo** extraídas das segmentações, e resolve a homografia por mínimos quadrados robustos (`scipy.optimize.least_squares`, `loss='soft_l1'`):

| Elemento | Origem | Restrição no campo |
|---|---|---|
| Linha do meio | borda esquerda do polígono direito de `Second Half Field` | x = 52,5 |
| Lateral de cima | topo da união dos polígonos de campo | y = 0 |
| Lateral de baixo | base da união dos polígonos de campo | y = 68 |
| Círculo central | arco dos dois `Second Half Central Circle` (pontos de corte sobre a linha do meio são descartados) | centro (52,5; 34), raio 9,15 |
| Área esquerda | pixels de linha branca (top-hat) dentro das máscaras `18Yard`/`18Yard Circle` | distância ao modelo da grande área: linha em x = 16,5, laterais y = 13,84 e 54,16, arco da meia-lua (marca do pênalti a 11 m, raio 9,15) |

Fluxo:
1. Inicialização analítica (interseções da linha do meio com as laterais + extremos do círculo em y = 34, usando mapa projetivo 1D e ponto de fuga das laterais).
2. **Etapa A:** refina com linha do meio + laterais + círculo.
3. **Etapa B:** adiciona a área esquerda, com peso `W_LEFT = 0.5` e resíduo truncado em `CLIP = 3.0` m para tolerar associação errada de pixels. Parte da solução da etapa A.
4. Janela visível em metros (`X_MIN`/`X_MAX`) calculada projetando o contorno do campo; warp com máscara dilatada para fechar a fenda na linha do meio.

**Por que a etapa B existe:** só com a etapa A, o lado esquerdo (longe dos pontos usados) tinha erro de ~2 m por extrapolação de perspectiva.

### Resultados já medidos (frame de exemplo)

```
A (só centro)  | meio 0.08 | lat.cima 0.55 | lat.baixo 0.08 | círculo 0.20 | área esq. mediana 1.75, RMS 1.64
B (com área)   | meio 0.08 | lat.cima 0.56 | lat.baixo 0.08 | círculo 0.30 | área esq. mediana 0.08, RMS 0.17
trecho visível: x = 1.7 a 80.1 m
```

Todos os valores em metros.

## Tarefas

1. **Criar a branch** e não alterar o método A.
2. **Refatorar** o método B em funções puras e testáveis em um módulo (ex.: `homography/segmentation_homography.py`), sem variáveis globais implícitas. Sugestão de API:
   - `estimate_homography(image, predictions, flip=False) -> HomographyResult` com `H_img2m`, `diagnostics` (dict com os RMS) e `visible_x_range`.
   - `img_to_field(H, points_xy) -> np.ndarray`.
   - `warp_to_field(image, H, mask, scale) -> np.ndarray`.
   Mantenha a mesma matemática do arquivo anexo. Não "melhore" o algoritmo sem avisar.
3. **Encapsular o método A** em uma função com a mesma interface (`estimate_homography_corners(...)`), copiando o comportamento existente, para poder comparar lado a lado.
4. **Script de comparação** (`scripts/compare_homographies.py`) que, para o mesmo frame, calcula para A e B:
   - posição projetada da linha do meio (esperado x = 52,5);
   - raio do círculo central projetado nos eixos x e y (esperado 9,15 m nos dois; reportar a razão x/y);
   - posição projetada da linha da grande área esquerda (esperado x = 16,5) e do arco (até x ≈ 20,15);
   - distância entre os pés de 3–5 jogadores convertidos por A e por B (mostra o impacto prático);
   - imagens retificadas lado a lado, com a linha do meio e o círculo de referência desenhados em ciano.
5. **Testes** (`pytest`) com dados sintéticos: gere uma homografia conhecida, projete os elementos do campo (linha do meio, círculo, área) para a imagem, e verifique que o método B recupera `H` com erro baixo. Inclua um teste para `flip=True` e um para falha controlada quando faltam detecções (sem `Second Half Field` duplo, sem círculo).
6. **Documentar** em `docs/homografia.md`: diferenças conceituais A vs B, tabela de métricas, limitações, e como rodar.

## Limitações que devem constar na documentação

- O erro da área esquerda foi medido nos **mesmos pixels usados no ajuste**, então não é validação independente. Uma validação real exige elementos de campo não usados no ajuste (ex.: outro frame, ou referências à direita).
- **Lado direito sem referência:** acima de x ≈ 62 m o resultado é extrapolado. Não reivindicar precisão ali sem medir.
- A lateral de cima é o pior termo (≈ 0,55 m), pois a reta é ajustada no topo da máscara e pode capturar publicidade ou oclusão.
- O modelo da área assume o gol esquerdo em x = 0. Para a imagem espelhada, usar `flip=True` (espelhamento em torno de x = 52,5) e conferir o modelo da área.
- A classe `18Yard Circle` tem baixa confiança (≈ 0,43); o ajuste não depende de qual classe é linha ou arco, pois usa pixels de linha branca associados ao elemento mais próximo do modelo.
- Limiares do top-hat (`tophat > 30`, `gray > 140`) foram ajustados para este frame e podem precisar de calibração em outros.

## Critérios de aceitação

- A branch roda do zero com `pytest` passando.
- O script de comparação reproduz (dentro de pequena tolerância) os números da seção "Resultados já medidos" para o frame de exemplo.
- O método A continua funcionando e inalterado em comportamento.
- `docs/homografia.md` explica, em linguagem simples, **por que** A produz coordenadas erradas em metros e **como** B corrige.

## Como proceder

Antes de escrever código, leia o arquivo anexo `homografia_campo.py` e me devolva um resumo curto do que entendeu, com as dúvidas que tiver. Depois siga as tarefas em ordem, com um commit por tarefa e mensagens claras. Se algo no arquivo anexo parecer um bug, **aponte antes de corrigir**.