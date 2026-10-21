# Análise de Futebol

Pipeline modular de código aberto para extrair análises estruturadas de futebol a partir de vídeos de partidas.
O projeto contempla, atualmente, as **etapas 1 a 9**: esquemas de domínio, geometria canônica do campo, geometria de finalizações validada, contratos para detectores/rastreadores, adaptadores Ultralytics para detecção de objetos e pontos-chave do campo, ByteTrack, homografia dinâmica robusta, trajetórias em coordenadas métricas, minimapas por quadro, rastreamento dedicado da bola, posse de bola suavizada, detecção de eventos de passe e finalização, um conjunto de dados histórico de finalizações validado, um modelo base treinável de *expected goals* (xG), estatísticas de partida, exportações em formato tabular, mapas de calor, mapas de finalizações, mapas de posicionamento médio e uma interface de linha de comando (CLI) para processamento de vídeo de ponta a ponta.

## Requisitos

- Python 3.12 ou superior
- [`uv`](https://docs.astral.sh/uv/) instalado

## Configuração e verificação

```bash
git clone <repositório>
cd football-analytics
uv sync
uv run pytest
uv run ruff check .
uv run football-analytics --help
```

Formate o código com:

```bash
uv run ruff format .
```

O arquivo `uv.lock` garante a reprodutibilidade dos ambientes de execução e desenvolvimento. Adicione dependências de execução com `uv add <pacote>` e ferramentas de desenvolvimento com `uv add --dev <pacote>`.

## Arquitetura

```text
vídeo
-> Detector -> YoloDetector -> registros de detecção
-> Rastreador (Tracker) -> ByteTrackTracker -> registros de rastreamento
-> Detector de pontos-chave do campo (PitchKeypointDetector) -> observações de marcos do campo
-> Homografia via RANSAC -> posições métricas filtradas por nível de confiança
-> Trajetórias métricas -> minimapa por quadro / sobreposição em vídeo
-> Associação com a bola -> posse de bola (jogador/equipe) suavizada temporalmente
-> Detector de passes / Detector de finalizações -> registros de eventos
-> Modelo de xG treinado -> probabilidades de finalização e xG por jogador/equipe
-> Estatísticas de jogador/equipe -> arquivos Parquet / CSV / JSON e visualizações
```

O pacote utiliza uma estrutura do tipo `src`. As etapas do pipeline trocam entre si registros Pydantic imutáveis, definidos em `storage.schemas`; o módulo `pitch` define o sistema de coordenadas canônico de 105 m x 68 m. e `analytics`
contém cálculos puros, passíveis de teste de forma independente. As posições dos jogadores e da bola são registradas separadamente, garantindo que futuras exportações em formato tabular permaneçam normalizadas.

As coordenadas canônicas têm origem no canto superior esquerdo de um campo, em uma perspectiva de cima para baixo. O eixo `x` varia de 0 a 105 m ao longo da linha lateral, e o eixo `y` varia de 0 a 68 m ao longo da largura do campo. O ângulo de finalização é definido como o ângulo visual entre os vetores que partem do local do chute em direção a cada um dos postes da meta — e não o ângulo em relação ao centro do gol.

## Detecção e rastreamento

O domínio depende de contratos abstratos `Detector` e `Tracker`. As bibliotecas Ultralytics e Roboflow estão isoladas por meio de adaptadores; assim, qualquer uma das implementações pode ser substituída sem a necessidade de alterar esquemas ou processos de análise subsequentes. O método `YoloDetector.detect(frame, frame_id=...)` retorna registros `Detection` imutáveis. O método `ByteTrackTracker.update(...)` atribui identificadores (IDs) persistentes e utiliza o ponto central da base da caixa delimitadora (*bounding box*) como a referência aproximada do ponto de contato com o gramado.

Utilize pesos do modelo YOLO específicos para futebol, cujos rótulos permitam distinguir entre `player` (jogador), `goalkeeper` (goleiro), `referee` (árbitro) e `ball` (bola). Pesos genéricos, treinados no conjunto de dados COCO, não conseguem distinguir essas funções no futebol de maneira confiável. Para fins de experimentação, o parâmetro `class_name_map` pode ser usado para mapear rótulos como `person` (pessoa) e `sports ball` (bola esportiva), enquanto o parâmetro `allowed_classes` filtra detecções indesejadas. 

```python
import cv2

from football_analytics.detection import YoloDetector, YoloDetectorConfig
from football_analytics.tracking import ByteTrackTracker

detector = YoloDetector(
    "models/football.pt",
    YoloDetectorConfig(allowed_classes=frozenset({"player", "goalkeeper", "referee", "ball"})),
)
tracker = ByteTrackTracker()

frame = cv2.imread("frame.jpg")
detections = detector.detect(frame, frame_id=0)
person_detections = [item for item in detections if item.class_name != "ball"]
tracks = tracker.update(person_detections, frame_id=0, frame=frame)
```

Chame `tracker.reset()` antes de processar outro vídeo. Os frames devem ser arrays BGR de dimensões HxWx3, não vazios e com valores do tipo `uint8`. Por uma escolha deliberada, os pesos do modelo não estão incluídos no repositório nem são baixados pela suíte de testes. O ByteTrack deve receber classes que representem pessoas; observações da bola são tratadas por um rastreador dedicado, que opera em um espaço métrico. Observações internas do ByteTrack que ainda não foram confirmadas são omitidas até que o sistema atribua a elas um identificador (ID) persistente não negativo.

Após a projeção por homografia, o `MetricIdentityResolver` reconecta fragmentos curtos de rastreamento do ByteTrack, utilizando as coordenadas do campo. Ele prevê a posição de cada identidade perdida recentemente com base em sua última posição e velocidade, descarta equipes incompatíveis, filtra distâncias fisicamente implausíveis e aplica uma atribuição global do tipo "um para um". Os IDs originais do ByteTrack permanecem nos dados exportados (`tracks`) para fins de auditoria; já as posições métricas, posse de bola, passes, finalizações, minimapas e estatísticas utilizam os IDs canônicos.

Cada arquivo `report.json` inclui também a seção `identity_resolution`, que apresenta: o número de IDs originais projetados, o número de identidades canônicas, a quantidade de fragmentos de rastreamento (tracklets) mesclados e o mapeamento completo de IDs originais para canônicos (`raw_to_canonical`). Utilize essas informações para comparar diferentes configurações de rastreamento no mesmo vídeo; o relatório contempla apenas as identidades que possuem uma projeção métrica válida, e não todas as observações brutas geradas pelo ByteTrack. As dependências de tempo de execução são gerenciadas pelo `uv`: Pydantic, NumPy, Polars, scikit-learn
rn, joblib,
SciPy, OpenCV, Ultralytics, Supervision e Roboflow Trackers. O PyTorch é incluído de forma transitiva pelo
Ultralytics. pytest e Ruff permanecem como dependências de desenvolvimento.

## Pontos-chave do campo e homografia

O modelo canônico contém os 32 pontos-chave (landmarks) utilizados pelo modelo de pose de campo de futebol da Roboflow:
interseções da linha de meta, interseções da linha central, pontos extremos do círculo central,
interseções da área penal e da pequena área, e marcas do pênalti. A ordem de seus índices corresponde à de `SoccerPitchConfiguration`;
as coordenadas de destino são expressas com base no campo canônico de 105 m x 68 m deste projeto.

O `YoloPitchKeypointDetector` exige, deliberadamente, um nível de confiança para cada ponto-chave.
Ele nunca inventa uma pontuação caso o modelo forneça apenas as coordenadas. Pontos-chave com baixa confiança
são descartados antes da etapa de estimativa de homografia.

```python
from football_analytics.pitch import PitchHomography, YoloPitchKeypointDetector

keypoint_detector = YoloPitchKeypointDetector("models/football_field.onnx")
homography = PitchHomography()

keypoints = keypoint_detector.detect(frame, frame_id=frame_id)
homography.estimate_keypoints(keypoints)

if homography.validate():
pitch_x, pitch_y = homography.transform_point(player_bottom_centre)
```

O `PitchHomography.estimate()` utiliza o algoritmo RANSAC do OpenCV, exigindo pelo menos quatro correspondências
confiáveis, únicas e não colineares. O nível de confiança resultante combina a proporção de *inliers* do RANSAC,
a confiança média dos pontos-chave e o erro quadrático médio (RMSE) de reprojeção dos *inliers*, expresso em metros.
Caso o recálculo falhe, a matriz anterior é descartada; estimativas não confiáveis ​​não podem ser usadas para transformar posições.
Isso evita a permanência de coordenadas obsoletas após cortes de câmera e impede a geração de estatísticas espaciais
quando as evidências no campo são insuficientes.

## Posições métricas e minimapa

O `MetricPositionEstimator` transforma o ponto central da base de cada objeto detectado (como jogadores)
por meio da homografia confiável vigente. O componente descarta resultados que estejam fora do campo
e calcula a velocidade, em metros por segundo, com base em observações consecutivas consideradas confiáveis. As primeiras observações e as lacunas maiores do que o limiar configurado resultam em velocidades desconhecidas, em vez de amostras artificiais com velocidade zero.

Goleiros e árbitros herdam seu papel semântico a partir da detecção. A atribuição de jogadores a equipes pode ser fornecida por meio de um mapeamento de ID de rastreamento para equipe, caso um classificador de equipes esteja disponível. Jogadores não classificados permanecem válidos, mas aparecem na cor cinza no minimapa.

~~~python
from football_analytics.analytics import MetricPositionEstimator
from football_analytics.visualization import MinimapRenderer

position_estimator = MetricPositionEstimator(homography)
minimap_renderer = MinimapRenderer()

timestamp = frame_id / frames_per_second
positions = position_estimator.project_tracks(
    tracks,
    timestamp=timestamp,
    team_assignments=team_assignments,
)
ball_position = position_estimator.project_ball(ball_detection, timestamp=timestamp)
minimap = minimap_renderer.render(positions, ball_position)
annotated_frame = minimap_renderer.overlay(frame, minimap)
~~~

O renderizador desenha o campo canônico, ambas as equipes, goleiros, árbitros, jogadores sem equipe atribuída e a bola. Cada renderização representa exatamente um quadro (frame) de vídeo; observações que abrangem múltiplos quadros são descartadas. A sobreposição (overlay) é redimensionada para se ajustar ao quadro original, mantendo o array de entrada inalterado.

## Rastreamento da bola e posse de bola

O `BallTracker` associa candidatos a posições da bola projetadas, utilizando a posição prevista e a distância métrica. Ele suaviza as medições, estima a velocidade da bola, preenche breves lacunas na detecção por meio de previsões baseadas em velocidade constante, reduz a confiança na previsão ao longo do tempo e marca cada posição prevista com `observed=False`. Detecções robustas podem recuperar explicitamente a trajetória após uma mudança repentina. Previsões que indiquem posições fora do campo ou que excedam o limite de lacuna configurado são descartadas.

O `BallPossessionEstimator` utiliza a distância entre o jogador e a bola, aplicando limiares distintos para a aquisição e a perda da posse. A aquisição, a troca de posse entre jogadores e a perda da posse exigem evidências repetidas (configuráveis). Esse mecanismo de histerese evita oscilações (flicker) quadro a quadro quando jogadores disputam a bola. O pipeline de vídeo ajusta essas janelas de quadros com base na referência de 25 FPS, de modo que uma entrada de 5 FPS não exija cinco vezes mais tempo real para confirmar a mesma ação.

~~~python
from football_analytics.ball import BallPossessionEstimator, BallTracker

ball_tracker = BallTracker()
possession_estimator = BallPossessionEstimator()

ball_candidates = position_estimator.project_balls(
    ball_detections,
    timestamp=timestamp,
)
tracked_ball = ball_tracker.update(
    ball_candidates,
    frame_id=frame_id,
    timestamp=timestamp,
)
possession = possession_estimator.update(
    positions,
    tracked_ball,
    frame_id=frame_id,
    timestamp=timestamp,
)
~~~

Os resumos de posse de bola são calculados com base na duração, e não na contagem de quadros. Eles apresentam os dados em segundos por jogador, segundos para a Equipe A e para a Equipe B, segundos de posse não identificada, além de porcentagens ao longo do tempo atribuídas às equipes conhecidas. A posse de bola pelo goleiro permanece na categoria de "equipe desconhecida" até que a afiliação do goleiro a uma equipe seja definida.

## Passes e finalizações

O EventFrame garante que a posse de bola e os indícios relacionados à bola ...pertencem ao mesmo quadro (*frame*) e registro de tempo (*timestamp*).
O `PassDetector` valida a sequência ordenada: controle, saída/deslocamento da bola e recepção. A recepção por um companheiro de equipe concretiza o passe; a recepção por um adversário identificado ou o término do tempo limite torna o passe incompleto. Uma mudança na identificação do jogador (*player-ID*) ou no jogador mais próximo, sem um deslocamento mensurável da bola, é descartada.
Transições diretas ainda podem ser classificadas como passe em vídeos com baixa taxa de quadros (*low-FPS*), desde que a bola tenha percorrido uma distância suficiente entre as amostras. Se a equipe que recebe a bola for desconhecida, o sistema não gera uma interceptação artificial; além disso, sequências curtas — que não atingem a distância ou a duração mínimas configuradas — são ignoradas.

O `ShotDetector` opera independentemente do xG. Ele exige a existência de uma posse de bola prévia, aceleração da bola, uma velocidade mínima, distanciamento em relação ao ponto de controle inicial e um vetor de velocidade direcionado à meta adversária. Localizações da bola previstas pelo sistema não geram, por padrão, eventos de finalização (*shots*), e um mecanismo de intervalo (*cooldown*) evita a criação de eventos duplicados.

~~~python
from football_analytics.events import EventFrame, PassDetector, ShotDetector

pass_detector = PassDetector()
shot_detector = ShotDetector()

event_frame = EventFrame(possession=possession, ball=tracked_ball)
passes = pass_detector.update(event_frame)
shots = shot_detector.update(event_frame)
~~~

Os passes contêm coordenadas de início e fim, distância, duração, status de conclusão e nível de confiança. As finalizações (*shots*) contêm a distância até o gol, o ângulo visual entre as traves, a velocidade da bola e o nível de confiança. Informações como parte do corpo utilizada, pressão, lances de bola parada e desfecho permanecem nulas até que detectores confiáveis ​​para esses parâmetros sejam implementados.

## Gols Esperados (xG)

O modelo base de xG utiliza uma regressão logística da biblioteca *scikit-learn*, treinada exclusivamente com dados históricos de finalizações — nunca com probabilidades geradas artificialmente. O vetor de atributos (*feature vector*) inicial é estruturado como `distância, ângulo`, sendo o ângulo medido em radianos. Os arquivos históricos podem estar nos formatos CSV, Parquet, JSON ou NDJSON e devem conter:

```text
shot_x, shot_y, distance, angle, goal
```

Campos opcionais incluem `body_part` (parte do corpo), `assist_type` (tipo de assistência), `set_piece` (lance de bola parada), `pressure` (pressão) e `goalkeeper_position` (posição do goleiro). Colunas adicionais desconhecidas presentes na fonte de dados são ignoradas. Cada linha é validada antes do treinamento, incluindo limites do campo, distância não negativa, ângulo entre zero e pi, e um resultado binário (gol ou não gol).

Treine o modelo fora do pipeline de vídeo:

```bash
uv run football-analytics train-xg data/shots.parquet --output models/xg.joblib
```

Em seguida, carregue o artefato treinado e estime a probabilidade do `ShotEvent` (evento de finalização) detectado:

```python
from football_analytics.analytics import ExpectedGoalsModel, aggregate_expected_goals

model = ExpectedGoalsModel.load("models/xg.joblib")
predictions = model.predict_shots(shots)
totals = aggregate_expected_goals(predictions)
```

Cada previsão é restrita ao intervalo `0 <= xG <= 1`. A agregação soma as previsões por jogador e por equipe, validando se os totais gerais coincidem. A detecção permanece separada: o pipeline de vídeo gera as características (features) das finalizações, enquanto apenas o modelo previamente treinado estima o xG.

O `joblib` utiliza um mecanismo de persistência compatível com o `pickle` e pode executar código durante o carregamento. Carregue apenas artefatos de modelo produzidos por este projeto e provenientes de uma fonte confiável; utilize, ainda, o mesmo ambiente de dependências fixadas (locked dependencies) empregado durante o treinamento.

## Estatísticas de partida e visualizações

O `MatchStatisticsCalculator` combina posições de métricas confiáveis, observações de posse de bola, passes, finalizações e previsões de xG. Ele retorna registros imutáveis ​​e serializáveis, denominados `PlayerStatistics` (estatísticas de jogador) e `TeamStatistics` (estatísticas de equipe):

```python
from football_analytics.analytics import MatchStatisticsCalculator

statistics = MatchStatisticsCalculator().calculate(
    positions=positions,
    possessions=possession_observations,
    passes=passes,
    shots=shots,
    xg_predictions=xg_predictions,
)
```

Os registros de jogador incluem: minutos em campo, distância percorrida, velocidade média e máxima, tempo de posse de bola, passes (incluindo precisão), finalizações, resultados conhecidos, xG, posição média e distância progressiva com a bola. As estatísticas de equipe incluem: posse de bola, passes, total de finalizações e xG (gols esperados), entradas no terço final e na área penal, passes progressivos, *field tilt* (inclinação do campo), posse territorial, largura e comprimento médios, compacidade e posicionamento da linha defensiva.

Por padrão, a Equipe A ataca na direção em que o valor de `x` aumenta, enquanto a Equipe B ataca na direção em que o valor de `x` diminui. Utilize `StatisticsConfig(team_a_attacks_right=False)` caso a orientação esteja invertida. Por padrão, consideram-se "passes progressivos" aqueles que avançam pelo menos 10 metros. O *field tilt* representa a proporção do tempo de posse de bola de cada equipe no terço final de ataque; a posse territorial refere-se à proporção correspondente no campo do adversário. As métricas de formação (*shape metrics*) exigem a presença de, pelo menos, dois companheiros de equipe identificados em um determinado instante (*frame*). O indicador PPDA permanece como `None`, uma vez que o modelo de eventos atual ainda não fornece dados confiáveis ​​sobre ações defensivas. Resultados de finalizações desconhecidos e casos em que o xG não está disponível também são representados como `None` — nunca sendo arbitrariamente definidos como zero.

As visualizações operam diretamente sobre coordenadas métricas canônicas e retornam imagens no formato BGR (padrão do OpenCV):

```python
import cv2

from football_analytics.visualization import (
    AveragePositionRenderer,
    HeatmapRenderer,
    ShotMapRenderer,
)

player_heatmap = HeatmapRenderer().render_positions(positions, player_id=9)
cv2.imwrite("outputs/player_9_heatmap.png", player_heatmap)
cv2.imwrite("outputs/pass_heatmap.png", HeatmapRenderer().render_passes(passes))
cv2.imwrite("outputs/shot_map.png", ShotMapRenderer().render(shots, xg_predictions))
average_positions = AveragePositionRenderer().render(positions)
cv2.imwrite("outputs/average_positions.png", average_positions)
```

O mesmo renderizador de mapas de calor (heatmap) suporta a visualização de locais de finalizações e de posses de bola confirmadas, por meio dos métodos
`render_shots()` e `render_possession()`. O tamanho dos marcadores no mapa de finalizações representa o xG (expectativa de gol), caso exista uma previsão correspondente, enquanto gols confirmados são destacados com um anel ao redor.

Exporte qualquer sequência de registros Pydantic para os formatos Parquet, CSV ou JSON:

```python
from football_analytics.storage import MatchDataExporter

exporter = MatchDataExporter("data")
exporter.export("positions", positions)
exporter.export("passes", passes, formats=("parquet", "csv", "json"))
exporter.export("shots", shots)
exporter.export("player_stats", statistics.players)
exporter.export("team_stats", statistics.teams)
```

## Análise de vídeo de ponta a ponta (end-to-end)

O comando `analyze` integra diversas funcionalidades: detecção de objetos, ByteTrack, classificação de uniformes, identificação de pontos-chave do campo, homografia dinâmica, cálculo de posições em métricas reais, rastreamento da bola, análise de posse de bola, passes, finalizações, xG (opcional), estatísticas, exportação de dados, geração de mapas estáticos e a criação de um vídeo com anotações:

```bash
uv run football-analytics analyze match.mp4 \
--object-model models/player_tracker.onnx \
--pitch-model models/football_field.onnx \
--xg-model models/xg.joblib \
--output outputs/match
```

O modelo padrão `models/player_tracker.onnx` é um detector de objetos com resolução de 1280 x 1280; o ByteTrack é utilizado para atribuir identificadores (IDs) de rastreamento persistentes. Os rótulos numéricos do Roboflow são traduzidos automaticamente, sendo: `0=bola`, `1=goleiro`, `2=jogador` e `3=árbitro`. Modelos como `models/best.pt`, que já utilizam rótulos semânticos, permanecem compatíveis. O arquivo `models/football_field.onnx` é o modelo YOLOv8 Pose com 32 pontos-chave (keypoints), utilizado para homografia. O `onnxruntime` faz parte do ambiente de execução (runtime) fixo. O arquivo `models/football_field.pt` — que possui nome semelhante — não é um checkpoint suportado pela Ultralytics e não deve ser passado via CLI.

O artefato xG (expectativa de gol) é opcional. Se o parâmetro `--xg-model` não for utilizado, o restante do pipeline continuará funcionando, mas os valores de xG para as finalizações detectadas permanecerão desconhecidos. Como a biblioteca `joblib` é compatível com `pickle`, carregue apenas artefatos xG em que você confie.

Para garantir identidades de equipe estáveis, forneça as cores RGB representativas dos uniformes:

```bash
uv run football-analytics analyze match.mp4 \
--team-a-color 220,30,30 \
--team-b-color 30,30,220
```

Se as cores não forem informadas, o pipeline agrupa os recortes (crops) dos uniformes em dois clusters via algoritmo K-means (de forma determinística); os nomes "Equipe A" e "Equipe B" resultantes são arbitrários. O modo baseado em cores de referência (colour-anchor) é preferível quando a direção do ataque e as estatísticas específicas de cada equipe são relevantes.

Realize um teste rápido (smoke test) antes de processar uma partida completa:

```bash
uv run football-analytics analyze match.mp4 --max-frames 100
```

Opções úteis incluem `--device cpu`, `--device mps`, `--device 0`, `--export-format parquet csv json`, `--no-annotated-video` e `--verbose`. O diretório de saída padrão é `outputs/<nome-do-video>/`, contendo:

```text
annotated.mp4
report.json
data/
detections.parquet
tracks.parquet
pitch_keypoints.parquet
positions.parquet
ball_positions.parquet
possessions.parquet
events.parquet
passes.parquet
shots.parquet
xg_predictions.parquet
player_stats.parquet
``` team_stats.parquet
visualizations/
pass_heatmap.png
shot_heatmap.png
possession_map.png
shot_map.png
average_positions.png
players/player_<id>_heatmap.png
```

Quando um quadro não apresenta uma homografia confiável, as posições métricas e o estado da bola são omitidos para esse quadro. O pipeline de processamento de vídeo utiliza um limiar de confiança calibrado de 0,40 para o modelo de 32 pontos-chave fornecido, mantendo, ao mesmo tempo, os filtros baseados em *inliers* do algoritmo RANSAC e no erro de reprojeção. Isso evita que uma geometria de câmera defasada comprometa as estatísticas da partida, sem que a maioria dos quadros válidos seja descartada.