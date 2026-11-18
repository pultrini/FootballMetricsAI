# Notas da refatoração

## Fase 0

O teste de caracterização em
`tests/characterization/test_analyze.py` executa `football-analytics analyze` com
`--max-frames 100` e compara o `report.json` e as tabelas Parquet selecionadas
com os arquivos em `tests/characterization/golden/`.

Vídeos e modelos não são versionados. Para gerar os golden no código atual,
defina os caminhos e execute:

```bash
export FA_TEST_CLIP=/caminho/para/clip.mp4
export FA_TEST_OBJECT_MODEL=/caminho/para/object-model.onnx
export FA_TEST_PITCH_MODEL=/caminho/para/pitch-model.onnx
uv run python scripts/make_golden.py
```

Opcionalmente, um modelo xG pode ser fornecido por `FA_TEST_XG_MODEL` ou
`--xg-model`. O teste lento usa as três variáveis obrigatórias e é pulado quando
alguma entrada ou golden ainda não existe.

O relatório ignora campos explicitamente voláteis (timestamps, caminhos e UUIDs).
Inteiros, strings e estrutura são comparados exatamente; valores de ponto
flutuante usam tolerância relativa de `1e-9`.

## Fase 1

Os módulos foram reorganizados sem alterar algoritmos ou formatos de saída.
Os caminhos antigos permanecem como shims de compatibilidade temporários.

`PitchHomography` permanece no domínio, mas as chamadas OpenCV foram extraídas
para a porta `HomographyBackend` e o adaptador
`OpenCvHomographyBackend`. Isso preserva o algoritmo e seus parâmetros sem
introduzir `cv2` no domínio. O shim legado injeta o adaptador para manter a
construção `PitchHomography()` compatível.

## Fase 2

`AnalysisSession` é o caso de uso de vídeo inteiro e recebe o motor de análise,
fonte, sink e reporter por injeção. `bootstrap.py` é a composition root: a CLI
apenas faz parsing, configura logging e executa a sessão. O pipeline existente
permanece como motor interno nesta fase para preservar exatamente a sequência e
os valores calculados; a extração dos componentes do hot path não é duplicada.

## Fase 3

`SessionConfig` em `application/config.py` é um modelo Pydantic imutável e
versionado (`schema_version` e `config_version`, ambos inicialmente `"1"`).
Pode ser carregado diretamente de YAML com `SessionConfig.from_yaml(path)`.
Quando a CLI recebe `--config`, os valores do YAML são carregados primeiro e
somente as flags explicitamente fornecidas sobrescrevem a configuração; os
defaults do parser não sobrescrevem valores do arquivo. Os defaults foram
mantidos iguais aos valores existentes nos adaptadores e no pipeline.

## Fase 4

A sessão agora expõe falhas esperadas como `AnalysisError` com códigos estáveis:
`VIDEO_UNREADABLE`, `MODEL_NOT_FOUND`, `CONFIG_INVALID`, `INFERENCE_FAILED`,
`HOMOGRAPHY_UNAVAILABLE` e `UNKNOWN`. A CLI converte esses códigos,
respectivamente, em exit codes distintos (2 a 6 e 1 para `UNKNOWN`) e não
expõe detalhes de credenciais em mensagens de erro.

O adaptador `JsonLinesProgressReporter` pode ser ativado com
`analyze --progress jsonl`. Ele emite um objeto JSON por linha para os eventos
`analysis.started`, `analysis.progress`, `analysis.completed` e
`analysis.failed`; o reporter padrão continua sendo no-op. O sink de vídeo
mantém a chamada única `write_analysis` para preservar a composição existente
dos artefatos e os formatos de saída.

Os módulos antigos continuam disponíveis como shims de compatibilidade pública
durante a migração. Eles não são usados pela composition root nova, e serão
removidos somente em uma fase posterior após a migração dos consumidores.

## Fase 5

`AnalysisSession.process_segment()` adiciona processamento incremental sem
alterar `run()`: o motor não é resetado entre segmentos, e o offset de origem
produz `frame_id` e timestamps globais. `WindowMetrics` contém somente métricas
da janela, com `coverage` e IDs canônicos; `final_report()` soma somente campos
aditivos e deixa métricas pontuais como `null`. Distância e posse usam o FPS da
fonte; a comparação aditiva aceita a tolerância numérica padrão do domínio.

`QueueWorker` mantém uma sessão por `stream_id`, descarta duplicatas, guarda
mensagens fora de ordem e sinaliza lacunas com `SEGMENT_GAP`. Os eventos públicos
usam o envelope Pydantic versionado `schema_version="1"` e podem ser publicados
por qualquer implementação de `MessagePublisher`.

Os contratos neutros para Java/Go são gerados por
`uv run python scripts/export_contracts.py` a partir de
`application/contracts.py`. O teste `tests/test_contracts.py` falha quando os
JSON Schemas versionados divergem dos modelos.
