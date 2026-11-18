# Laboratório de homografia

Laboratório exploratório para diagnosticar e comparar estabilizações da
homografia imagem → campo. Ele não altera o pacote de produção em `src/`.

## Execução

```bash
uv sync --group notebooks
export FA_CLIP=/caminho/clip.mp4
export FA_OBJECT_MODEL=/caminho/models/object.pt
export FA_PITCH_MODEL=/caminho/models/football_field.onnx
uv run --group notebooks jupyter lab notebooks/homography
```

Execute os notebooks na ordem `00` a `09`. O `02b` é opcional e o `08` é
experimental. Cada notebook pode ser executado sem os artefatos externos:
nesse caso, ele salva um relatório de pré-requisitos em `results/` e não
fabrica métricas.

## Ordem e objetivo

| Notebook | Objetivo |
| --- | --- |
| N0 | Validar configuração e preparar cache |
| N1 | Reproduzir a auditoria baseline |
| N2 | Inspecionar frames de falha |
| N2b | Anotar ground truth manual, opcional |
| N2c | Testar atribuição, filtros, RANSAC e MAGSAC antes das variantes |
| N10 | Detectar linhas do campo e testar homografia nos 10 frames críticos |
| validação pipeline | Validar escala manual e estabilização temporal da homografia |
| validação híbrida roboflow | Combinar inferência Roboflow, ROI do campo e geometria de linhas |
| N3 | Separar salto bruto de movimento de câmera |
| N4 | Avaliar gates geométrico e temporal |
| N5 | Comparar suavização pelos cantos |
| N6 | Medir alinhamento de linhas |
| N7 | Fundir homografia e movimento da câmera |
| N8 | Avaliar o modelo experimental de tripé |
| N9 | Comparar variantes e gerar decisão |

O laboratório usa CPU, sementes fixas e caminhos definidos por variáveis de
ambiente. Vídeos, modelos, caches e resultados não devem ser versionados.

## Visualizações

Os notebooks salvam imagens Matplotlib em `results/`:

- N1: cantos da imagem e grade projetados no campo canônico;
- N2: frame com keypoints numerados e coloridos por confiança, seguido da
  projeção da homografia no campo;
- N9: visualização de referência para acompanhar a decisão final.

Sem um vídeo configurado, as figuras usam um frame sintético determinístico
apenas para validar o fluxo visual. Com `FA_CLIP`, o helper
`lab.viz.read_frame()` lê o frame real e as células de análise devem fornecer
os keypoints e a matriz baseline calculados naquele frame.
