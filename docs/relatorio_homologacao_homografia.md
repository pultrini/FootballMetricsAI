# Relatório de homologação da homografia

## Resumo executivo

O baseline atual **não foi homologado para produzir métricas físicas finais**.
Ele cobre boa parte dos frames, mas gera velocidades incompatíveis com o
movimento de jogadores. A principal evidência é o p99 de velocidade de
**57,30 m/s**, enquanto o critério de aceitação definido é **12 m/s**.

O baseline deve ser mantido como referência técnica para comparar as próximas
variantes de estabilização, não como saída final de velocidade, distância ou
ângulo no campo.

## Entrada analisada

| Item | Valor |
|---|---|
| Vídeo | `data/0bfacc_0_18fps.mp4` |
| Duração | 30 s |
| Frames | 540 |
| FPS | 18 |
| Resolução | 1920 × 1080 |
| Modelo de objetos | `models/best.pt` |
| Modelo de campo | `models/football_field.onnx` |
| Execução | CPU |
| xG | Não executado; nenhum modelo xG foi fornecido |

Os hashes dos artefatos estão registrados em
`notebooks/homography/results/00_setup_cache.json`.

## Procedimento

O vídeo foi processado pela CLI oficial da aplicação:

```bash
uv run football-analytics analyze \
  data/0bfacc_0_18fps.mp4 \
  --object-model models/best.pt \
  --pitch-model models/football_field.onnx \
  --device cpu \
  --no-annotated-video \
  --output notebooks/homography/cache/baseline_full \
  --export-format parquet json \
  --progress jsonl
```

Foram analisadas as posições métricas, as correspondências de keypoints, a
razão de inliers, o RMSE de reprojeção e os saltos temporais do centro da
imagem projetado no campo.

## Cobertura e rastreamento

| Métrica | Resultado |
|---|---:|
| Frames com posições métricas | 485 / 540 |
| Cobertura de posições | 89,8% |
| Linhas de posições | 10.248 |
| IDs brutos | 29 |
| Identidades canônicas | 27 |
| Tracklets mesclados | 2 |
| Eventos detectados | 9 |

## Velocidade dos jogadores

| Métrica | Resultado | Critério |
|---|---:|---:|
| Mediana | 3,84 m/s | — |
| P95 | 33,67 m/s | — |
| P99 | 57,30 m/s | ≤ 12 m/s |
| Máximo | 137,90 m/s | — |
| Amostras acima de 12 m/s | 1.447 | idealmente ≤ 1% |
| Percentual acima de 12 m/s | 14,16% | — |

O relatório estatístico por jogador apresentou velocidade máxima de
aproximadamente **271,47 m/s** em pelo menos um jogador. Esses valores são
fisicamente implausíveis e indicam erro de projeção, instabilidade temporal
da homografia ou troca de correspondência/rastreamento.

## Qualidade da homografia

| Métrica | Resultado |
|---|---:|
| Frames com homografia estimada | 504 |
| Keypoints por frame — mediana | 16 |
| Keypoints por frame — mínimo/máximo | 12 / 19 |
| Razão de inliers — mediana | 1,000 |
| Razão de inliers — mínima | 0,842 |
| RMSE de reprojeção — mediana | 1,43 m |
| RMSE de reprojeção — P95 | 1,92 m |

A razão de inliers alta não foi suficiente para garantir uma transformação
estável. Portanto, ela não deve ser usada como único critério de aprovação.

## Estabilidade temporal

| Métrica | Resultado |
|---|---:|
| Salto mediano do centro projetado | 0,16 m |
| P95 do salto | 3,60 m |
| Maior salto | 10,55 m |
| Saltos acima de 5 m | 18 |

Os maiores saltos foram observados nos frames **95, 269, 270, 349, 350,
352, 354, 481 e 490**, entre outros. A lista completa está em
`notebooks/homography/results/baseline_homography_diagnostics.csv`.

## Critérios de homologação

| Critério | Limite |
|---|---:|
| Cobertura sem corte | ≥ 80% |
| P99 da velocidade | ≤ 12 m/s |
| Salto residual P95 | ≤ 10 px |
| Custo adicional | ≤ 25 ms/frame |

O critério de salto residual ainda não foi medido com compensação explícita do
movimento da câmera. Portanto, não deve ser considerado aprovado apenas com o
salto bruto do centro projetado.

## Decisão

**Baseline: REPROVADO para homologação como saída final de métricas físicas.**

Motivos:

1. O p99 de velocidade é 4,8 vezes maior que o limite de 12 m/s.
2. 14,16% das amostras de velocidade excedem 12 m/s.
3. Há saltos temporais de até 10,55 m no centro projetado.
4. O RMSE de reprojeção mediano é 1,43 m.
5. A razão de inliers alta não prediz estabilidade suficiente.

O baseline pode continuar sendo usado como **referência de comparação**.

## Evidências visuais

As visualizações reais dos frames críticos estão em:

- [Frame 4 — keypoints](../notebooks/homography/results/real_frame_004_keypoints.png)
- [Frame 4 — homografia](../notebooks/homography/results/real_frame_004_homography.png)
- [Frame 5 — keypoints](../notebooks/homography/results/real_frame_005_keypoints.png)
- [Frame 5 — homografia](../notebooks/homography/results/real_frame_005_homography.png)
- [Frame 77 — keypoints](../notebooks/homography/results/real_frame_077_keypoints.png)
- [Frame 77 — homografia](../notebooks/homography/results/real_frame_077_homography.png)
- [Frame 257 — keypoints](../notebooks/homography/results/real_frame_257_keypoints.png)
- [Frame 257 — homografia](../notebooks/homography/results/real_frame_257_homography.png)
- [Diagnóstico temporal](../notebooks/homography/results/baseline_homography_diagnostics.png)

## Próxima etapa recomendada

Comparar, usando o mesmo vídeo e os mesmos modelos:

1. baseline atual;
2. gate geométrico e temporal;
3. retenção da última homografia válida;
4. suavização dos quatro cantos projetados;
5. compensação do movimento da câmera;
6. fallback separado em pixels/s quando não houver homografia confiável.

Cada variante deve ser aprovada somente se apresentar cobertura, velocidade,
estabilidade e custo dentro dos critérios acima. Sem ground truth manual, o
resultado deve ser descrito como **validado por consistência temporal**, não
como precisão absoluta.

## Artefatos de suporte

- [Resumo JSON do baseline](../notebooks/homography/results/baseline_real_summary.json)
- [Diagnóstico por frame](../notebooks/homography/results/baseline_homography_diagnostics.csv)
- [Relatório bruto da aplicação](../notebooks/homography/cache/baseline_full/report.json)
- [Posições Parquet](../notebooks/homography/cache/baseline_full/data/positions.parquet)
