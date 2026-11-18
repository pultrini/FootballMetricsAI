# Problemas encontrados nas velocidades

## Escopo da auditoria

Foi auditada a saída de 12 fps em [outputs/yolo26_12_fps_4metros](../outputs/yolo26_12_fps_4metros).
O campo `maximum_speed_mps` significa metros por segundo: as posições estão em coordenadas
métricas de um campo de 105 m por 68 m e a velocidade é calculada como distância/tempo.

O parâmetro de 4 m é usado pelo estimador de posse (`possession_acquisition_distance_m` e
`possession_release_distance_m`). Ele altera a posse e os eventos de passe, mas não participa
do cálculo da velocidade dos jogadores.

## Evidências observadas

Na saída de 12 fps:

- há 8 passes, todos completos;
- o maior valor é `135,987 m/s` (aproximadamente `489,6 km/h`);
- vários jogadores apresentam máximas acima de 100 m/s;
- algumas velocidades médias também são incompatíveis com um jogador humano, por exemplo,
  aproximadamente 16,38 m/s para o jogador 4;
- os maiores saltos ocorrem em grupos de frames, principalmente 4-5, 71-72, 256-258 e
  259-260.

O maior salto identificado nas posições exportadas do resultado comparável de 12 fps foi:

| Jogador | Frames | Distância | Intervalo | Velocidade |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 259 -> 260 | 11,332 m | 0,0833 s | 135,987 m/s |

Outros saltos próximos ao mesmo problema incluem 10,919 m para o jogador 11 entre os frames
256-257 e 9,483 m para o jogador 14 entre os frames 256-257. Vários jogadores saltarem
simultaneamente é evidência de problema global de projeção, e não de aceleração real.

## Causas prováveis

### 1. Instabilidade da homografia

Os pontos-chave do campo são detectados frame a frame. Uma pequena mudança nesses pontos pode
deslocar a transformação de pixels para coordenadas do campo em vários metros, sobretudo nas
regiões mais afastadas ou nas bordas da imagem. O efeito aparece como um salto coletivo nas
posições métricas.

### 2. Detecção deslocada

O ponto usado para representar o jogador pode mudar abruptamente quando a caixa detectada muda
de tamanho, quando há oclusão ou quando o modelo confunde partes de jogadores próximos. Uma
mudança de poucos pixels pode virar muitos metros após a homografia.

### 3. Rastreamento/identidade aceita uma posição ruim

O resolvedor de identidade usa uma velocidade máxima de 12 m/s para limitar associações entre
tracklets. Esse limite ajuda a associar identidades, mas não rejeita toda posição absurda já
aceita pelo mesmo track. Consequentemente, a posição ruim chega ao cálculo estatístico.

### 4. Máxima sem filtro físico

Em [statistics.py](../src/football_analytics/domain/analytics/statistics.py), cada segmento
válido entra no conjunto de velocidades e o maior valor é retornado. Um único salto de 11 m em
1/12 s domina `maximum_speed_mps`.

### 5. Resolução temporal baixa

Em 12 fps, cada amostra está separada por aproximadamente 83 ms. Isso não causa sozinho uma
velocidade de 136 m/s, mas torna saltos de detecção mais impactantes e reduz a informação
disponível para distinguir movimento real de erro.

## Sugestões de correção

1. **Rejeitar segmentos fisicamente impossíveis antes de calcular estatísticas.** Usar um limite
   conservador, por exemplo 12-13 m/s para jogadores, marcar o segmento como inválido e não
   somá-lo à distância nem à velocidade máxima.
2. **Usar um filtro robusto por track.** Aplicar mediana móvel ou filtro Hampel às coordenadas
   antes da diferenciação. A mediana deve ser aplicada com cuidado para não apagar sprints reais.
3. **Interpolar apenas pequenos gaps rejeitados.** Depois de remover um outlier isolado,
   interpolar sua posição entre vizinhos válidos. Não interpolar longos trechos sem evidência.
4. **Suavizar a homografia.** Reutilizar a última homografia confiável por poucos frames,
   rejeitar mudanças com erro de reprojeção muito alto e validar o deslocamento dos pontos-chave.
5. **Revalidar a posição do jogador.** Exigir coerência com a posição anterior, a velocidade
   estimada e a área do campo antes de atualizar o estado do track.
6. **Registrar diagnósticos.** Exportar `position_outlier`, `homography_confidence`,
   `segment_speed_mps` e o motivo da rejeição para permitir auditoria sem inferência visual.
7. **Separar métricas brutas e corrigidas.** Não sobrescrever o dado original; gerar colunas ou
   arquivos `positions_corrected` e comparar distância, média e máxima.
8. **Confirmar com 25 fps ou vídeo original.** Se o salto desaparecer em maior frequência, o
   problema é predominantemente amostragem/detecção; se persistir no mesmo instante, investigar
   homografia e caixas.

## Notebook de validação

O notebook [notebooks/correcao_velocidade.ipynb](../notebooks/correcao_velocidade.ipynb) implementa
quatro cenários sobre as posições exportadas:

1. bruto;
2. rejeição de segmentos acima do limite físico;
3. mediana móvel;
4. rejeição seguida de interpolação local.

Ele também lista os maiores saltos, compara as métricas por jogador e salva CSVs em
`outputs/yolo26_12_fps_4metros/velocity_audit/`. Essas correções são experimentais para
validação; a correção definitiva deve ser aplicada no produtor das posições/homografia e
depois reexecutar o pipeline completo.
