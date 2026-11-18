#!/usr/bin/env bash
# Teste A/B: commit pré-refatoração (be2df94) x versão atual, mesmo clipe, CPU.
#
# Uso (da raiz do repositório):
#   chmod +x scripts/ab_test.sh
#   scripts/ab_test.sh --allow-extra-new          # flags extras vão para o compare_outputs.py
#
# Variáveis opcionais:
#   CLIP=/caminho/clip.mp4  MAX_FRAMES=359  OLD_COMMIT=be2df94  scripts/ab_test.sh
set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
OLD_COMMIT="${OLD_COMMIT:-be2df94}"
OLD_DIR="${OLD_DIR:-$BASE/../fa-old}"
CLIP="${CLIP:-$BASE/data/0bfacc_0_12fps.mp4}"
OBJ="${OBJ:-$BASE/models/best.pt}"
PITCH="${PITCH:-$BASE/models/football_field.onnx}"
MAX_FRAMES="${MAX_FRAMES:-100}"
OUT="$BASE/outputs/testeAB"

for f in "$CLIP" "$OBJ" "$PITCH"; do
  [ -f "$f" ] || { echo "arquivo não encontrado: $f" >&2; exit 2; }
done

echo "== 1/4 Preparando worktree do commit $OLD_COMMIT"
if [ ! -d "$OLD_DIR" ]; then
  git -C "$BASE" worktree add "$OLD_DIR" "$OLD_COMMIT"
fi
if [ "$(git -C "$OLD_DIR" rev-parse HEAD)" != "$(git -C "$BASE" rev-parse "$OLD_COMMIT")" ]; then
  echo "worktree em $OLD_DIR não está em $OLD_COMMIT" >&2
  exit 2
fi
(cd "$OLD_DIR" && uv sync)

rm -rf "$OUT/old" "$OUT/new"
mkdir -p "$OUT"

# Só a flag do modelo difere entre as versões. Todo o resto é idêntico.
ARGS_OLD=("$CLIP" --object-model "$OBJ" --pitch-model "$PITCH" --device cpu --max-frames "$MAX_FRAMES")
ARGS_NEW=("$CLIP" --object-model "$OBJ" --pitch-model "$PITCH" --device cpu --max-frames "$MAX_FRAMES")

echo "== 2/4 Versão ANTIGA ($OLD_COMMIT)"
SECONDS=0
(cd "$OLD_DIR" && uv run football-analytics analyze "${ARGS_OLD[@]}" --output "$OUT/old")
echo "   tempo: ${SECONDS}s"

echo "== 3/4 Versão NOVA ($(git -C "$BASE" rev-parse --short HEAD))"
SECONDS=0
(cd "$BASE" && uv run football-analytics analyze "${ARGS_NEW[@]}" --output "$OUT/new")
echo "   tempo: ${SECONDS}s"

echo "== 4/4 Comparando"
status=0
(cd "$BASE" && uv run python scripts/compare_outputs.py "$OUT/old" "$OUT/new" "$@") || status=$?

if [ "$status" -eq 0 ]; then
  echo "A/B OK: saídas equivalentes."
else
  echo "A/B FALHOU: veja as linhas DIFF acima." >&2
fi
echo "Saídas em: $OUT/{old,new}"
echo "Para remover o worktree: git worktree remove \"$OLD_DIR\""
exit "$status"