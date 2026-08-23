#!/usr/bin/env bash
# Reconstruction de forme sur GPU, avec arrêt automatique.
#
# Le script est écrit pour une machine facturée à la minute : il installe,
# exécute, écrit son résultat, puis **s'arrête de lui-même**. Une session
# oubliée coûte plus cher que le calcul lui-même.
set -euo pipefail

BUNDLE="${BUNDLE:-/workspace/bundle}"
OUT="${OUT:-/workspace/out}"
BACKEND="${BACKEND:-vggt}"
STOP_WHEN_DONE="${STOP_WHEN_DONE:-1}"

log() { printf '\n=== %s ===\n' "$1"; }

# L'arrêt est armé avant tout travail : une erreur d'installation ne doit pas
# laisser tourner la machine. `trap` couvre le succès comme l'échec.
finish() {
  code=$?
  log "terminé (code $code)"
  if [ "$STOP_WHEN_DONE" = "1" ]; then
    log "arrêt du pod"
    # runpodctl est présent dans les images officielles RunPod.
    if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
      runpodctl stop pod "$RUNPOD_POD_ID"
    else
      echo "runpodctl indisponible — arrêter le pod manuellement"
    fi
  fi
}
trap finish EXIT

log "environnement"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

log "dépendances"
pip install --quiet --no-cache-dir "numpy<2" pillow
case "$BACKEND" in
  vggt)
    pip install --quiet --no-cache-dir git+https://github.com/facebookresearch/vggt.git
    ;;
  mapanything)
    pip install --quiet --no-cache-dir mapanything
    ;;
  *)
    echo "backend inconnu : $BACKEND"; exit 2 ;;
esac

log "inférence ($BACKEND)"
mkdir -p "$OUT"
python /workspace/infer_shape.py \
  --images "$BUNDLE/images" \
  --out "$OUT" \
  --backend "$BACKEND"

log "résultat"
ls -la "$OUT"
du -sh "$OUT"
