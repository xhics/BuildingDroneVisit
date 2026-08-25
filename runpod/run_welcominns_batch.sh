#!/usr/bin/env bash
# Lot GPU unique WelcomINNS : validation, VGGT, contrôle, empaquetage, arrêt.
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/BuildingDroneVisit}"
HOTEL_ID="${HOTEL_ID:-welcominns-boucherville}"
BUNDLE="${BUNDLE:-${PROJECT_ROOT}/runpod/bundle}"
VENV_DIR="${VENV_DIR:-/workspace/.venvs/buildingdrone-batch}"
CACHE_ROOT="${CACHE_ROOT:-/workspace/.cache}"
BATCH_ID="${BATCH_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-/workspace/out/${HOTEL_ID}-${BATCH_ID}}"
BACKENDS="${BACKENDS:-vggt}"
RUN_SEMANTIC="${RUN_SEMANTIC:-0}"
STOP_WHEN_DONE="${STOP_WHEN_DONE:-1}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_SYSTEM="${PYTHON_SYSTEM:-python3}"
VGGT_COMMIT="${VGGT_COMMIT:-a288dd0f14786c93483e45524328726ab7b1b4ce}"
MAPANYTHING_COMMIT="${MAPANYTHING_COMMIT:-3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9}"
STATE_DIR="${OUT}/.stages"
LOG_DIR="${OUT}/logs"

log() { printf '\n=== %s ===\n' "$1"; }
fail() { printf 'ERREUR: %s\n' "$1" >&2; exit 2; }

finish() {
  code=$?
  if [ "$DRY_RUN" = "1" ]; then
    return
  fi
  log "fin du lot (code ${code})"
  if [ "$STOP_WHEN_DONE" = "1" ]; then
    if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
      log "arrêt automatique du pod"
      runpodctl stop pod "$RUNPOD_POD_ID" || true
    else
      printf 'ATTENTION: arrêt automatique indisponible; arrêter le pod manuellement.\n' >&2
    fi
  fi
}
trap finish EXIT

run_stage() {
  name="$1"
  shift
  marker="${STATE_DIR}/${name}.done"
  if [ -f "$marker" ]; then
    log "${name} déjà terminé"
    return
  fi
  log "$name"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  touch "$marker"
}

preflight() {
  [ -f "$PROJECT_ROOT/runpod/infer_shape.py" ] || fail \
    "script d'inférence absent: ${PROJECT_ROOT}/runpod/infer_shape.py"
  [ -f "$PROJECT_ROOT/runpod/validate_batch.py" ] || fail \
    "validateur absent: ${PROJECT_ROOT}/runpod/validate_batch.py"
  if [ "$RUN_SEMANTIC" = "1" ]; then
    [ -d "$PROJECT_ROOT/.git" ] || fail \
      "RUN_SEMANTIC=1 exige le dépôt complet sous ${PROJECT_ROOT}"
  fi
  [ -f "$BUNDLE/shape_input.json" ] || fail "lot absent: ${BUNDLE}"
  [ "${ACK_REFERENCE_ONLY:-0}" = "1" ] || fail \
    "le lot est DEMO_ONLY; relancer avec ACK_REFERENCE_ONLY=1"
  "$PYTHON_SYSTEM" "$PROJECT_ROOT/runpod/validate_batch.py" input \
    --bundle "$BUNDLE" --report "$OUT/input_validation.json"
  if [ "$DRY_RUN" = "1" ]; then
    return
  fi
  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi absent"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  memory_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  [ "${memory_mb:-0}" -ge 22000 ] || fail "22 Go de VRAM minimum requis"
}

setup_environment() {
  mkdir -p "$(dirname "$VENV_DIR")" "$CACHE_ROOT/pip" "$CACHE_ROOT/huggingface"
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    "$PYTHON_SYSTEM" -m venv --system-site-packages "$VENV_DIR"
  fi
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  PIP_CACHE_DIR="$CACHE_ROOT/pip" python -m pip install --upgrade pip setuptools wheel
  PIP_CACHE_DIR="$CACHE_ROOT/pip" python -m pip install pillow
  case ",${BACKENDS}," in
    *,vggt,*)
      PIP_CACHE_DIR="$CACHE_ROOT/pip" python -m pip install \
        "numpy<2" "git+https://github.com/facebookresearch/vggt.git@${VGGT_COMMIT}"
      ;;
  esac
  case ",${BACKENDS}," in
    *,mapanything,*)
      PIP_CACHE_DIR="$CACHE_ROOT/pip" python -m pip install \
        "git+https://github.com/facebookresearch/map-anything.git@${MAPANYTHING_COMMIT}"
      ;;
  esac
  python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA indisponible"
print("GPU:", torch.cuda.get_device_name(0))
print("Torch:", torch.__version__)
PY
}

run_shape_backend() {
  backend="$1"
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  HF_HOME="$CACHE_ROOT/huggingface" TORCH_HOME="$CACHE_ROOT/torch" \
    python "$PROJECT_ROOT/runpod/infer_shape.py" \
      --images "$BUNDLE/images" --out "$OUT/$backend" --backend "$backend"
}

run_semantic() {
  [ "$RUN_SEMANTIC" = "1" ] || return
  bash "$PROJECT_ROOT/runpod/setup_semantic_gpu.sh"
  # shellcheck disable=SC1090
  source /workspace/.venvs/buildingdrone/bin/activate
  export HOTEL_PIPELINE_WORK="${PROJECT_ROOT}/work"
  export HF_HOME="$CACHE_ROOT/huggingface"
  export TORCH_HOME="$CACHE_ROOT/torch"
  export SAM2_CHECKPOINT="$CACHE_ROOT/sam2/sam2.1_hiera_large.pt"
  hotel-pipeline conditioning semantic-detect "$HOTEL_ID" \
    --limit 6 --device cuda --cache-only --segmentation sam2 \
    --model-id IDEA-Research/grounding-dino-base \
    --threshold 0.22 --text-threshold 0.20 \
    --prompt "hotel building. hotel entrance. entrance door. entrance canopy. brick column. structural column. horizontal structural beam. window. hotel sign. road sign. lamp post. rooftop air conditioning unit. gutter. balcony. evergreen tree. deciduous tree."
  hotel-pipeline conditioning semantic-link "$HOTEL_ID"
  hotel-pipeline conditioning register-colmap-lidar "$HOTEL_ID"
  hotel-pipeline conditioning semantic-register "$HOTEL_ID"
  hotel-pipeline conditioning semantic-surface "$HOTEL_ID"
}

validate_and_pack() {
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  expected_images="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["images"]))' "$BUNDLE/shape_input.json")"
  python "$PROJECT_ROOT/runpod/validate_batch.py" output \
    --out "$OUT" --backends "$BACKENDS" --expected-images "$expected_images" \
    --report "$OUT/output_validation.json"
  cp "$BUNDLE/shape_input.json" "$OUT/shape_input.json"
  find "$OUT" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
    xargs -0 sha256sum > "$OUT/SHA256SUMS.txt"
  tar -C "$(dirname "$OUT")" -czf "${OUT}.tar.gz" "$(basename "$OUT")"
  printf 'RESULTAT=%s\n' "${OUT}.tar.gz"
}

mkdir -p "$STATE_DIR" "$LOG_DIR"
preflight
if [ "$DRY_RUN" = "1" ]; then
  log "simulation validée; aucun calcul GPU lancé"
  exit 0
fi
run_stage environment setup_environment
if [ "$RUN_SEMANTIC" = "1" ]; then
  run_stage semantic run_semantic
fi
old_ifs="$IFS"
IFS=','
for backend in $BACKENDS; do
  backend="$(printf '%s' "$backend" | xargs)"
  case "$backend" in
    vggt|mapanything) ;;
    *) fail "backend interdit: ${backend}" ;;
  esac
  run_stage "shape_${backend}" run_shape_backend "$backend"
done
IFS="$old_ifs"
run_stage validate validate_and_pack
