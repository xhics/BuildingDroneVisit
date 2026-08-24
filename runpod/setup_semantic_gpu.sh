#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/BuildingDroneVisit}"
VENV_DIR="${VENV_DIR:-/workspace/.venvs/buildingdrone}"
CACHE_ROOT="${CACHE_ROOT:-/workspace/.cache}"
SAM2_ROOT="${SAM2_ROOT:-/workspace/vendor/sam2}"
SAM2_COMMIT="2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_CHECKPOINT="${CACHE_ROOT}/sam2/sam2.1_hiera_large.pt"
SAM2_SHA256="2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"

mkdir -p "${CACHE_ROOT}/huggingface" "${CACHE_ROOT}/torch" \
  "${CACHE_ROOT}/pip" "${CACHE_ROOT}/sam2" "$(dirname "${VENV_DIR}")" \
  "$(dirname "${SAM2_ROOT}")"

python3 -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
PIP_CACHE_DIR="${CACHE_ROOT}/pip" python -m pip install \
  -e "${PROJECT_ROOT}[semantic-vision,sfm,geo]" opencv-python-headless pytest

if [ ! -d "${SAM2_ROOT}/.git" ]; then
  git clone --no-checkout https://github.com/facebookresearch/sam2.git "${SAM2_ROOT}"
fi
git -C "${SAM2_ROOT}" fetch --depth 1 origin "${SAM2_COMMIT}"
git -C "${SAM2_ROOT}" checkout --detach "${SAM2_COMMIT}"
SAM2_BUILD_CUDA=0 PIP_CACHE_DIR="${CACHE_ROOT}/pip" \
  python -m pip install -e "${SAM2_ROOT}"

if [ ! -f "${SAM2_CHECKPOINT}" ] || ! \
  printf '%s  %s\n' "${SAM2_SHA256}" "${SAM2_CHECKPOINT}" | \
    sha256sum --check --status; then
  curl --fail --location --retry 4 --continue-at - \
    --output "${SAM2_CHECKPOINT}" \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
fi

printf '%s  %s\n' "${SAM2_SHA256}" "${SAM2_CHECKPOINT}" | sha256sum --check

HF_HOME="${CACHE_ROOT}/huggingface" TORCH_HOME="${CACHE_ROOT}/torch" \
python - <<'PY'
import torch
from sam2.build_sam import build_sam2

assert torch.cuda.is_available(), "CUDA indisponible"
print("GPU:", torch.cuda.get_device_name(0))
print("Torch:", torch.__version__)
print("SAM 2: import valide", build_sam2.__module__)
PY
