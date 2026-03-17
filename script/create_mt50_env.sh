#!/usr/bin/env bash

set -euo pipefail

ENV_NAME=${1:-lingbot-mt50}
INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN:-0}
PYTHON_VERSION=${PYTHON_VERSION:-3.10.16}

if ! command -v conda >/dev/null 2>&1; then
    for candidate in \
        "${HOME}/miniconda3/etc/profile.d/conda.sh" \
        "${HOME}/anaconda3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh"; do
        if [ -f "${candidate}" ]; then
            # shellcheck disable=SC1090
            . "${candidate}"
            break
        fi
    done
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found. Install Miniconda or Anaconda first." >&2
    exit 1
fi

conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
conda install -n "${ENV_NAME}" -c conda-forge ffmpeg cmake ninja pkg-config -y

# shellcheck disable=SC1091
. "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-mt50.txt
python -m pip install lerobot==0.3.3 --no-deps

if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then
    python -m pip install flash-attn --no-build-isolation
fi

python -m pip install -e . --no-deps

echo
echo "Environment '${ENV_NAME}' is ready."
echo "Run the smoke test with:"
echo "  conda activate ${ENV_NAME}"
echo "  python script/check_mt50_env.py"
