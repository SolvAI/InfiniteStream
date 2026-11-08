#!/usr/bin/env bash
# One-shot install: upstream code, Python dependencies and model weights (about 23 GB).
# Run it inside a fresh Python 3.10 environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

UPSTREAM_COMMIT=eb8705b2f00b7c361193b25aa3aae9e7edc94667
CLIP_COMMIT=d05afc436d78f1c48dc0dbf8e5980a9d471f35f6

python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 10))' \
  || { echo "Python 3.10 is required (the flash-attn wheel is built for it)."; exit 1; }
command -v nvidia-smi >/dev/null || { echo "No NVIDIA driver found (nvidia-smi is missing)."; exit 1; }

echo "==> Fetching Causal Forcing"
if [ ! -d Causal-Forcing ]; then
  git clone https://github.com/thu-ml/Causal-Forcing.git
  git -C Causal-Forcing checkout --quiet "$UPSTREAM_COMMIT"
fi

echo "==> Installing Python dependencies"
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install "git+https://github.com/openai/CLIP.git@$CLIP_COMMIT"
ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abi${ABI}-cp310-cp310-linux_x86_64.whl"
pip install -e ./Causal-Forcing

echo "==> Downloading model weights"
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir Causal-Forcing/wan_models/Wan2.1-T2V-1.3B \
  --exclude "assets/*" --exclude "examples/*"
hf download zhuhz22/Causal-Forcing causal-forcing++/framewise-2step.pt --local-dir Causal-Forcing/checkpoints
curl -fL https://github.com/madebyollin/taehv/raw/main/taew2_1.pth -o Causal-Forcing/checkpoints/taew2_1.pth

echo "==> Done. Start the stream with: python live.py"
