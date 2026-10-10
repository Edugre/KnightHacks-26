#!/usr/bin/env bash
# One-time / per-session setup on AUP Learning Cloud (VSCode Server GPU, R9700).
# Idempotent. Installs into ~/.local and ~/bin (persistent NFS home); never touches system torch.
set -euo pipefail

PROFILE_LINE='export PATH="$HOME/.local/bin:$HOME/bin:$PATH"'
grep -qxF "$PROFILE_LINE" ~/.bashrc 2>/dev/null || echo "$PROFILE_LINE" >> ~/.bashrc
export PATH="$HOME/.local/bin:$HOME/bin:$PATH"

PIP="pip install --user --break-system-packages -q"

torch_ver() { python3 -c "import torch; print(torch.__version__)"; }
BEFORE="$(torch_ver)"
echo "torch before: $BEFORE"

$PIP ultralytics lap pyyaml fastapi uvicorn python-multipart httpx pytest pydantic imageio-ffmpeg ftfy regex transformers accelerate pillow
# Ultralytics pulls GUI OpenCV (needs libGL); swap for headless. Both share cv2/, so force-reinstall headless.
pip uninstall -y -q --break-system-packages opencv-python 2>/dev/null || true
$PIP --force-reinstall --no-deps opencv-python-headless
# CLIP for YOLOE text prompts; --no-deps so it can't replace torch.
python3 -c "import clip" 2>/dev/null || $PIP --no-deps "git+https://github.com/ultralytics/CLIP.git"

# ffmpeg: static binary from imageio-ffmpeg.
mkdir -p ~/bin
[ -x ~/bin/ffmpeg ] || ln -sf "$(python3 -c 'import imageio_ffmpeg as f; print(f.get_ffmpeg_exe())')" ~/bin/ffmpeg

AFTER="$(torch_ver)"
echo "torch after:  $AFTER"
if [ "$BEFORE" != "$AFTER" ]; then
  echo "ERROR: torch changed ($BEFORE -> $AFTER). ROCm build may be gone." >&2
  exit 1
fi
python3 -c "import torch, cv2, clip, lap, ultralytics, transformers; assert torch.cuda.is_available(); print('ok:', torch.cuda.get_device_name(0), '| cv2', cv2.__version__, '| ultralytics', ultralytics.__version__, '| transformers', transformers.__version__)"
ffmpeg -version | head -1

# Fetch every model now, while the network is allowed: ./run.sh loads them offline (censor/netguard.py).
# Ultralytics' weights_dir is the relative path "weights", so YOLOE and MobileCLIP land in <repo>/weights:
# run this from the repo root, which is where run.sh runs.
cd "$(dirname "$0")/.."
python3 - <<'EOF'
import os
from huggingface_hub import snapshot_download
from ultralytics.utils import SETTINGS
from ultralytics.utils.downloads import attempt_download_asset

print("vlm:", snapshot_download(os.environ.get("CENSOR_VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")))
os.makedirs(SETTINGS["weights_dir"], exist_ok=True)
os.chdir(SETTINGS["weights_dir"])
for name in ("yoloe-11s-seg.pt", "mobileclip_blt.ts"):
    print("weights:", os.path.abspath(attempt_download_asset(name)))
EOF
