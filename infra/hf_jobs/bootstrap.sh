#!/usr/bin/env bash
# Komi POC — HF Job bootstrap. Runs inside a stock public CUDA+PyTorch image (see Dockerfile for the reference image):
#   1. installs musubi-tuner at the pinned commit into a venv that reuses the image's torch,
#   2. hands over to train_job.py (cache + base weights download, training, checkpoint push, resume).
# Started by the one-line entry command built in launch.py, which creates $KOMI_VENV, installs huggingface_hub and
# downloads this folder (jobs/) from the PRIVATE dataset repo at a pinned revision.
#
# Environment: MUSUBI_COMMIT (required unless FAKE_TRAINER=1), KOMI_VENV, WORK_DIR, plus everything train_job.py reads
# (KIND, EXPERT, RUN_NAME, CACHE_REPO, CACHE_PATH, LORA_REPO, MAX_STEPS, SAVE_EVERY, PUSH_EVERY_MIN, RESUME=1, ...).
# Secret: HF_TOKEN (never echoed; `set -x` is deliberately not used).
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export KOMI_VENV="${KOMI_VENV:-/tmp/komi_venv}"
export WORK_DIR="${WORK_DIR:-/tmp/komi}"
export MUSUBI_DIR="${MUSUBI_DIR:-$WORK_DIR/musubi-tuner}"
export HF_HOME="${HF_HOME:-$WORK_DIR/hf_home}"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
PY="$KOMI_VENV/bin/python"
mkdir -p "$WORK_DIR" "$HF_HOME"
t0=$(date +%s)
stamp() { echo "[bootstrap +$(( $(date +%s) - t0 ))s] $*"; }

stamp "start $(date -u +%FT%TZ) job=${JOB_ID:-local} run=${RUN_NAME:-?} kind=${KIND:-?} fake=${FAKE_TRAINER:-0}"
stamp "user=$(id -u):$(id -g) host=$(hostname) cpus=$(nproc)"
df -h "$WORK_DIR" 2>/dev/null | tail -1 || true
free -g 2>/dev/null | sed -n 2p || true
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi
[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN secret is missing"; exit 2; }
[ -x "$PY" ] || { echo "venv $KOMI_VENV is missing (the entry command creates it)"; exit 2; }

if [ "${FAKE_TRAINER:-0}" != "1" ]; then
  : "${MUSUBI_COMMIT:?MUSUBI_COMMIT is required}"
  "$PY" - <<'PYEOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PYEOF
  if [ ! -f "$MUSUBI_DIR/pyproject.toml" ]; then
    stamp "fetching musubi-tuner @ $MUSUBI_COMMIT"
    "$PY" - <<'PYEOF'
import io, os, tarfile, urllib.request
commit, dest = os.environ["MUSUBI_COMMIT"], os.environ["MUSUBI_DIR"]
url = f"https://github.com/kohya-ss/musubi-tuner/archive/{commit}.tar.gz"
data = urllib.request.urlopen(url, timeout=120).read()
with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
    root = tf.getnames()[0].split("/")[0]
    try:
        tf.extractall(os.path.dirname(dest), filter="data")
    except TypeError:  # Python < 3.11.4
        tf.extractall(os.path.dirname(dest))
os.rename(os.path.join(os.path.dirname(dest), root), dest)
print("musubi-tuner extracted to", dest, f"({len(data) / 1e6:.1f} MB)")
PYEOF
  fi
  # keep the image's torch: pip must not replace it
  TORCH_VER="$("$PY" -c 'import torch; print(torch.__version__.split("+")[0])')"
  TV_VER="$("$PY" -c 'import torchvision; print(torchvision.__version__.split("+")[0])' 2>/dev/null || true)"
  { echo "torch==$TORCH_VER"; [ -n "$TV_VER" ] && echo "torchvision==$TV_VER"; } > "$WORK_DIR/constraints.txt"
  stamp "installing musubi-tuner dependencies (torch $TORCH_VER stays)"
  "$PY" -m pip install -q -c "$WORK_DIR/constraints.txt" -e "$MUSUBI_DIR" tensorboard
  # opencv-python needs libGL, which slim images lack: use the headless build of the same version
  "$PY" -m pip uninstall -q -y opencv-python >/dev/null 2>&1 || true
  "$PY" -m pip install -q -c "$WORK_DIR/constraints.txt" "opencv-python-headless==4.10.0.84"
  "$PY" - <<'PYEOF'
import importlib, os
import torch
if not torch.cuda.is_available():  # CPU-only dry runs: wan/modules/t5.py calls torch.cuda.current_device() at import
    torch.cuda.current_device = lambda: 0
kind = os.environ.get("KIND", "wan")
mod = {"wan": "musubi_tuner.wan_train_network", "kf": "musubi_tuner.qwen_image_train_network"}[kind]
importlib.import_module(mod)
import accelerate, bitsandbytes, diffusers, transformers, cv2
print("imports OK:", mod, "| accelerate", accelerate.__version__, "| transformers", transformers.__version__,
      "| diffusers", diffusers.__version__, "| bitsandbytes", bitsandbytes.__version__, "| cv2", cv2.__version__)
PYEOF
  # torch.compile (--compile in EXTRA_ARGS) needs a C/C++ compiler for inductor; the runtime image has none
  if [[ "${EXTRA_ARGS:-}" == *"--compile"* ]] && ! command -v g++ >/dev/null 2>&1; then
    stamp "installing g++ for torch.compile"
    (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends g++ >/dev/null) \
      || echo "WARNING: could not install g++ (torch.compile may fail)"
  fi
  stamp "musubi-tuner ready"
fi

if [ "${BOOTSTRAP_ONLY:-0}" = "1" ]; then
  stamp "BOOTSTRAP_ONLY=1: environment verified, not training"
  echo 'KOMI_EVENT {"event": "bootstrap_ok"}'
  exit 0
fi

stamp "handing over to train_job.py"
exec "$PY" "$JOB_DIR/train_job.py"
