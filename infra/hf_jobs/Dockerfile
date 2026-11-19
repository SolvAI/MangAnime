# Komi POC — reference training image (CUDA + PyTorch + musubi-tuner at a pinned commit).
#
# This is the *definition* of the environment the HF Jobs run in. It is NOT pushed to a registry (decision D-008: no
# private registry for the POC): launch.py starts the stock public base image below and bootstrap.sh performs the same
# installation steps at job start (measured: ~40 s on rtx-pro-6000, ~70 s on cpu-basic). Keep the two in sync:
# the pins here must match configs/train_infra.yaml (musubi.commit, jobs.image).
#
# Build locally if a registry becomes available:
#   docker build -t <registry>/komi-poc-train:f8a1b037 -f infra/hf_jobs/Dockerfile infra/hf_jobs
#   then: launch.py ... --image <registry>/komi-poc-train:f8a1b037 --env KOMI_VENV=/opt/komi_venv --env MUSUBI_DIR=/opt/musubi-tuner
# (bootstrap.sh skips the download when $MUSUBI_DIR/pyproject.toml exists; pip then only verifies the pins.)

# pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime  (2025-08-06)  — cu128 is required for RTX PRO 6000 Blackwell
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385

ARG MUSUBI_COMMIT=f8a1b03794a49239a3539015075f5123d6c07d66
ENV KOMI_VENV=/opt/komi_venv \
    MUSUBI_DIR=/opt/musubi-tuner \
    MUSUBI_COMMIT=${MUSUBI_COMMIT} \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# venv on top of the image's torch (system site packages), so pip never replaces torch 2.8.0+cu128
RUN python -m venv --system-site-packages ${KOMI_VENV} \
 && ${KOMI_VENV}/bin/pip install --no-cache-dir huggingface_hub==1.32.0

# musubi-tuner at the pinned commit (tarball: the runtime image has no git)
RUN python - <<'PY'
import io, os, tarfile, urllib.request
commit, dest = os.environ["MUSUBI_COMMIT"], os.environ["MUSUBI_DIR"]
data = urllib.request.urlopen(f"https://github.com/kohya-ss/musubi-tuner/archive/{commit}.tar.gz", timeout=120).read()
with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
    root = tf.getnames()[0].split("/")[0]
    tf.extractall(os.path.dirname(dest))
os.rename(os.path.join(os.path.dirname(dest), root), dest)
PY

# musubi pins (pyproject.toml @ f8a1b037): accelerate 1.15.0, diffusers 0.40.0, transformers 5.17.0, safetensors 0.8.0,
# av 17.1.0, huggingface-hub 1.32.0, bitsandbytes (latest, 0.50.2 on 2026-11-10). opencv-python -> headless (no libGL).
RUN printf 'torch==2.8.0\ntorchvision==0.23.0\n' > /opt/constraints.txt \
 && ${KOMI_VENV}/bin/pip install --no-cache-dir -c /opt/constraints.txt -e ${MUSUBI_DIR} tensorboard \
 && ${KOMI_VENV}/bin/pip uninstall -y opencv-python \
 && ${KOMI_VENV}/bin/pip install --no-cache-dir -c /opt/constraints.txt opencv-python-headless==4.10.0.84

# job scripts (the HF job fetches the current versions from the private dataset repo instead)
COPY bootstrap.sh train_job.py logparse.py fake_trainer.py /opt/komi_jobs/
WORKDIR /opt/komi_jobs
ENTRYPOINT ["bash", "/opt/komi_jobs/bootstrap.sh"]
