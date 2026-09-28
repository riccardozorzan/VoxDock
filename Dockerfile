# syntax=docker/dockerfile:1.7
# =============================================================================
#  VoxDock (voxdock-tts) — containerised GPU TTS service on dots.tts
#     image: voxdock-tts:latest      (name declared in app/__init__.py)
#  Base: NVIDIA CUDA runtime image (no host pollution, GPU via nvidia-container-toolkit)
#
#  DEVIATION FROM SPEC (documented on purpose):
#    The spec asked for `nvidia/cuda:12.1.1-runtime-ubuntu22.04` + PyTorch cu121.
#    dots.tts 0.3.x requires `torch>=2.8.0`, and PyTorch stopped publishing cu121
#    wheels after 2.5.x. The official dots.tts benchmark stack is
#    "torch 2.8 + CUDA 12.8", so we use the CUDA 12.8 runtime image + cu128
#    wheels. Any host driver >= 525 (CUDA 12.x) works with these.
# =============================================================================
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu128

# ---- Environment -----------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Keep the 4.4 GB of weights on the persistent docker volume
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    # Reduce allocator fragmentation on the 6 GB RTX 4050
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    # RTX 4050 Laptop = Ada Lovelace / SM 8.9
    TORCH_CUDA_ARCH_LIST="8.9" \
    OMP_NUM_THREADS=8 \
    TTS_MODEL_NAME=dots-studio/dots.tts-mf \
    TTS_PRECISION=bfloat16 \
    TTS_OPTIMIZE=true \
    TTS_WARMUP_ON_OPTIMIZE=true \
    TTS_MAX_GENERATE_LENGTH=128 \
    TTS_MAX_SEQUENCE_LENGTH=2048 \
    TTS_VRAM_LIMIT_GB=5.5

# ---- System dependencies ---------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        git \
        curl \
        ca-certificates \
        ffmpeg \
        libsndfile1 \
        libgomp1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3  /usr/local/bin/pip \
    && python --version

# ---- Python tooling --------------------------------------------------------
RUN python -m pip install --upgrade "pip>=24.2" "setuptools>=68" wheel

# ---- PyTorch (CUDA 12.8 build, matches the dots.tts reference stack) -------
RUN python -m pip install \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCH_VERSION}" \
        --index-url "${TORCH_CUDA_INDEX}" \
    && python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# ---- Python requirements (dots.tts, fastapi, librosa, noisereduce, ...) ----
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt

# ---- Application code ------------------------------------------------------
COPY app ./app
COPY scripts ./scripts
RUN mkdir -p /app/data/references /app/data/input /app/data/temp_chunks /app/data/output

# ---- Runtime ---------------------------------------------------------------
EXPOSE 8000

# torch.compile warmup on a laptop GPU is slow -> generous start period.
HEALTHCHECK --interval=30s --timeout=15s --start-period=3600s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "120", "--log-level", "info"]
