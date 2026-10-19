# Test image for diffusion attention backends (fp8_fa_sm120 on SM89 / SM120).
#
# Installs SGLang with the diffusion extra from this checkout, without the Rust
# extensions. The CUDA devel base matches docker/Dockerfile: a full generation
# JIT-compiles FlashInfer kernels on first use, which needs nvcc.
#
# The start script keeps the container alive, so a RunPod pod does not restart it, and
# starts sshd when PUBLIC_KEY is set; run the tests and benchmarks from a shell
# (benchmark/diffusion_fp8_attn/README.md).
#
#   docker build -f docker/diffusion-fp8-attn.Dockerfile -t sglang-diffusion-fp8 .
#   docker run --rm --gpus all sglang-diffusion-fp8 python -m pytest -q \
#       python/sglang/multimodal_gen/test/unit/test_fp8_fa_sm120_attn.py
ARG CUDA_VERSION=13.0.3
ARG PYTHON_VERSION=3.12

FROM ghcr.io/astral-sh/uv:0.9.0 AS uv

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04
ARG PYTHON_VERSION

# git: version metadata. libnuma1: NUMA binding of diffusion workers.
# openssh-server: SSH access on rented GPUs, started by the start script.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git libnuma1 ca-certificates openssh-server \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv

# uv-managed CPython ships Python.h, which Triton needs for its driver stub.
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    PYTHONUNBUFFERED=1
RUN uv venv --python ${PYTHON_VERSION} --seed ${VIRTUAL_ENV}

WORKDIR /sgl-workspace/sglang
COPY python ./python
COPY benchmark/diffusion_fp8_attn ./benchmark/diffusion_fp8_attn

# The Rust extensions need the rust/ workspace, which this image does not copy.
ENV SGLANG_BUILD_RUST_EXTS=none
RUN uv pip install --no-cache -e "./python[diffusion]" pytest

# SSH login shells do not see the image's ENV; give them the venv, CUDA and the checkout.
RUN printf '%s\n' \
    'export VIRTUAL_ENV=/opt/venv CUDA_HOME=/usr/local/cuda SGLANG_BUILD_RUST_EXTS=none' \
    'export PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH' \
    'cd /sgl-workspace/sglang' > /etc/profile.d/sglang.sh

COPY docker/diffusion-fp8-attn-start.sh /usr/local/bin/start.sh
CMD ["/usr/local/bin/start.sh"]
