# The Stage 12 training/inference image (spec section 32): Python, PyTorch, Transformers,
# TRL, PEFT, bitsandbytes, Accelerate, Datasets, pytest, and this project's package,
# pinned exactly to requirements.lock.
#
# This is NOT the candidate sandbox (docker/evaluator/Dockerfile) and must never be used
# to execute generated Python (spec section 33, CLAUDE.md's Security rule). The two
# containers exist for opposite reasons: this one is trusted -- it runs *this project's
# own* orchestrator, training loop and inference code, with GPU access and no execution
# limits -- while the evaluator image is deliberately untrusted-hostile: no GPU, no
# network, a non-root user, dropped capabilities, and hard resource limits, because it
# runs model-generated code. Never point generated candidate code at this image. See
# docker/README.md for the full separation.
#
# The pip wheels for torch/nvidia-* bundle their own CUDA runtime (see requirements.lock),
# so this builds from the same slim base as the sandbox rather than an nvidia/cuda image;
# the host still needs an NVIDIA driver and the NVIDIA Container Toolkit, and containers
# from this image must be run with `--gpus all` (or an equivalent device request) to see
# the GPU at all.
#
# Build:
#   docker build -t python-dpo-trainer:1.0 -f docker/Dockerfile .
#
# Run (from the project root, with the GPU and the project's data/config mounted):
#   docker run --rm --gpus all \
#     -v "$(pwd)":/workspace -w /workspace \
#     python-dpo-trainer:1.0 \
#     python -m python_dpo experiment run --config configs/experiments/qwen_python_dpo_v1.yaml

FROM python:3.12-slim

WORKDIR /workspace

# Docker itself is needed inside this container only if the orchestrator is run from
# inside it (it shells out to the `docker` CLI for candidate_execution/candidate_evaluation,
# spec section 34) -- omitted here since the intended, verified way to run this image is
# with the host's Docker socket mounted in and the `docker` CLI installed via the host's
# own package manager on top of this image when that path is needed. This base image does
# not attempt Docker-in-Docker.

COPY requirements.lock /workspace/requirements.lock
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.lock

COPY pyproject.toml /workspace/pyproject.toml
COPY src/ /workspace/src/
RUN pip install --no-cache-dir --root-user-action=ignore --no-deps -e ".[training]"
