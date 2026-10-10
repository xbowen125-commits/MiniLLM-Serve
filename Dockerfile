# syntax=docker/dockerfile:1.7

# Official PyTorch runtime: Linux amd64, Python 3.12, PyTorch 2.11, CUDA 12.8.
# The digest prevents a mutable tag from silently changing production builds.
ARG BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime@sha256:eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="MiniLLM-Serve" \
      org.opencontainers.image.description="Single-GPU OpenAI-compatible local LLM serving" \
      org.opencontainers.image.version="0.1.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/minillm/.cache/huggingface \
    MINILLM_HOST=0.0.0.0 \
    MINILLM_PORT=8000

# Ubuntu's Python is PEP 668 managed and its minimal installation omits
# ensurepip. Install the matching venv package while the build still runs as
# root, then remove apt metadata from the same layer.
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Keep the base image's CUDA-enabled torch visible, while installing all
# application packages into a PEP 668-safe virtual environment.
COPY pyproject.toml README.md constraints-container.txt ./
RUN python -c "import torch; assert torch.__version__.startswith('2.11.0+cu128'), torch.__version__; assert torch.version.cuda == '12.8', torch.version.cuda" \
    && python -m venv --system-site-packages /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

# Install the stable dependency layer before source code, so ordinary source
# edits do not reinstall dependencies. The inherited torch satisfies both the
# project requirement and constraint. Accelerate is installed without resolving
# dependencies so pip has no path that can download a second torch wheel;
# pip check below still verifies all of its requirements.
RUN python -m pip install --no-cache-dir --constraint constraints-container.txt \
      fastapi "uvicorn[standard]" pydantic transformers psutil \
    && python -m pip install --no-cache-dir --no-deps \
      --constraint constraints-container.txt accelerate

COPY src ./src

RUN python -m pip install --no-cache-dir --no-deps . \
    && python -m pip check \
    && python -c "from pathlib import Path; import accelerate, fastapi, pydantic, torch, transformers, uvicorn; assert torch.__version__.startswith('2.11.0+cu128'), torch.__version__; assert torch.version.cuda == '12.8', torch.version.cuda; assert not Path(torch.__file__).is_relative_to('/opt/venv'), torch.__file__; assert transformers.__version__ == '5.17.0'"

# Fixed numeric IDs make bind-mount permissions predictable on Linux hosts.
RUN groupadd --gid 10001 minillm \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin minillm \
    && mkdir -p /home/minillm/.cache/huggingface /app \
    && chown -R 10001:10001 /home/minillm /app

USER 10001:10001

EXPOSE 8000
STOPSIGNAL SIGTERM

# This only checks readiness metadata; it never invokes model generation.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD ["/opt/venv/bin/python", "-c", "import json,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3)); assert d.get('status')=='ok' and d.get('loaded') is True"]

# One worker is mandatory: every worker would load another model and scheduler.
CMD ["/opt/venv/bin/python", "-m", "uvicorn", "minillm.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--lifespan", "on", "--timeout-graceful-shutdown", "300"]
