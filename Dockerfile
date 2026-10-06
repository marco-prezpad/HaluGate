FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /workspace

COPY pyproject.toml .

RUN touch README.md

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir numpy scipy pandas pyarrow scikit-learn \
    transformers "accelerate>=1.1.0" datasets evaluate pyyaml tqdm matplotlib seaborn \
    pytest pytest-cov jupyter black ruff

COPY . .

RUN pip install -e .[all]

CMD ["bash"]
