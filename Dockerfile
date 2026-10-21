FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04@sha256:a5dc56795348ea8fe46305612afaf7f9882cb438b08a582d4e5bd762369d7922

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

COPY pyproject.toml requirements-lock.txt ./

RUN touch README.md

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r requirements-lock.txt

COPY . .

RUN pip install --no-cache-dir --no-deps -e .

CMD ["bash"]