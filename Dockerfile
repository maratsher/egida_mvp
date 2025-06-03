# ── Базовый образ: минимальный CUDA-рантайм + Ubuntu 22.04 ──────────────
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# ── Системные зависимости (OpenCV, build-tools) ─────────────────────────
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev build-essential git \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --upgrade pip

RUN apt-get update && apt-get install -y fonts-dejavu-core

# ── Python-зависимости ──────────────────────────────────────────────────
#  Отдельно кладём requirements.txt, чтобы слоё сохранялся при изменении кода.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Копируем исходники ──────────────────────────────────────────────────
COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python3", "-m", "src"]
