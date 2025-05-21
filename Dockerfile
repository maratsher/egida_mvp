# CUDA Runtime с Ubuntu 22.04
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Системные зависимости
RUN apt-get update && apt-get install -y \
    python3-pip libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# --- python ---
# 1) onnxruntime-gpu тащит себе numpy-1.26.x как зависимость
RUN pip3 install --no-cache-dir onnx onnxruntime-gpu==1.22.0 

# 2) гарантируем, что версия numpy < 2
RUN pip3 install --no-cache-dir 'numpy<2.0' pyyaml watchdog opencv-python-headless

# Код приложения
WORKDIR /app
COPY ./ /app

# Запуск пакета src
ENTRYPOINT ["python3", "-m", "src"]