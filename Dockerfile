# 1. Базовый образ с TensorRT и Python3
FROM nvcr.io/nvidia/tensorrt:23.02-py3

# 2. Обновляем пакеты и ставим системные зависимости
RUN apt-get update && apt-get install -y \
    python3-opencv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 3. Устанавливаем необходимые Python‑библиотеки
RUN pip3 install --no-cache-dir \
    numpy \
    onnx \
    onnx-tensorrt \
    pycuda \
    # + любые ваши зависимости, например:
    watchdog

# 4. Создаём рабочую директорию и копируем в неё ваш код
WORKDIR /app
COPY ./code /app

# 5. Указываем точку входа — просто запускаем ваш главный скрипт
#    Предполагаем, что внутри app.py у вас есть цикл обработки кадров
ENTRYPOINT ["python3", "app.py"]
