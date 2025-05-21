# import os
# import yaml
# import cv2
# import numpy as np
# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit
# from collections import namedtuple


# class EngineBuilder:
#     """
#     Строит TensorRT engine из ONNX, если его нет по указанному пути.
#     Конфигурация YAML должна содержать:
#       onnx_path: путь к .onnx
#       engine_path: путь для сохранения .engine
#       max_workspace_size: объем рабочей памяти в байтах (по умолчанию 1GiB)
#       fp16: true/false — использовать FP16 (опционально)
#       input_shape: [C, H, W] — для профиля оптимизации
#     """
#     def __init__(self, cfg_path: str):
#         cfg = yaml.safe_load(open(cfg_path, 'r'))
#         self.onnx_path = cfg['onnx_path']
#         self.engine_path = cfg['engine_path']
#         self.max_workspace = int(cfg.get('max_workspace_size', 1 << 30))
#         self.fp16 = bool(cfg.get('fp16', False))
#         self.input_shape = tuple(cfg.get('input_shape', [3, 800, 800]))
#         self.logger = trt.Logger(trt.Logger.WARNING)

#     def build(self):
#         # Инициализация плагинов TensorRT (например, ROIAlign)
#         trt.init_libnvinfer_plugins(self.logger, '')

#         if os.path.exists(self.engine_path):
#             print(f"Engine already exists: {self.engine_path}")
#             return

#         builder = trt.Builder(self.logger)
#         network = builder.create_network(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
#         parser = trt.OnnxParser(network, self.logger)

#         # Парсинг ONNX-модели
#         with open(self.onnx_path, 'rb') as f:
#             onnx_data = f.read()
#             if not parser.parse(onnx_data):
#                 for i in range(parser.num_errors):
#                     print(parser.get_error(i))
#                 raise RuntimeError('Failed to parse ONNX file with TensorRT parser')

#         # Создание конфигурации сборки
#         config = builder.create_builder_config()
#         # Устанавливаем рабочий пул памяти
#         config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.max_workspace)
#         if self.fp16:
#             config.set_flag(trt.BuilderFlag.FP16)

#         # Добавляем профиль оптимизации для динамических входов
#         profile = builder.create_optimization_profile()
#         input_name = network.get_input(0).name
#         min_shape = (1,) + self.input_shape
#         # одни и те же значения для min, opt, max — статическая форма
#         profile.set_shape(input_name, min_shape, min_shape, min_shape)
#         config.add_optimization_profile(profile)

#         print("Building TensorRT engine... This may take several minutes.")
#         serialized_engine = builder.build_serialized_network(network, config)
#         if serialized_engine is None:
#             raise RuntimeError('Failed to build serialized TensorRT engine')

#         os.makedirs(os.path.dirname(self.engine_path), exist_ok=True)
#         with open(self.engine_path, 'wb') as f:
#             f.write(serialized_engine)
#         print(f"Engine successfully saved at: {self.engine_path}")

#         # Инициализация плагинов TensorRT (например, ROIAlign)
#         trt.init_libnvinfer_plugins(self.logger, '')

#         if os.path.exists(self.engine_path):
#             print(f"Engine already exists: {self.engine_path}")
#             return

#         builder = trt.Builder(self.logger)
#         network = builder.create_network(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
#         parser = trt.OnnxParser(network, self.logger)

#         # Парсинг ONNX-модели
#         with open(self.onnx_path, 'rb') as f:
#             onnx_data = f.read()
#             if not parser.parse(onnx_data):
#                 for i in range(parser.num_errors):
#                     print(parser.get_error(i))
#                 raise RuntimeError('Failed to parse ONNX file with TensorRT parser')

#         config = builder.create_builder_config()
#         config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.max_workspace)
#         if self.fp16:
#             config.set_flag(trt.BuilderFlag.FP16)

#         print("Building TensorRT engine... This may take several minutes.")
#         # Сборка сериализованного движка
#         serialized_engine = builder.build_serialized_network(network, config)
#         if serialized_engine is None:
#             raise RuntimeError('Failed to build serialized TensorRT engine')

#         # Сохраняем в файл
#         os.makedirs(os.path.dirname(self.engine_path), exist_ok=True)
#         with open(self.engine_path, 'wb') as f:
#             f.write(serialized_engine)
#         print(f"Engine successfully saved at: {self.engine_path}")