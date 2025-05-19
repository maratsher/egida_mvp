import os
import yaml
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from collections import namedtuple


class EngineBuilder:
    """
    Строит TensorRT engine из ONNX, если его нет по указанному пути.
    Конфигурация YAML должна содержать:
      onnx_path: путь к .onnx
      engine_path: путь для сохранения .engine
      max_workspace_size: объем рабочей памяти в байтах (по умолчанию 1GB)
      fp16: true/false — использовать FP16 (опционально)
    """
    def __init__(self, cfg_path: str):
        cfg = yaml.safe_load(open(cfg_path, 'r'))
        self.onnx_path = cfg['onnx_path']
        self.engine_path = cfg['engine_path']
        self.max_workspace = int(cfg.get('max_workspace_size', 1<<30))
        self.fp16 = bool(cfg.get('fp16', False))
        self.logger = trt.Logger(trt.Logger.WARNING)

    def build(self):
        if os.path.exists(self.engine_path):
            print(f"Engine already exists: {self.engine_path}")
            return
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)
        with open(self.onnx_path, 'rb') as f:
            data = f.read()
            if not parser.parse(data):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError('Failed to parse ONNX file')
        config = builder.create_builder_config()
        config.max_workspace_size = self.max_workspace
        if self.fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        print("Building engine... This may take a while.")
        engine = builder.build_engine(network, config)
        with open(self.engine_path, 'wb') as f:
            f.write(engine.serialize())
        print(f"Engine saved at: {self.engine_path}")