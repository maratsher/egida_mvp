# import os
# import yaml
# import cv2
# import numpy as np
# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit
# from collections import namedtuple

# Buffers = namedtuple('Buffers', ['host', 'device'])

# class TRTInference:
#     """
#     Загружает TensorRT-engine и выполняет инференс.
#     Отображает маску на изображении, если debug=true.
#     Конфигурация YAML должна содержать:
#       engine_path: путь к .engine
#       input_shape: [C, H, W]
#       debug: true/false
#       debug_output: папка для сохранения debug-изображений
#       overlay_color: [B, G, R]
#     """
#     def __init__(self, cfg_path: str):
#         cfg = yaml.safe_load(open(cfg_path, 'r'))
#         self.engine_path = cfg['engine_path']
#         self.input_shape = tuple(cfg['input_shape'])
#         self.debug = bool(cfg.get('debug', False))
#         self.debug_dir = cfg.get('debug_output', './debug')
#         os.makedirs(self.debug_dir, exist_ok=True)
#         self.color = tuple(cfg.get('overlay_color', [0, 255, 0]))

#         # Инициализация рантайма и загрузка движка
#         runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
#         trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), '')
#         with open(self.engine_path, 'rb') as f:
#             engine_data = f.read()
#         self.engine = runtime.deserialize_cuda_engine(engine_data)
#         if self.engine is None:
#             raise RuntimeError('Failed to deserialize TensorRT engine')
#         self.context = self.engine.create_execution_context()

#         # Установка входной формы (батч=1)
#         input_idx = [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)][0]
#         self.context.set_binding_shape(input_idx, (1,) + self.input_shape)

#         # Подготовка буферов
#         self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

#     def _allocate_buffers(self):
#         inputs, outputs, bindings = [], [], []
#         stream = cuda.Stream()
#         for idx in range(self.engine.num_bindings):
#             shape = self.context.get_binding_shape(idx)
#             size = int(np.prod(shape))
#             dtype = trt.nptype(self.engine.get_binding_dtype(idx))
#             host_mem = cuda.pagelocked_empty(size, dtype)
#             dev_mem = cuda.mem_alloc(host_mem.nbytes)
#             bindings.append(int(dev_mem))
#             buf = Buffers(host=host_mem, device=dev_mem)
#             if self.engine.binding_is_input(idx):
#                 inputs.append(buf)
#             else:
#                 outputs.append(buf)
#         return inputs, outputs, bindings, stream

#     def infer(self, image_path: str) -> dict:
#         img = cv2.imread(image_path)
#         c, h, w = self.input_shape
#         inp = cv2.resize(img, (w, h)).astype(np.float32) / 255.0
#         inp = np.transpose(inp, (2, 0, 1)).ravel()
#         np.copyto(self.inputs[0].host, inp)
#         cuda.memcpy_htod(self.inputs[0].device, self.inputs[0].host)

#         self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

#         for buf in self.outputs:
#             cuda.memcpy_dtoh(buf.host, buf.device)

#         # Разбор выходов: boxes, scores, classes и masks
#         boxes = self.outputs[0].host.reshape(-1, 4)
#         scores = self.outputs[1].host.tolist()
#         classes = self.outputs[2].host.astype(np.int32).tolist()
#         masks = self.outputs[3].host.reshape(-1, h, w)

#         result = {'boxes': boxes.tolist(), 'scores': scores, 'classes': classes}

#         if self.debug:
#             overlay = img.copy()
#             for m in masks:
#                 mask_bin = (m > 0.5).astype(np.uint8)
#                 colored = np.zeros_like(img)
#                 colored[mask_bin == 1] = self.color
#                 overlay = cv2.addWeighted(overlay, 1.0, colored, 0.5, 0)
#             fname = os.path.basename(image_path)
#             cv2.imwrite(os.path.join(self.debug_dir, fname), overlay)

#         return result