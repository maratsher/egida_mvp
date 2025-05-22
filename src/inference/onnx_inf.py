# src/inference/onnx_inf.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort
import torch
from pathlib import Path
from typing import Tuple, Optional

from ultralytics.utils import ops
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml


class Yolov8SegONNX:
    """
    Инференс YOLOv8-Seg (ONNX Runtime).
    Возвращает одну бинарную маску (uint8 0/255) с максимальным confidence
    и оверлей для отладки.
    """

    # ------------------------------------------------------------------ #
    #  ИНИЦИАЛИЗАЦИЯ                                                     #
    # ------------------------------------------------------------------ #
    def __init__(self, cfg: dict):
        self.model_path: str | Path = cfg["onnx_model"]
        self.imgsz: Tuple[int, int] = (
            (cfg.get("imgsz"), cfg.get("imgsz"))
            if isinstance(cfg.get("imgsz"), int)
            else tuple(cfg.get("imgsz", [640, 640]))
        )
        self.conf: float = cfg.get("conf_thres", 0.25)
        self.iou: float = cfg.get("iou_thres", 0.7)
        self.device: int | None = cfg.get("device", 0)
        self.debug: bool = bool(cfg.get("debug", False))
        self.nc: int = int(cfg.get("nc", 1))
        self.overlay_color: Tuple[int, int, int] = tuple(
            cfg.get("overlay_color", [0, 255, 0])
        )  # BGR

        # ONNX Runtime session
        providers, provider_options = (["CPUExecutionProvider"], None)
        if torch.cuda.is_available():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = [{"device_id": str(self.device)}, {}]

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers,
            provider_options=provider_options,
        )
        self.input_name = self.session.get_inputs()[0].name

        # class names (не обязательны, но полезны)
        #self.names = YAML.load(check_yaml("coco8.yaml"))["names"]

    # ------------------------------------------------------------------ #
    #  ГЛАВНЫЙ ВХОД                                                      #
    # ------------------------------------------------------------------ #
    def __call__(self, img_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        prep, r_info = self._preprocess(img_bgr)
        pred, protos = self.session.run(None, {self.input_name: prep})
        mask = self._postprocess(img_bgr, pred, protos, r_info)
        if mask is None:
            return None, None
        overlay = self._draw_overlay(img_bgr, mask)
        return mask, overlay

    # ------------------------------------------------------------------ #
    #  ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ                                            #
    # ------------------------------------------------------------------ #
    def _letterbox(self, im: np.ndarray, new_shape: Tuple[int, int]):
        h0, w0 = im.shape[:2]
        nh, nw = new_shape
        r = min(nh / h0, nw / w0)
        hw_new = (int(round(w0 * r)), int(round(h0 * r)))
        dw, dh = nw - hw_new[0], nh - hw_new[1]
        dw /= 2
        dh /= 2

        if (w0, h0) != hw_new:
            im = cv2.resize(im, hw_new, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return im, r, dw, dh

    def _preprocess(self, img_bgr: np.ndarray):
        img, r, dw, dh = self._letterbox(img_bgr, self.imgsz)
        img = img[..., ::-1].transpose(2, 0, 1)          # BGR→RGB, HWC→CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return img[None, ...], (r, dw, dh, img_bgr.shape[:2])

    def _postprocess(
        self,
        img_bgr: np.ndarray,
        preds: np.ndarray,
        protos: np.ndarray,
        r_info,
    ) -> Optional[np.ndarray]:
        r, dw, dh, (h0, w0) = r_info

        # --- фикс: добавляем размерность batch, если её нет ----
        preds_t = torch.from_numpy(preds).float()
        if preds_t.ndim == 2:            # (N, dim)  → (1, N, dim)
            preds_t = preds_t.unsqueeze(0)

        protos_t = torch.from_numpy(protos[0]).float()   # (c, mh, mw)

        det_list = ops.non_max_suppression(
            preds_t, self.conf, self.iou, nc=self.nc
        )
        dets = det_list[0]
        if dets is None or not len(dets):
            return None

        # берём только самую уверенную детекцию
        dets = dets[dets[:, 4].argmax().unsqueeze(0)]
        dets[:, :4] = ops.scale_boxes(self.imgsz, dets[:, :4], (h0, w0))

        masks = self._mask_from_proto(
            protos_t, dets[:, 6:], dets[:, :4], (h0, w0)
        )
        return (masks[0].cpu().numpy().astype(np.uint8) * 255)  # (H,W)

    @staticmethod
    def _mask_from_proto(
        protos: torch.Tensor,
        mcoef: torch.Tensor,
        bboxes: torch.Tensor,
        img_shape: Tuple[int, int],
    ) -> torch.Tensor:
        c, mh, mw = protos.shape
        masks = (mcoef @ protos.reshape(c, -1)).reshape(-1, mh, mw)
        masks = ops.scale_masks(masks[None], img_shape)[0]
        masks = ops.crop_mask(masks, bboxes)
        return masks.gt_(0.0)

    def _draw_overlay(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        color = np.array(self.overlay_color, dtype=np.uint8)
        overlay = img.copy()
        overlay[mask == 255] = 0.4 * overlay[mask == 255] + 0.6 * color
        return overlay.astype(np.uint8)
