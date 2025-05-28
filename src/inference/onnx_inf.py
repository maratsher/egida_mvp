#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Универсальный инференс YOLOv8-Seg (ONNX Runtime) для:
  • профиля  (combine="best")  – одна лучшая маска
  • дефектов (combine="union") – объединение всех масок > conf
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, Literal

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics.utils import ops


class Yolov8SegONNX:
    # ----------------------------- init ---------------------------------
    def __init__(self, cfg: dict):
        # общий config + под-модель
        self.model_path: str | Path = cfg["onnx_model"]
        self.imgsz          = (cfg.get("imgsz", 640),) * 2 if isinstance(cfg.get("imgsz", 640), int) else tuple(cfg["imgsz"])
        self.conf: float    = cfg.get("conf_thres", 0.25)
        self.iou: float     = cfg.get("iou_thres", 0.7)
        self.nc: int        = cfg.get("num_classes", 1)
        self.combine: Literal["best", "union"] = cfg.get("combine", "best")  # best | union
        self.device: int | None = cfg.get("device", 0)
        self.overlay_color = tuple(cfg.get("overlay_color", [0, 255, 0]))  # BGR

        providers, provider_opts = (["CPUExecutionProvider"], None)
        if torch.cuda.is_available():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_opts = [{"device_id": str(self.device)}, {}]

        self.session = ort.InferenceSession(
            str(self.model_path), providers=providers, provider_options=provider_opts
        )
        self.input_name = self.session.get_inputs()[0].name

    # ----------------------------- call ---------------------------------
    def __call__(self, img_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        prep, r_info = self._preprocess(img_bgr)
        preds, protos = self.session.run(None, {self.input_name: prep})
        mask = self._postprocess(img_bgr, preds, protos, r_info)
        if mask is None:
            return None, None
        overlay = self._draw_overlay(img_bgr, mask)
        return mask, overlay

    # ------------------------- helpers ----------------------------------
    def _letterbox(self, im: np.ndarray, new_shape):
        h0, w0 = im.shape[:2]
        nh, nw = new_shape
        r = min(nh / h0, nw / w0)
        hw_new = (int(round(w0 * r)), int(round(h0 * r)))
        dw, dh = (nw - hw_new[0]) / 2, (nh - hw_new[1]) / 2
        if (w0, h0) != hw_new:
            im = cv2.resize(im, hw_new, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return im, r, dw, dh

    def _preprocess(self, img_bgr):
        img, r, dw, dh = self._letterbox(img_bgr, self.imgsz)
        img = img[..., ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return img[None], (r, dw, dh, img_bgr.shape[:2])

    # ------------------------- postprocess ------------------------------
    def _postprocess(self, img_bgr, preds, protos, r_info) -> Optional[np.ndarray]:
        r, dw, dh, (h0, w0) = r_info
        preds_t = torch.tensor(preds, dtype=torch.float32)
        if preds_t.ndim == 2:
            preds_t = preds_t.unsqueeze(0)

        protos_t = torch.tensor(protos[0], dtype=torch.float32)  # (c,mh,mw)
        dets = ops.non_max_suppression(preds_t, self.conf, self.iou, nc=self.nc)[0]
        if dets is None or not len(dets):
            return None

        # ----- choose detections ---------------------------------------
        if self.combine == "best":
            dets = dets[dets[:, 4].argmax().unsqueeze(0)]

        dets[:, :4] = ops.scale_boxes(self.imgsz, dets[:, :4], (h0, w0))
        masks = self._mask_from_proto(protos_t, dets[:, 6:], dets[:, :4], (h0, w0))

        if self.combine == "best":
            mask = masks[0]
        else:  # union
            mask = masks.any(dim=0)

        return (mask.cpu().numpy().astype(np.uint8) * 255)

    @staticmethod
    def _mask_from_proto(protos, mcoef, bboxes, img_shape):
        c, mh, mw = protos.shape
        mcoef = mcoef[:, :c]        # подрез — устраняет несоответствие 33×32
        masks = (mcoef @ protos.reshape(c, -1)).reshape(-1, mh, mw)
        masks = ops.scale_masks(masks[None], img_shape)[0]
        masks = ops.crop_mask(masks, bboxes)
        return masks.gt_(0.0)

    # ------------------------- overlay ----------------------------------
    def _draw_overlay(self, img, mask):
        overlay = img.copy()
        col = np.array(self.overlay_color, dtype=np.uint8)
        overlay[mask == 255] = 0.4 * overlay[mask == 255] + 0.6 * col
        return overlay.astype(np.uint8)
