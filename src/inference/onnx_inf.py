# src/inference/onnx_inf.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Универсальный инференс YOLOv8-Seg (ONNX Runtime).

Поддерживает два режима:
  • combine="best"   – возвращается одна маска с максимальным confidence
    (используем для сегментации профиля).
  • combine="all"    – возвращаются ВСЕ маски, прошедшие NMS-порог;
    вместе с классами и «склеенной» union-маской (для дефектов).

call() → tuple:
    mask_union : np.ndarray | None      # uint8 (H×W) 0/255   (best → одиночная)
    overlay    : np.ndarray | None      # BGR-картинка с заливкой
    masks_all  : np.ndarray | None      # (N,H,W) uint8 0/255 (None если combine="best")
    cls_ids    : list[int]  | None      # классы длиной N      (None если combine="best")
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
    # ------------------------------------------------------------------ #
    def __init__(self, cfg: dict):
        # ── основные параметры -----------------------------------------
        self.model_path: str | Path = cfg["onnx_model"]
        imgsz = cfg.get("imgsz", 640)
        self.imgsz: Tuple[int, int] = (imgsz, imgsz) if isinstance(imgsz, int) else tuple(imgsz)
        self.conf: float = cfg.get("conf_thres", 0.25)
        self.iou:  float = cfg.get("iou_thres", 0.7)
        self.nc:   int   = cfg.get("num_classes", 1)
        self.combine: Literal["best", "all"] = cfg.get("combine", "best")

        self.overlay_color = tuple(cfg.get("overlay_color", [0, 255, 0]))  # BGR
        self.device: int | None = cfg.get("device", 0)

        # ── ORT session -------------------------------------------------
        providers, provider_opts = (["CPUExecutionProvider"], None)
        if torch.cuda.is_available():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_opts = [{"device_id": str(self.device)}, {}]

        self.session = ort.InferenceSession(
            str(self.model_path), providers=providers, provider_options=provider_opts
        )
        self.input_name = self.session.get_inputs()[0].name

    # ------------------------------------------------------------------ #
    def __call__(
        self, img_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
               Optional[np.ndarray], Optional[list[int]]]:
        prep, r_info = self._preprocess(img_bgr)
        preds, protos = self.session.run(None, {self.input_name: prep})

        mask_union, masks_all, cls_ids = self._postprocess(img_bgr, preds, protos, r_info)
        if mask_union is None:
            return None, None, None, None

        overlay = self._draw_overlay(img_bgr, mask_union)
        return mask_union, overlay, masks_all, cls_ids

    # =============================  helpers  =========================== #
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
        img = img[..., ::-1].transpose(2, 0, 1)        # BGR → RGB, HWC → CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return img[None], (r, dw, dh, img_bgr.shape[:2])

    # ------------------------------------------------------------------ #
    def _postprocess(self, img_bgr, preds, protos, r_info):
        r, dw, dh, (h0, w0) = r_info

        preds_t = torch.tensor(preds, dtype=torch.float32)
        if preds_t.ndim == 2:
            preds_t = preds_t.unsqueeze(0)             # (N,dim) → (1,N,dim)

        protos_t = torch.tensor(protos[0], dtype=torch.float32)  # (c, mh, mw)
        dets = ops.non_max_suppression(preds_t, self.conf, self.iou, nc=self.nc)[0]
        if dets is None or not len(dets):
            return None, None, None

        if self.combine == "best":                     # одна маска
            dets = dets[dets[:, 4].argmax().unsqueeze(0)]

        # масштаб bbox-ов к исходному warped-изображению
        dets[:, :4] = ops.scale_boxes(self.imgsz, dets[:, :4], (h0, w0))

        masks = self._mask_from_proto(protos_t, dets[:, 6:], dets[:, :4], (h0, w0))
        cls_ids = dets[:, 5].int().tolist()

        if self.combine == "best":
            mask_union = masks[0]
            return mask_union.cpu().numpy().astype(np.uint8) * 255, None, None

        # combine == "all" : берём union-маску + возвращаем все маски и классы
        mask_union = masks.any(dim=0)
        masks_np = (masks.cpu().numpy().astype(np.uint8) * 255)  # (N,H,W)
        return (
            mask_union.cpu().numpy().astype(np.uint8) * 255,
            masks_np,
            cls_ids,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _mask_from_proto(protos, mcoef, bboxes, img_shape):
        c, mh, mw = protos.shape
        mcoef = mcoef[:, :c]                # подрезаем, если mcoef > c
        masks = (mcoef @ protos.reshape(c, -1)).reshape(-1, mh, mw)
        masks = ops.scale_masks(masks[None], img_shape)[0]
        masks = ops.crop_mask(masks, bboxes)
        return masks.gt_(0.0)

    # ------------------------------------------------------------------ #
    def _draw_overlay(self, img, mask_union):
        overlay = img.copy()
        col = np.array(self.overlay_color, dtype=np.uint8)
        overlay[mask_union == 255] = 0.4 * overlay[mask_union == 255] + 0.6 * col
        return overlay.astype(np.uint8)
