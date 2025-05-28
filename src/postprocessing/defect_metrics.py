#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Подсчёт площадей дефектов по классам + overlay с легендой.
"""
from __future__ import annotations
import cv2, numpy as np
from typing import Dict, List, Tuple

class DefectMetrics:
    def __init__(self, class_conf: Dict[int, dict], font_scale=1.0, thickness=2):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fs   = font_scale
        self.th   = thickness
        # class_conf: {id: {"name": str, "color": [B,G,R]}}
        self.cls  = {int(k): v for k, v in class_conf.items()}

    # ------------------------------------------------------------------
    def __call__(
        self,
        masks: np.ndarray,          # (N, H, W) uint8 0/255
        cls_ids: List[int],         # len N
        scale: float,
        base_overlay: np.ndarray,
        area_first_m2: float,
    ):
        H, W = base_overlay.shape[:2]
        overlay = base_overlay.copy()
        metrics = {}

        # суммируем по классам
        uniq = sorted(set(cls_ids))
        alpha = 0.45
        legend_y = 25

        for cid in uniq:
            cls_mask = np.any(masks[np.array(cls_ids) == cid], axis=0)  # (H,W) bool
            if not cls_mask.any():
                continue
            color = np.array(self.cls[cid]["color"], dtype=np.uint8)
            overlay[cls_mask] = (1 - alpha) * overlay[cls_mask] + alpha * color

            # площадь
            pixels = int(cls_mask.sum())
            area_cm2 = pixels / (scale * scale)
            area_m2  = area_cm2 / 10_000
            pct      = round(100 * area_m2 / area_first_m2, 2) if area_first_m2 else 0

            name = self.cls[cid]["name"]
            metrics[f"{name}_m2"]  = round(area_m2, 3)
            metrics[f"{name}_pct"] = pct

            # легенда
            cv2.rectangle(overlay, (25, legend_y - 15), (45, legend_y + 5), color.tolist(), -1)
            text = f"{name}: {pct}% ({area_m2:.3f} м²)"
            cv2.putText(overlay, text, (55, legend_y), self.font, self.fs,
                        (0, 0, 0), self.th, cv2.LINE_AA)
            legend_y += 35

        return metrics, overlay
