# src/postprocessing/defect_metrics.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Закраска дефектов + расчёт площадей.
• Площадь каждого класса → **см²**.
• Доля (%) считается по площади 1-го сорта.
Названия можно задавать короче прямо в `config.yaml`,
например «Дефект перех. блок», «Трещины».
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import cv2
import numpy as np


class DefectMetrics:
    def __init__(
        self,
        class_conf: Dict[int, dict],   # {id: {"name": str, "color": [B,G,R]}}
        alpha: float = 0.45,           # прозрачность заливки
    ):
        self.cls   = {int(k): v for k, v in class_conf.items()}
        self.alpha = alpha

    # ------------------------------------------------------------------ #
    def __call__(
        self,
        masks: np.ndarray,             # (N,H,W) uint8 0/255
        cls_ids: List[int],            # классы длиной N
        scale: float,                  # px / см
        base_overlay: np.ndarray,      # BGR-изображение
        area_first_m2: float,          # площадь 1-го сорта (м²)
    ) -> Tuple[Dict[str, float], np.ndarray]:

        overlay = base_overlay.copy()
        metrics: Dict[str, float] = {}

        if masks is None or len(masks) == 0:
            return metrics, overlay

        # перевод для процента: 1-й сорт м² → см²
        area_first_cm2 = area_first_m2 * 10_000

        for cid in sorted(set(cls_ids)):
            cls_mask = np.any(masks[np.array(cls_ids) == cid], axis=0)
            if not cls_mask.any():
                continue

            color_bgr = tuple(int(c) for c in self.cls[cid]["color"])
            overlay[cls_mask] = (
                (1 - self.alpha) * overlay[cls_mask] + self.alpha * np.array(color_bgr, np.uint8)
            )

            # площадь дефекта, см²
            pixels = float(cls_mask.sum())
            area_cm2 = pixels / (scale * scale)             # px² / (px/cm)²
            pct = round(100 * area_cm2 / area_first_cm2, 2) if area_first_cm2 > 0 else None

            name = self.cls[cid]["name"]                    # уже «короткое» имя
            metrics[f"{name}_cm2"] = round(area_cm2, 1)
            metrics[f"{name}_pct"] = pct if pct is not None else "-"

        return metrics, overlay
