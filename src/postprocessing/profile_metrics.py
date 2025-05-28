#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Подсчёт метрик профиля + визуализация (1-й / 2-й сорт)
с центрированной подписью внутри детали.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, Dict


class ProfileMetrics:
    def __init__(
        self,
        width_threshold: float = 0.8,
        font_scale: float = 1.2,
        thickness: int = 2,
        color_first: Tuple[int, int, int] = (0, 255, 0),
        color_second: Tuple[int, int, int] = (0, 165, 255),
    ):
        self.T = width_threshold
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fs = font_scale
        self.th = thickness
        self.c1 = tuple(int(x) for x in color_first)
        self.c2 = tuple(int(x) for x in color_second)

    # ------------------------------------------------------------------ #
    def __call__(
        self,
        mask: np.ndarray,
        scale: float,
        warped_img: np.ndarray,
        overlay: np.ndarray | None = None,
    ) -> Tuple[Dict[str, float], np.ndarray]:
        if overlay is None:
            overlay = warped_img.copy()

        H, W = mask.shape
        row_sum = mask.sum(axis=1) // 255
        Wmax = row_sum.max()
        if Wmax == 0:
            return {}, overlay

        boundary_row = next(
            (i for i, s in enumerate(row_sum) if s >= self.T * Wmax), 0
        )

        mask_second = np.zeros_like(mask)
        mask_second[:boundary_row, :] = mask[:boundary_row, :]
        mask_first = mask.copy()
        mask_first[:boundary_row, :] = 0

        # -------- метрики -------------------------------------------------
        x, y, w_box, h_box = cv2.boundingRect(mask)
        width_cm  = w_box / scale
        height_cm = h_box / scale

        area_px_total   = mask.sum()          // 255
        area_px_first   = mask_first.sum()    // 255
        area_px_second  = mask_second.sum()   // 255

        area_cm2_total  = area_px_total  / (scale * scale)
        area_cm2_first  = area_px_first  / (scale * scale)
        area_cm2_second = area_px_second / (scale * scale)

        area_m2_total   = area_cm2_total  / 10_000
        area_m2_first   = area_cm2_first  / 10_000
        area_m2_second  = area_cm2_second / 10_000

        first_pct  = round(100 * area_m2_first  / area_m2_total, 1) if area_m2_total else 0
        second_pct = round(100 * area_m2_second / area_m2_total, 1) if area_m2_total else 0

        metrics = {
            "width_cm":       round(width_cm,  1),
            "height_cm":      round(height_cm, 1),
            "area_total_m2":  round(area_m2_total,  3),
            "area_first_m2":  round(area_m2_first,  3),
            "area_second_m2": round(area_m2_second, 3),
            "first_pct":      first_pct,
            "second_pct":     second_pct,
        }

        # -------- визуализация -------------------------------------------
        alpha = 0.45
        out = overlay.copy()

        c1_arr = np.array(self.c1, dtype=np.uint8)
        out[mask_first == 255] = (
            (1 - alpha) * out[mask_first == 255] + alpha * c1_arr
        )

        c2_arr = np.array(self.c2, dtype=np.uint8)
        out[mask_second == 255] = (
            (1 - alpha) * out[mask_second == 255] + alpha * c2_arr
        )

        cv2.line(out, (0, boundary_row), (W - 1, boundary_row), (255, 0, 0), 2)

        # -------- центр подписи ------------------------------------------
        cx = x + w_box // 2
        cy = y + h_box // 2

        line1 = f"W={metrics['width_cm']}cm  H={metrics['height_cm']}cm  S={metrics['area_total_m2']}m2"
        line2 = f"1_sort: {metrics['first_pct']}%   2_sort: {metrics['second_pct']}%"

        # размеры текста
        (w1, h1), base1 = cv2.getTextSize(line1, self.font, self.fs, self.th)
        (w2, h2), base2 = cv2.getTextSize(line2, self.font, self.fs, self.th)

        # координаты: центрируем по X, делаем небольшой вертикальный сдвиг
        y1 = int(cy - h2 * 0.8)          # чуть выше центра
        y2 = int(cy + h1 * 1.2)          # чуть ниже центра

        x1 = int(cx - w1 / 2)
        x2 = int(cx - w2 / 2)

        # не даём выйти за границы изображения
        x1 = max(10, min(x1, W - w1 - 10))
        x2 = max(10, min(x2, W - w2 - 10))
        y1 = max(h1 + 10, min(y1, H - 10))
        y2 = max(h2 + 10, min(y2, H - 10))

        cv2.putText(out, line1, (x1, y1), self.font, self.fs, (0, 0, 0), self.th, cv2.LINE_AA)
        cv2.putText(out, line2, (x2, y2), self.font, self.fs, (0, 0, 0), self.th, cv2.LINE_AA)

        return metrics, out
