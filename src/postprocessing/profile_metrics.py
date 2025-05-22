#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Подсчёт метрик профиля + визуализация (1-й / 2-й сорт).
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
        self.c1 = tuple(int(x) for x in color_first)     # 1-й сорт
        self.c2 = tuple(int(x) for x in color_second)    # 2-й сорт

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

        # ---------- 1. горизонтальная проекция -------------------------
        row_sum = mask.sum(axis=1) // 255
        Wmax = row_sum.max()
        if Wmax == 0:
            return {}, overlay

        # ---------- 2. поиск границы 1-го сорта ------------------------
        boundary_row = next(
            (i for i, s in enumerate(row_sum) if s >= self.T * Wmax), 0
        )

        # ---------- 3. делим маску ------------------------------------
        mask_second = np.zeros_like(mask)
        mask_second[:boundary_row, :] = mask[:boundary_row, :]
        mask_first = mask.copy()
        mask_first[:boundary_row, :] = 0

        # ---------- 4. bounding-box всего профиля ---------------------
        x, y, w_box, h_box = cv2.boundingRect(mask)
        width_cm  = w_box / scale
        height_cm = h_box / scale

        # ---------- 5. площади ----------------------------------------
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

        # ---------- 6. визуализация -----------------------------------
        alpha = 0.45
        out = overlay.copy()

        # заливка 1-го сорта
        c1_arr = np.array(self.c1, dtype=np.uint8)
        out[mask_first == 255] = (
            (1 - alpha) * out[mask_first == 255] + alpha * c1_arr
        )

        # заливка 2-го сорта
        c2_arr = np.array(self.c2, dtype=np.uint8)
        out[mask_second == 255] = (
            (1 - alpha) * out[mask_second == 255] + alpha * c2_arr
        )

        # линия границы
        #cv2.line(out, (0, boundary_row), (W - 1, boundary_row), (255, 0, 0), 2)

        # ---------- 7. подписи ----------------------------------------
        line1 = (
            f"W={metrics['width_cm']}cm  "
            f"H={metrics['height_cm']}cm  "
            f"S={metrics['area_total_m2']}m2"
        )
        line2 = (
            f"1_sort: {metrics['first_pct']}%   "
            f"2_sort: {metrics['second_pct']}%"
        )

        cv2.putText(out, line1, (20, 50), self.font, self.fs, (0, 0, 0), self.th, cv2.LINE_AA)
        cv2.putText(out, line2, (20, 95), self.font, self.fs, (0, 0, 0), self.th, cv2.LINE_AA)

        return metrics, out
