#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Профиль-метрики (1-й / 2-й сорт) с русскими подписями,
центрированными внутри детали.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, Dict
from PIL import ImageFont

from src.utils import draw_text_ru   # общий helper для UTF-8 текста


class ProfileMetrics:
    def __init__(
        self,
        width_threshold: float = 0.8,
        font_scale: float = 1.2,                     # ≈ 28 px
        thickness: int = 2,
        color_first: Tuple[int, int, int] = (0, 255, 0),
        color_second: Tuple[int, int, int] = (0, 165, 255),
        font_path: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        self.T = width_threshold
        self.fs = font_scale
        self.th = thickness
        self.c1 = tuple(int(x) for x in color_first)
        self.c2 = tuple(int(x) for x in color_second)

        self.font_size = int(24 * font_scale)
        try:
            self.pil_font = ImageFont.truetype(font_path, size=self.font_size)
        except OSError:
            self.pil_font = ImageFont.load_default()

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

        boundary_row = next((i for i, s in enumerate(row_sum) if s >= self.T * Wmax), 0)

        # ----------- маски 1-го / 2-го сорта ----------------------------
        mask_second = np.zeros_like(mask)
        mask_second[:boundary_row, :] = mask[:boundary_row, :]
        mask_first = mask.copy()
        mask_first[:boundary_row, :] = 0

        # ------------------- геометрия ----------------------------------
        x, y, w_box, h_box = cv2.boundingRect(mask)
        width_cm  = w_box / scale
        height_cm = h_box / scale

        area_px_total   = mask.sum()          // 255
        area_px_first   = mask_first.sum()    // 255
        area_px_second  = mask_second.sum()   // 255

        area_m2_total   = area_px_total  / (scale * scale) / 10_000
        area_m2_first   = area_px_first  / (scale * scale) / 10_000
        area_m2_second  = area_px_second / (scale * scale) / 10_000

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

        # ---------------- визуализация ----------------------------------
        alpha = 0.45
        out = overlay.copy()
        out[mask_first == 255]  = (1 - alpha) * out[mask_first == 255]  + alpha * np.array(self.c1, np.uint8)
        out[mask_second == 255] = (1 - alpha) * out[mask_second == 255] + alpha * np.array(self.c2, np.uint8)
        cv2.line(out, (0, boundary_row), (W - 1, boundary_row), (255, 0, 0), 2)

        # ------------- центр подписи ------------------------------------
        cx = x + w_box // 2
        cy = y + h_box // 2

        line1 = f"Ширина: {metrics['width_cm']} см   Высота: {metrics['height_cm']} см   Площадь: {metrics['area_total_m2']} м²"
        line2 = f"1-й сорт: {metrics['first_pct']} %   2-й сорт: {metrics['second_pct']} %"

        # размеры текста (по PIL)
        w1, h1 = self.pil_font.getbbox(line1)[2:]
        w2, h2 = self.pil_font.getbbox(line2)[2:]

        y1 = int(cy - h2 * 0.8)
        y2 = int(cy + h1 * 1.2)
        x1 = int(cx - w1 / 2)
        x2 = int(cx - w2 / 2)

        x1 = max(10, min(x1, W - w1 - 10))
        x2 = max(10, min(x2, W - w2 - 10))
        y1 = max(h1 + 10, min(y1, H - 10))
        y2 = max(h2 + 10, min(y2, H - 10))

        # рисуем текст через Pillow-helper
        out = draw_text_ru(out, line1, (x1, y1 - h1), font_size=self.font_size,
                           color_bgr=(0, 0, 0), stroke_width=0)
        out = draw_text_ru(out, line2, (x2, y2 - h2), font_size=self.font_size,
                           color_bgr=(0, 0, 0), stroke_width=0)

        return metrics, out
