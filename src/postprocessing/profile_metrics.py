# src/postprocessing/profile_metrics.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Расчёт геометрии блока.
• Ширины: W1 (низ), W3 (граница сортов), W2 (середина) – в мм.
• Высоты: H1 (по реальной маске), H2 (до границы сортов) – в мм.
• Площади и доли 1‑го/2‑го сорта (м², %).

Overlay: полупрозрачная заливка (зелёный / оранжевый) + синяя
линия границы. Текст выводится справа в правой панели.
"""
from __future__ import annotations
from typing import Tuple, Dict

import cv2
import numpy as np


class ProfileMetrics:
    def __init__(
        self,
        width_threshold: float = 0.8,                # 0…1 доли максимальной ширины
        color_first: Tuple[int, int, int] = (0, 255, 0),
        color_second: Tuple[int, int, int] = (0, 165, 255),
        alpha: float = 0.45,                         # прозрачность заливки
    ):
        self.T = width_threshold
        self.c1 = np.array(color_first,  dtype=np.uint8)
        self.c2 = np.array(color_second, dtype=np.uint8)
        self.alpha = float(alpha)

    def __call__(
        self,
        mask: np.ndarray,           # (H,W) uint8 0/255
        scale: float,               # px / см
        warped_img: np.ndarray,     # BGR
        overlay: np.ndarray | None = None,
    ) -> Tuple[Dict[str, float], np.ndarray]:

        if overlay is None:
            overlay = warped_img

        H, W = mask.shape
        # 1) горизонтальная проекция (сумма битов) для каждой строки
        row_sum = (mask // 255).sum(axis=1)
        Wmax = int(row_sum.max())
        if Wmax == 0:
            # ничего не найдено
            return {}, overlay.copy()

        # 2) находим первую строку, в которой ширина >= T * Wmax
        threshold = self.T * Wmax
        boundary_row = next(i for i, s in enumerate(row_sum) if s >= threshold)

        # 3) формируем две маски: mask_first (нижняя часть) и mask_second (верхняя часть)
        mask_first = mask.copy()
        mask_first[:boundary_row, :] = 0

        mask_second = mask.copy()
        mask_second[boundary_row:, :] = 0

        # 4) функция: ширина «маски» в строке y (в см)
        def width_at_row(y: int) -> float:
            cols = np.where(mask[y] == 255)[0]
            if len(cols) == 0:
                return 0.0
            return (cols[-1] - cols[0] + 1) / scale

        # 5) вычисляем индексы всех строк, где mask != 0
        nonzero_rows = np.where(row_sum > 0)[0]
        if len(nonzero_rows) == 0:
            # неожиданный случай: маска вдруг полностью пустая
            mask_y_top = 0
            mask_y_bottom = 0
        else:
            mask_y_top    = int(nonzero_rows.min())
            mask_y_bottom = int(nonzero_rows.max())

        # 6) вычисляем W1 (ширина «нижней» маски) как медиану последних K строк:
        K = 25
        if len(nonzero_rows) == 0:
            W1_cm = 0.0
        else:
            last_rows = nonzero_rows[-K:]
            widths_cm = [width_at_row(int(y)) for y in last_rows]
            W1_cm = float(np.median(widths_cm))
        W1_mm = round(W1_cm * 10)

        # 7) W3 = ширина именно в boundary_row
        W3_cm = width_at_row(boundary_row)
        W3_mm = round(W3_cm * 10)

        # 8) W2 = ширина в средней строке между boundary и mask_y_bottom
        mid_row = (boundary_row + mask_y_bottom) // 2
        W2_cm = width_at_row(mid_row)
        W2_mm = round(W2_cm * 10)

        # 9) Фактическая высота блока (H1): высота непрерывного диапазона непустых рядов
        h_pixels = mask_y_bottom - mask_y_top + 1
        H1_cm = h_pixels / scale
        H1_mm = round(H1_cm * 10)

        # 10) H2: высота от boundary_row до «низа» маски (mask_y_bottom)
        h2_pixels = mask_y_bottom - boundary_row + 1
        H2_cm = h2_pixels / scale
        H2_mm = round(H2_cm * 10)

        # 11) площади всех пикселей:
        px_total  = (mask // 255).sum()
        px_first  = (mask_first // 255).sum()
        px_second = (mask_second // 255).sum()

        m2_total   = px_total   / (scale * scale) / 10_000
        m2_first   = px_first   / (scale * scale) / 10_000
        m2_second  = px_second  / (scale * scale) / 10_000

        pct_first  = round(100 * m2_first  / m2_total, 1) if m2_total else 0.0
        pct_second = round(100 * m2_second / m2_total, 1) if m2_total else 0.0

        metrics: Dict[str, float] = {
            "area_total_m2":  round(m2_total,  3),
            "area_first_m2":  round(m2_first,  3),
            "area_second_m2": round(m2_second, 3),
            "first_pct":      pct_first,
            "second_pct":     pct_second,
            "W1_mm": W1_mm,
            "W2_mm": W2_mm,
            "W3_mm": W3_mm,
            "H1_mm": H1_mm,
            "H2_mm": H2_mm,
        }

        # 12) визуальная заливка (профиль)
        out = overlay.astype(np.float32)
        mask1_bool = (mask_first == 255)
        mask2_bool = (mask_second == 255)

        if mask1_bool.any():
            out[mask1_bool] = (1 - self.alpha) * out[mask1_bool] + self.alpha * self.c1.astype(np.float32)
        if mask2_bool.any():
            out[mask2_bool] = (1 - self.alpha) * out[mask2_bool] + self.alpha * self.c2.astype(np.float32)

        # синяя линия границы (на y = boundary_row)
        cv2.line(out, (0, boundary_row), (W - 1, boundary_row), (255, 0, 0), 2)

        final_overlay = out.round().astype(np.uint8)
        return metrics, final_overlay
