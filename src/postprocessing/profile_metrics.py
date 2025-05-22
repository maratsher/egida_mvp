#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Подсчёт габаритов профиля по бинарной маске.
Возвращает словарь метрик и изображение с наложенным текстом.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, Dict


class ProfileMetrics:
    """
    :param font_scale: масштаб шрифта для cv2.putText
    :param thickness:  толщину контура текста
    :param color:      BGR-цвет текста и рамки
    """

    def __init__(
        self,
        font_scale: float = 1.2,
        thickness: int = 2,
        color: Tuple[int, int, int] = (255, 255, 255),
    ):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fs = font_scale
        self.th = thickness
        self.col = color

    # ------------------------------------------------------------------ #
    #  Основной вызов                                                    #
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        mask: np.ndarray,
        scale: float,
        warped_img: np.ndarray,
        overlay: np.ndarray | None = None,
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """
        :param mask:       бинарная маска (uint8 0/255) того же размера, что warped_img
        :param scale:      px / реальная_единица (см).  -> 1 см = scale px
        :param warped_img: BGR-изображение после перспективного преобразования
        :param overlay:    BGR с полупрозрачной маской; если None – скопируем warped_img
        """
        if overlay is None:
            overlay = warped_img.copy()

        # --- 1. контуры и bounding box ----------------------------------
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return {"width_cm": 0, "height_cm": 0, "area_cm2": 0}, overlay

        cnt = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)          # axis-aligned bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), self.col, 2)

        # --- 2. метрики --------------------------------------------------
        width_cm = w / scale          # см
        height_cm = h / scale         # см
        area_px = cv2.countNonZero(mask)
        area_cm2 = area_px / (scale * scale)
        area_m2 = round(area_cm2 / 10_000, 3)

        metrics = {
            "width_cm": round(width_cm, 1),
            "height_cm": round(height_cm, 1),
            "area_m2": round(area_m2, 2),
        }

        # --- 3. подписи --------------------------------------------------
        txt = f"W={metrics['width_cm']} cm  H={metrics['height_cm']} cm  S={metrics['area_m2']} m²"
        cv2.putText(
            overlay,
            txt,
            (20, 50),
            self.font,
            self.fs,
            self.col,
            self.th,
            cv2.LINE_AA,
        )

        return metrics, overlay
