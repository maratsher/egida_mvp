# src/utils/panel.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Формирование правой белой панели с метриками.

Использует draw_text_ru() → корректное отображение кириллицы.
"""
from __future__ import annotations
from typing import Dict, Any, Tuple, List

import cv2
import numpy as np
from PIL import ImageFont

from .view import draw_text_ru                       # helper


# путь к TTF-шрифту (DejaVu Sans — есть в докер-образе)
FONT_TTF = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _prep_lines(prof: Dict[str, Any], defect: Dict[str, Any]) -> List[str]:
    """Готовит список строк для вывода в панель (без пустых)."""
    lines: List[str] = [
        f"W1  = {prof['W1_mm']:.0f} мм",
        f"W2  = {prof['W2_mm']:.0f} мм",
        f"W3  = {prof['W3_mm']:.0f} мм",
        "",
        f"H1  = {prof['H1_mm']:.0f} мм",
        f"H2  = {prof['H2_mm']:.0f} мм",
        "",
        f"S = {prof['area_total_m2']:.2f} м²",
        "",
        f"1-й сорт = {prof['first_pct']} %",
        f"3-й сорт = {prof['second_pct']} %",
        "",
    ]

    # --- дефекты -------------------------------------------------------
    for k in defect:
        if not k.endswith("_pct"):
            continue
        name = k[:-4]                          # убираем суффикс "_pct"
        pct  = defect[k]
        cm2  = defect.get(f"{name}_cm2", 0)
        lines.append(f"{name}: {cm2:.0f} см² ({pct} %)")

    # убираем возможный последний пустой элемент
    while lines and lines[-1] == "":
        lines.pop()

    return lines


# --------------------------------------------------------------------- #
def draw_panel(
    img_bgr: np.ndarray,
    prof_metrics: Dict[str, Any],
    defect_metrics: Dict[str, Any],
    font_size: int = 42,
    col_w: int = 500,
) -> np.ndarray:
    """
    Добавляет к изображению белую колонку справа и выводит все метрики.

    Parameters
    ----------
    img_bgr : np.ndarray
        Исходное BGR-изображение (уже с заливкой масок).
    prof_metrics : dict
        Результат ProfileMetrics (W1-3 мм, H1-2 мм, проценты сортов …).
    defect_metrics : dict
        Результат DefectMetrics (…_cm2, …_pct).
    font_size : int
        Размер шрифта в px (по TTF навыно).
    col_w : int
        Ширина правой панели, px.

    Returns
    -------
    np.ndarray
        Новое изображение BGR с панелью.
    """
    h, w = img_bgr.shape[:2]
    out = cv2.copyMakeBorder(
        img_bgr, 0, 0, 0, col_w, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )

    # шрифт Pillow
    try:
        font = ImageFont.truetype(FONT_TTF, size=font_size)
    except OSError:
        font = ImageFont.load_default()

    # подготовка строк
    lines = _prep_lines(prof_metrics, defect_metrics)

    # координаты вывода
    x0, y = w + 20, 30
    pad_y = int(font_size * 0.3)

    # вывод строк
    for text in lines:
        if text == "":
            y += pad_y
            continue

        out = draw_text_ru(
            out,
            text,
            (x0, y),
            font_size=font_size,
            color_bgr=(0, 0, 0),
            stroke_width=0,
            font_path=FONT_TTF,
        )
        y += font_size + pad_y

    return out
