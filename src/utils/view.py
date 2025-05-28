from __future__ import annotations
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ──────────────────────────────────────────────────────────────────────────
_DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def draw_text_ru(
    img_bgr: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_size: int = 24,
    color_bgr: Tuple[int, int, int] = (255, 255, 255),
    stroke_width: int = 1,
    stroke_color_bgr: Tuple[int, int, int] = (0, 0, 0),
    font_path: str | Path = _DEFAULT_FONT,
) -> np.ndarray:
    """
    Рисует UTF-8 текст (включая кириллицу) на cv2-изображении BGR.

    Returns
    -------
    np.ndarray
        Изображение BGR с наложенным текстом.
    """
    # cv2 BGR → PIL RGB
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except OSError:
        font = ImageFont.load_default()

    # Pillow работает в RGB
    draw.text(
        org,
        text,
        font=font,
        fill=tuple(reversed(color_bgr)),
        stroke_width=stroke_width,
        stroke_fill=tuple(reversed(stroke_color_bgr)),
    )

    # PIL → cv2 обратно
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)