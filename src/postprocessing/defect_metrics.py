#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Подсчёт площадей дефектов + overlay-легенда.
"""
from __future__ import annotations
import cv2, numpy as np
from typing import Dict, List, Tuple
from PIL import ImageFont
from src.utils import draw_text_ru


class DefectMetrics:
    def __init__(
        self,
        class_conf: Dict[int, dict],
        font_scale: float = 1.2,
        thickness: int = 2,
        font_path: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        self.cls = {int(k): v for k, v in class_conf.items()}
        self.alpha = 0.45
        self.th = thickness
        self.font_size = int(24 * font_scale)
        try:
            self.pil_font = ImageFont.truetype(font_path, size=self.font_size)
        except OSError:
            self.pil_font = ImageFont.load_default()

    # ------------------------------------------------------------------ #
    def __call__(
        self,
        masks: np.ndarray,           # (N,H,W) uint8 0/255
        cls_ids: List[int],
        scale: float,
        base_overlay: np.ndarray,
        area_first_m2: float,        # ← площадь «1-го сорта» из ProfileMetrics
    ) -> Tuple[Dict[str, float], np.ndarray]:

        overlay = base_overlay.copy()
        metrics: Dict[str, float] = {}
        legend_items: list[Tuple[str, str, Tuple[int, int, int]]] = []
        # формат: (подпись, area_m2_str, цвет)

        for cid in sorted(set(cls_ids)):
            cls_mask = np.any(masks[np.array(cls_ids) == cid], axis=0)
            if not cls_mask.any():
                continue

            color_bgr = np.array(self.cls[cid]["color"], dtype=np.uint8)
            overlay[cls_mask] = (
                (1 - self.alpha) * overlay[cls_mask] + self.alpha * color_bgr
            )

            pixels = cls_mask.astype(np.float64).sum()
            area_m2 = pixels / (scale * scale) / 10_000          # px² → м²
            pct = None
            if area_first_m2 > 0:
                pct = round(100 * area_m2 / area_first_m2, 2)

            name = self.cls[cid]["name"]
            metrics[f"{name}_m2"] = round(area_m2, 3)
            metrics[f"{name}_pct"] = pct if pct is not None else "-"

            pct_text = f"{pct}%" if pct is not None else "—"
            legend_items.append(
                (f"{name}: {pct_text} ({area_m2:.3f} м²)", color_bgr)
            )

        # ---------- карточки-легенды -----------------------------------
        y, pad, sq = 60, 6, 20
        for text, color_bgr in legend_items:
            tw, th = self.pil_font.getbbox(text)[2:]
            card_h = max(th, sq) + pad * 2
            card_w = sq + pad * 3 + tw

            cv2.rectangle(overlay, (25, y), (25 + card_w, y + card_h), (255, 255, 255), -1)
            cv2.rectangle(
                overlay,
                (25 + pad, y + (card_h - sq) // 2),
                (25 + pad + sq, y + (card_h - sq) // 2 + sq),
                tuple(map(int, color_bgr)),      # ← было: color_bgr
                -1,
            )
            overlay = draw_text_ru(
                overlay,
                text,
                (25 + pad * 2 + sq, y + (card_h - th) // 2),
                font_size=self.font_size,
                color_bgr=(0, 0, 0),
                stroke_width=0,
            )
            y += card_h + 10

        return metrics, overlay
