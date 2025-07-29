# src/service.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сервис: .pic → JPEG → пред- и пост-обработка, единый overlay с
правой панелью. Все метрики (W1–W3, H1–H2, площади, проценты дефектов)
выводятся только в панели; сам торец блока не перегружен подписью.
"""
from __future__ import annotations
import json, logging, mmap, time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import yaml
import numpy as np
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.preprocessing import DistortionCorrector, PerspectiveTransformer
from src.inference import Yolov8SegONNX
from src.postprocessing.profile_metrics import ProfileMetrics
from src.postprocessing.defect_metrics import DefectMetrics
#from src.utils.panel import draw_panel
from src.utils.panel import _prep_lines


# --------------------------------------------------------------------------- #
class PipelineHandler(FileSystemEventHandler):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        self.input_dir  = Path(cfg["input_dir"])
        self.output_dir = Path(cfg.get("output_dir", "./results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── pre-processing ──────────────────────────────────────────────
        self.dist_corr = DistortionCorrector(cfg)
        self.persp     = PerspectiveTransformer(cfg)

        # ── ONNX-модели ─────────────────────────────────────────────────
        self.seg_profile = Yolov8SegONNX({**cfg, **cfg["models"]["profile"]})
        self.seg_defect  = Yolov8SegONNX({**cfg, **cfg["models"]["defect"]})

        # ── пост-процессинг ─────────────────────────────────────────────
        self.prof_post = ProfileMetrics(
            width_threshold = cfg.get("width_threshold", 0.8),
            color_first     = tuple(cfg.get("color_first",  [0, 255, 0])),
            color_second    = tuple(cfg.get("color_second", [0, 165, 255])),
            alpha           = cfg.get("profile_alpha", 0.45),
        )
        self.def_post = DefectMetrics(
            cfg["defect_classes"],
            alpha = cfg.get("defect_alpha", 0.45),
        )

        # параметры панели
        # self.panel_font_size = int(cfg.get("panel_font_size", 36))
        # self.panel_width_px  = int(cfg.get("panel_width_px", 500))

    # ------------------------ helpers ----------------------------------
    def _extract_last_jpeg(self, pic_path: Path) -> Optional[Path]:
        with open(pic_path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            end = mm.rfind(b"\xFF\xD9")
            start = mm.rfind(b"\xFF\xD8", 0, end)
            if end == -1 or start == -1:
                return None
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            out_jpg = self.output_dir / f"{pic_path.stem}_{ts}.jpg"
            with open(out_jpg, "wb") as o:
                o.write(mm[start:end + 2])
            return out_jpg

    # production-callback (.pic закрыт) ---------------------------------
    def on_closed(self, event):
        src = Path(event.src_path)
        if src.suffix.lower() != ".pic":
            return

        # ждём остановки роста файла
        prev = -1
        while src.stat().st_size != prev:
            prev = src.stat().st_size
            time.sleep(0.1)

        jpg = self._extract_last_jpeg(src)
        if jpg:
            logging.info(f"Extracted JPEG: {jpg.name}")
            self.process_jpg(jpg)

    # -------------------- основной pipeline ---------------------------
    def process_jpg(self, jpg_path: Path):
        try:
            stem = jpg_path.stem
            out_dir = self.output_dir / stem
            out_dir.mkdir(parents=True, exist_ok=True)

            # 1) чтение + undistort
            frame = cv2.imread(str(jpg_path))
            undist = self.dist_corr.correct(frame)

            # 2) перспектива
            warped, scale = self.persp.transform(undist)
            cv2.imwrite(str(out_dir / "warped.png"), warped)

            # 3) сегментация профиля
            #    ▶️ ov_prof игнорируем, вернём overlay чисто из ProfileMetrics
            prof_mask, _, _, _ = self.seg_profile(warped)
            if prof_mask is None:
                logging.warning(f"No profile mask for {jpg_path.name}")
                return

            # 4) сегментация дефектов
            def_mask, _, masks_nd, cls_ids = self.seg_defect(warped)
            # def_mask может быть None если дефектов нет

            # 5) пост-процессинг профиля
            #    ▶️ НЕ передаём в overlay предыдущий ov_prof, ProfileMetrics сам
            #       закрасит профиль поверх оригинала
            prof_metrics, ov_prof = self.prof_post(prof_mask, scale, warped)

            # 6) пост-процессинг дефектов (накладываем поверх ov_prof)
            def_metrics, ov_with_defects = {}, None
            if def_mask is not None and masks_nd is not None:
                # Передаём base_overlay=ov_prof ⇒ дефекты рисуются над профилем
                def_metrics, ov_with_defects = self.def_post(
                    masks_nd,
                    cls_ids,
                    scale,
                    ov_prof,
                    prof_metrics["area_first_m2"],
                )

            # 7) финальный combined-overlay:
            #    если есть дефекты – берём ov_with_defects, иначе – ov_prof
            combined = ov_with_defects if ov_with_defects is not None else ov_prof
            # гарантируем uint8
            combined = combined.round().astype(np.uint8)

            # 8) рисуем правую панель с метриками
            # final = draw_panel(
            #     combined,
            #     prof_metrics,
            #     def_metrics,
            #     font_size = self.panel_font_size,
            #     col_w     = self.panel_width_px,
            # )
            # 8) сохраняем оверлей без панели
            final = combined        
        
            # 9) сохраняем
            txt_lines = _prep_lines(prof_metrics, def_metrics)
            with open(out_dir / "metrics.txt", "w", encoding="utf-8") as tf:
                tf.write("\n".join(txt_lines))
                
            cv2.imwrite(str(out_dir / "overlay_final.jpg"), final)
            cv2.imwrite(str(out_dir / "mask_profile.png"), prof_mask)
            if def_mask is not None:
                cv2.imwrite(str(out_dir / "mask_defects.png"), def_mask)

            with open(out_dir / "metrics.json", "w", encoding="utf-8") as jf:
                json.dump({**prof_metrics, **def_metrics}, jf, ensure_ascii=False, indent=2)

            logging.info(f"{stem}: {prof_metrics | def_metrics}")

        except Exception as e:
            logging.exception(f"Error processing {jpg_path.name}: {e}")


# --------------------------------------------------------------------------- #
def main():
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s %(levelname)s: %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    cfg = yaml.safe_load(open("config.yaml", "r"))
    handler = PipelineHandler(cfg)

    # ---------- test-mode ----------------------------------------------
    if cfg.get("test_mode", False):
        tdir = Path(cfg["test_images_dir"])
        logging.info(f"Test mode ON — processing images from {tdir}")
        for img in sorted(tdir.iterdir()):
            if img.suffix.lower() in (".jpg", ".png"):
                handler.process_jpg(img)
        return

    # ---------- production ---------------------------------------------
    observer = Observer()
    observer.schedule(handler, path=str(Path(cfg["input_dir"]).resolve()), recursive=True)
    observer.start()
    logging.info(f"Watching {cfg['input_dir']} for .pic files…  Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
