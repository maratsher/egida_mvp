#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сервис: слушает .pic, делает две сегментации (профиль + дефекты) и
сохраняет результаты. Метрики профиля остаются как раньше.
"""
import time, json, yaml, logging, mmap
from pathlib import Path
from datetime import datetime
from typing import Optional

import cv2
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from src.preprocessing import DistortionCorrector, PerspectiveTransformer
from src.inference import Yolov8SegONNX
from src.postprocessing import ProfileMetrics, DefectMetrics


# --------------------------------------------------------------------------- #
class PipelineHandler(FileSystemEventHandler):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        mdl_prof = cfg["models"]["profile"]
        mdl_def  = cfg["models"]["defect"]
        self.input_dir  = Path(cfg["input_dir"])
        self.output_dir = Path(cfg.get("output_dir", "./results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── стадии ───────────────────────────────────────────────────────
        self.dist_corr = DistortionCorrector(cfg)
        self.persp     = PerspectiveTransformer(cfg)

        # две модели YOLOv8-Seg
        self.seg_profile = Yolov8SegONNX({**cfg, **mdl_prof})
        self.seg_defect  = Yolov8SegONNX({**cfg, **mdl_def})

        # пост-процессинг профиля
        self.post = ProfileMetrics(
            width_threshold=cfg.get("width_threshold", 0.8),
            font_scale     =cfg.get("font_scale",     1.2),
            thickness      =cfg.get("font_thickness", 2),
            color_first    =tuple(cfg.get("color_first",  [0, 255, 0])),
            color_second   =tuple(cfg.get("color_second", [0, 165, 255])),
        )

        # пост процессинг дефектов
        def_conf = cfg["defect_classes"]
        self.def_post = DefectMetrics(def_conf, font_scale=cfg.get("font_scale",1.0),
                              thickness=cfg.get("font_thickness",2))

    # -------------------- production-callback ---------------------------
    def on_closed(self, event):
        src = Path(event.src_path)
        if src.suffix.lower() != ".pic":
            return
        size_prev = -1
        while src.stat().st_size != size_prev:
            size_prev = src.stat().st_size
            time.sleep(0.1)
        jpg = self.extract_last_jpeg(src)
        if not jpg:
            logging.error(f"JPEG not found in {src.name}")
            return
        logging.info(f"Extracted JPEG: {jpg.name}")
        self.process_jpg(jpg)

    # ------------------------ helpers -----------------------------------
    def extract_last_jpeg(self, pic_path: Path) -> Optional[Path]:
        with open(pic_path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            end = mm.rfind(b"\xFF\xD9");  start = mm.rfind(b"\xFF\xD8", 0, end)
            if end == -1 or start == -1: return None
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            out = self.output_dir / f"{pic_path.stem}_{ts}.jpg"
            with open(out, "wb") as o: o.write(mm[start:end+2])
            return out

    # -------------------- основной пайп-лайн ----------------------------
    def process_jpg(self, jpg_path: Path):
        """
        Для production jpg_path выглядит так:
            …/results/VA104-…_20250528_221055_123.jpg
        stem = VA104-…_20250528_221055_123   ← в stem уже есть ts
        Для test-режима stem = имя файла без расширения.
        По stem создаём подпапку, куда складываем всё остальное.
        """
        try:
            stem = jpg_path.stem
            out_dir = self.output_dir / stem
            out_dir.mkdir(parents=True, exist_ok=True)

            # 1) undistort
            frame  = cv2.imread(str(jpg_path))
            undist = self.dist_corr.correct(frame)

            # 2) perspective
            warped, scale = self.persp.transform(undist)
            cv2.imwrite(str(out_dir / "warped.png"), warped)

            # 3-a) профиль
            mask_prof, ov_prof, _, _ = self.seg_profile(warped)
            if mask_prof is None:
                logging.warning(f"No profile mask for {jpg_path.name}")
                return

            # 3-b) дефекты
            mask_def, _, masks_nd, cls_ids = self.seg_defect(warped)

            # 4) пост-процессинг
            prof_metrics, ov_prof = self.post(mask_prof, scale, warped, ov_prof)

            def_metrics, ov_def = {}, None
            if mask_def is not None and masks_nd is not None:
                def_metrics, ov_def = self.def_post(
                    masks_nd, cls_ids, scale, warped, prof_metrics["area_first_m2"]
                )

            # 5) сохранения
            cv2.imwrite(str(out_dir / "mask_profile.png"),   mask_prof)
            cv2.imwrite(str(out_dir / "overlay_profile.jpg"), ov_prof)

            if mask_def is not None:
                cv2.imwrite(str(out_dir / "mask_defects.png"),   mask_def)
            if ov_def is not None:
                cv2.imwrite(str(out_dir / "overlay_defects.jpg"), ov_def)

            # JSON
            with open(out_dir / "metrics.json", "w", encoding="utf-8") as jf:
                json.dump({**prof_metrics, **def_metrics},
                        jf, ensure_ascii=False, indent=2)

            logging.info(f"{stem}: {prof_metrics | def_metrics}")

        except Exception as e:
            logging.exception(f"Error processing {jpg_path.name}: {e}")

# --------------------------------------------------------------------------- #
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = yaml.safe_load(open("config.yaml", "r"))
    cfg["config_path"] = "config.yaml"

    handler = PipelineHandler(cfg)

    if cfg.get("test_mode", False):
        test_dir = Path(cfg["test_images_dir"])
        logging.info(f"Test mode ON: processing images from {test_dir}")
        for img in sorted(test_dir.iterdir()):
            if img.suffix.lower() in [".jpg", ".png"]:
                handler.process_jpg(img)
        return

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
