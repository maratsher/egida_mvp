#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Основной сервис:
  • production-режим — слушает папку с .pic, извлекает последний JPEG и запускает весь пайплайн.
  • test_mode       — обходит изображения в test_images_dir и выполняет тот же пайплайн.

Результаты (JSON + маска + overlay + warped) сохраняются в output_dir.
"""
import time
import json
import yaml
import logging
import mmap
from pathlib import Path
from datetime import datetime
from typing import Optional

import cv2
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from src.preprocessing import DistortionCorrector, PerspectiveTransformer
from src.inference import Yolov8SegONNX
from src.postprocessing import ProfileMetrics


# --------------------------------------------------------------------------- #
#  Обработчик файлов .pic                                                     #
# --------------------------------------------------------------------------- #
class PipelineHandler(FileSystemEventHandler):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.input_dir = Path(cfg["input_dir"])
        self.output_dir = Path(cfg.get("output_dir", "./results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- подготовка стадий ---
        self.dist_corr = DistortionCorrector(cfg)
        self.persp = PerspectiveTransformer(cfg)
        self.seg = Yolov8SegONNX(cfg)
        self.post = ProfileMetrics(
            width_threshold=cfg.get("width_threshold", 0.8),
            font_scale=cfg.get("font_scale", 1.2),
            thickness=cfg.get("font_thickness", 2),
            color_first=tuple(cfg.get("color_first",  [0, 255, 0])),
            color_second=tuple(cfg.get("color_second", [0, 165, 255])),
        )

    # -------------------- production-callback ------------------------------
    def on_closed(self, event):
        src = Path(event.src_path)
        if src.suffix.lower() != ".pic":
            return

        # ждём, пока файл полностью запишется
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

    # ------------------------ helpers --------------------------------------
    def extract_last_jpeg(self, pic_path: Path) -> Optional[Path]:
        with open(pic_path, "rb") as f, mmap.mmap(
            f.fileno(), 0, access=mmap.ACCESS_READ
        ) as mm:
            end = mm.rfind(b"\xFF\xD9")
            if end == -1:
                return None
            start = mm.rfind(b"\xFF\xD8", 0, end)
            if start == -1:
                return None

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            out_name = f"{pic_path.stem}_{ts}.jpg"
            out_path = self.output_dir / out_name
            with open(out_path, "wb") as o:
                o.write(mm[start : end + 2])
            return out_path

    # -------------------- основной пайплайн --------------------------------
    def process_jpg(self, jpg_path: Path):
        try:
            # 1) Коррекция дисторсии
            frame = cv2.imread(str(jpg_path))
            undist = self.dist_corr.correct(frame)

            # 2) Перспективное выравнивание
            warped, scale = self.persp.transform(undist)
            cv2.imwrite(
                str(self.output_dir / f"{jpg_path.stem}_warped.png"), warped
            )

            # 3) Сегментация
            mask, overlay = self.seg(warped)
            if mask is None:
                logging.warning(f"No mask found for {jpg_path.name}")
                return

            # 4) Пост-процессинг (метрики + финальный overlay)
            metrics, overlay = self.post(mask, scale, warped, overlay)

            # 5) Сохранения
            cv2.imwrite(
                str(self.output_dir / f"{jpg_path.stem}_mask.png"), mask
            )
            cv2.imwrite(
                str(self.output_dir / f"{jpg_path.stem}_overlay.jpg"), overlay
            )
            with open(
                self.output_dir / f"{jpg_path.stem}.json", "w", encoding="utf-8"
            ) as jf:
                json.dump(metrics, jf, ensure_ascii=False, indent=2)

            logging.info(f"{jpg_path.name}: {metrics}")

        except Exception as e:
            logging.exception(f"Error processing {jpg_path.name}: {e}")


# --------------------------------------------------------------------------- #
#  Точка входа                                                                #
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

    # ------------------ тестовый режим -------------------------------------
    if cfg.get("test_mode", False):
        test_dir = Path(cfg["test_images_dir"])
        logging.info(f"Test mode ON: processing images from {test_dir}")
        for img in sorted(test_dir.iterdir()):
            if img.suffix.lower() in [".jpg", ".png"]:
                handler.process_jpg(img)
        return

    # ------------------ production-режим -----------------------------------
    observer = Observer()
    observer.schedule(handler, path=cfg["input_dir"], recursive=False)
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
