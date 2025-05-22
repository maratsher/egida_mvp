#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Основной сервис:
  • production‑режим — слушает папку с .pic, извлекает последний JPEG и запускает ONNX‑инференс.
  • test_mode       — обходит изображения в test_images_dir и выполняет тот же пайплайн.

Результаты (JSON + debug‑оверлеи) сохраняются в output_dir.
"""
import os
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

class PipelineHandler(FileSystemEventHandler):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.input_dir = Path(cfg['input_dir'])
        self.output_dir = Path(cfg.get('output_dir', './results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Предобработка
        self.dist_corr = DistortionCorrector(cfg)
        self.persp = PerspectiveTransformer(cfg)

        self.seg = Yolov8SegONNX(cfg)

    # production: callback when .pic closed
    def on_closed(self, event):
        src = Path(event.src_path)
        if src.suffix.lower() != '.pic':
            return
        # wait until file stops growing
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

    # -------- helpers --------
    def extract_last_jpeg(self, pic_path: Path) -> Optional[Path]:
        with open(pic_path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            end = mm.rfind(b"\xFF\xD9")
            if end == -1:
                return None
            start = mm.rfind(b"\xFF\xD8", 0, end)
            if start == -1:
                return None
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            out_name = f"{pic_path.stem}_{ts}.jpg"
            outp = self.output_dir / out_name
            with open(outp, 'wb') as o:
                o.write(mm[start:end+2])
            return outp

    def process_jpg(self, jpg_path: Path):
        try:
            # 1) Distortion correction
            frame = cv2.imread(str(jpg_path))
            undist = self.dist_corr.correct(frame)
            # 2) Perspective warp
            warped, scale = self.persp.transform(undist)

            warped_path = self.output_dir / f"{jpg_path.stem}_warped.png"
            cv2.imwrite(str(warped_path), warped)

            # 3) SEGMENTATION
            mask, overlay = self.seg(warped)
            if mask is None:
                logging.warning(f"No mask found for {jpg_path.name}")
                return

            # ---------- сохранение ----------
            mask_path = self.output_dir / f"{jpg_path.stem}_mask.png"
            cv2.imwrite(str(mask_path), mask)

            if overlay is not None:
                dbg_path  = self.output_dir / f"{jpg_path.stem}_overlay.jpg"
                cv2.imwrite(str(dbg_path), overlay)

        except Exception as e:
            logging.exception(f"Error processing {jpg_path.name}: {e}")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    cfg = yaml.safe_load(open('config.yaml', 'r'))
    cfg['config_path'] = 'config.yaml'

    handler = PipelineHandler(cfg)

    if cfg.get('test_mode', False):
        test_dir = Path(cfg['test_images_dir'])
        logging.info(f"Test mode ON: processing images from {test_dir}")
        for img in sorted(test_dir.iterdir()):
            if img.suffix.lower() in ['.jpg', '.png']:
                handler.process_jpg(img)
        return

    observer = Observer()
    observer.schedule(handler, path=cfg['input_dir'], recursive=False)
    observer.start()
    logging.info(f"Watching {cfg['input_dir']} for .pic files…  Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == '__main__':
    main()
