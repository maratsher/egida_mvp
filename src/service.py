#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Основной сервис: бесконечно слушает папку .pic (production) или обрабатывает все изображения из test_images_dir (test_mode),
применяет дисторсию и перспективу, строит engine (если нужно), выполняет инференс и сохраняет JSON.
"""
import time
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
# from trt_service import EngineBuilder, TRTInference


class PipelineHandler(FileSystemEventHandler):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.input_dir = Path(cfg['input_dir'])
        self.output_dir = Path(cfg.get('output_dir', '.'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dist_corr = DistortionCorrector(cfg)
        self.persp = PerspectiveTransformer(cfg)

        # self.engine_builder = EngineBuilder(cfg_path=cfg['config_path'])
        # self.inferer = TRTInference(cfg_path=cfg['config_path'])

    def on_closed(self, event):
        src = Path(event.src_path)
        if src.suffix.lower() != '.pic':
            return
        # ждём окончания записи
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
            # коррекция дисторсии
            frame = cv2.imread(str(jpg_path))
            undist = self.dist_corr.correct(frame)
            # коррекция перспективы
            warped, scale = self.persp.transform(undist)
            print(scale)
            warped_path = self.output_dir / f"{jpg_path.stem}_warped.jpg"
            cv2.imwrite(str(warped_path), warped)
            # # билд engine
            # self.engine_builder.build()
            # # инференс
            # result = self.inferer.infer(str(warped_path))
            # result['scale'] = scale
            # # сохраняем JSON
            # jpath = self.output_dir / f"{jpg_path.stem}.json"
            # with open(jpath, 'w') as jf:
            #     json.dump(result, jf, ensure_ascii=False, indent=2)
            # logging.info(f"Result saved: {jpath.name}")
        except Exception as e:
            logging.exception(f"Error processing {jpg_path.name}: {e}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    cfg = yaml.safe_load(open('config.yaml', 'r'))
    cfg['config_path'] = 'config.yaml'

    test_mode = bool(cfg.get('test_mode', False))
    handler = PipelineHandler(cfg)

    if test_mode:
        test_dir = Path(cfg['test_images_dir'])
        logging.info(f"Test mode ON: processing all images in {test_dir}")
        for img in sorted(test_dir.iterdir()):
            if img.suffix.lower() in ['.jpg', '.png']:  # любые форматы
                logging.info(f"Processing test image: {img.name}")
                handler.process_jpg(img)
        return

    observer = Observer()
    observer.schedule(handler, path=cfg['input_dir'], recursive=False)
    observer.start()
    logging.info(f"Watching {cfg['input_dir']} for new .pic files...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == '__main__':
    main()
