#!/usr/bin/env python3
"""
Batch decompress Hikvision .pic files in a directory tree.

Для каждого найденного .pic файла извлекаются все JPEG‑кадры и сохраняются в единую
папку вывода с именами:
    корневая_папка-подпапка1-подпапка2-имя_файла-номер_кадра.jpg

Пример запуска:
    python3 batch_decompress_pic.py \
      --input-dir /data/pics_root \
      --output-dir /data/output_jpegs \
      --every 1
"""

import argparse
import mmap
from pathlib import Path

def decompress_pic_file(pic_path: Path, input_root: Path, output_dir: Path, every: int):
    file_size = pic_path.stat().st_size  # <<<
    if file_size == 0:                   # <<<
        print(f"► Пропускаем пустой файл: {pic_path.relative_to(input_root)}")  # <<<
        return                            # <<<

    rel_parts = pic_path.relative_to(input_root).with_suffix('').parts

    with open(pic_path, 'rb') as f, \
         mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:

        picture_count = 0
        start = 0
        while True:
            start = mm.find(b'\xFF\xD8', start)
            if start < 0:
                break
            end = mm.find(b'\xFF\xD9', start)
            if end < 0:
                break
            end += 2

            picture_count += 1
            if picture_count % every == 0:
                parts = [input_root.name, *rel_parts, str(picture_count)]
                out_name = "-".join(parts) + ".jpg"
                out_path = output_dir / out_name
                with open(out_path, 'wb') as out_f:
                    out_f.write(mm[start:end])

            start = end

    print(f"[{picture_count}] frames in {pic_path.relative_to(input_root)} processed.")

def main():
    parser = argparse.ArgumentParser(
        description="Batch decompress Hikvision .pic files into JPEGs"
    )
    parser.add_argument(
        '-i', '--input-dir',
        required=True,
        help="Корневая папка, внутри которой искать .pic файлы"
    )
    parser.add_argument(
        '-o', '--output-dir',
        required=True,
        help="Папка для сохранения всех извлечённых .jpg"
    )
    parser.add_argument(
        '-e', '--every',
        type=int,
        default=1,
        help="Сохранять только каждый N‑й кадр (по умолчанию 1)"
    )

    args = parser.parse_args()
    input_root = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # обходим все .pic файлы рекурсивно
    pic_files = list(input_root.rglob("*.pic"))
    if not pic_files:
        print("Не найдено ни одного .pic файла в", input_root)
        return

    for pic_path in pic_files:
        decompress_pic_file(pic_path, input_root, output_dir, args.every)

if __name__ == "__main__":
    main()
