#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Утилита для интерактивного выбора четырёх точек на исправленном изображении
в порядке TL, BL, TR, BR и вывода их координат в консоль.
"""
import argparse
import cv2
import yaml
import numpy as np

points = []
labels = ['TL', 'BL', 'TR', 'BR']


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"{labels[len(points)-1]}: x={x}, y={y}")


def load_calibration(yaml_path):
    """Load camera matrix and distortion coefficients from YAML file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    cam = np.array(data['camera_matrix'], dtype=np.float32)
    dist = np.array(data['dist_coefs'], dtype=np.float32).reshape(-1, 1)
    return cam, dist


def main():
    parser = argparse.ArgumentParser(
        description="Выберите 4 точки на искажённом изображении после коррекции дисторсии"
    )
    parser.add_argument(
        '-i', '--image', required=True,
        help='Путь к исходному изображению'
    )
    parser.add_argument(
        '-y', '--yaml', required=True,
        help='Путь к YAML-файлу с параметрами камеры'
    )
    args = parser.parse_args()

    # Загрузка изображения
    img = cv2.imread(args.image)
    if img is None:
        print(f"Ошибка: не удалось загрузить {args.image}")
        return

    # Загрузка параметров камеры
    cam_mtx, dist_coefs = load_calibration(args.yaml)

    # Коррекция дисторсии
    h, w = img.shape[:2]
    new_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(cam_mtx, dist_coefs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(img, cam_mtx, dist_coefs, None, new_cam_mtx)

    window_name = 'Select 4 points (TL,BL,TR,BR) on undistorted image'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    print("Кликните по четырём точкам в окне, в порядке TL → BL → TR → BR")
    print("Нажмите 'q' после последнего клика для выхода.")

    while True:
        disp = undistorted.copy()
        for idx, (x, y) in enumerate(points):
            cv2.circle(disp, (x, y), 5, (0, 0, 255), 2)
            cv2.putText(disp, labels[idx], (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow(window_name, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') and len(points) == 4:
            break

    cv2.destroyAllWindows()

    if len(points) != 4:
        print(f"Выбрано {len(points)} точек, требуется 4.")
    else:
        print("Итоговые координаты (на исправленном изображении):")
        for label, (x, y) in zip(labels, points):
            print(f"{label}: ({x}, {y})")


if __name__ == '__main__':
    main()