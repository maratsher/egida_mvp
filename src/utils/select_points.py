#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Утилита для интерактивного выбора четырёх точек на изображении
в порядке TL, BL, TR, BR и вывода их координат в консоль.
"""
import argparse
import cv2

points = []
labels = ['TL', 'BL', 'TR', 'BR']


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"{labels[len(points)-1]}: x={x}, y={y}")


def main():
    parser = argparse.ArgumentParser(
        description="Выберите 4 точки на изображении в порядке: TL, BL, TR, BR"
    )
    parser.add_argument(
        '-i', '--image', required=True,
        help='Путь к изображению'
    )
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Ошибка: не удалось загрузить {args.image}")
        return

    window_name = 'Select 4 points (TL,BL,TR,BR)'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    print("Кликните по четырём точкам в окне, в порядке TL → BL → TR → BR")
    print("Нажмите 'q' после последнего клика для выхода.")

    while True:
        disp = img.copy()
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
        print("Итоговые координаты:")
        for label, (x, y) in zip(labels, points):
            print(f"{label}: ({x}, {y})")


if __name__ == '__main__':
    main()
