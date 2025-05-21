import os
import yaml
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path


class ONNXInference:
    """
    Инференс Mask R-CNN (Detectron2 → ONNX).
    Принимает любые BGR-изображения float32 0-255 без нормализации и без ресайза.
    """

    def __init__(self, cfg_path: str | Path):
        cfg = yaml.safe_load(open(cfg_path, 'r'))
        self.onnx_path  = cfg['onnx_path']
        self.providers  = cfg.get('providers',
                                  ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.debug      = bool(cfg.get('debug', False))
        self.debug_dir  = cfg.get('debug_output', './debug')
        os.makedirs(self.debug_dir, exist_ok=True)
        self.color      = tuple(cfg.get('overlay_color', [0, 255, 0]))  # BGR

        # ONNX Runtime с динамическим H×W
        self.session    = ort.InferenceSession(self.onnx_path,
                                               providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        self.out_names  = [o.name for o in self.session.get_outputs()]

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
        """
        Подготавливает тензор 1×3×H×W (float32, BGR, 0-255).
        Никакого изменения размера и нормализации не выполняется.
        """
        img_f = img_bgr.astype(np.float32)
        inp = np.transpose(img_f, (2, 0, 1))[None]   # 1×3×H×W
        return inp

    # ──────────────────────────────────────────────────────────────────────────
    def infer(self, image_path: str | Path,
              conf_thr: float = 0.5,
              mask_thr: float = 0.5,
              alpha: float = 0.35) -> dict:
        """Возвращает dict с полями boxes, scores, classes."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(image_path)

        inp = self._preprocess(img)
        boxes, scores, classes, masks = self.session.run(
            self.out_names, {self.input_name: inp}
        )

        boxes   = boxes.astype(np.float32)   # (N, 4)  — XYXY
        scores  = scores.astype(np.float32)  # (N,)
        classes = classes.astype(np.int32)   # (N,)
        masks   = masks[:, 0]                # (N, 28, 28) логиты 0-1

        print(classes, flush=True)
        print(masks, flush=True)
        print(scores, flush=True)


        keep_boxes, keep_scores, keep_classes = [], [], []
        overlay = img.copy() if self.debug else None
        H, W = img.shape[:2]

        for box, score, cls, m28 in zip(boxes, scores, classes, masks):
            if score < conf_thr:
                continue

            # Координаты уже даны в масштабе исходного изображения
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, W), min(y2, H)
            if x2 <= x1 or y2 <= y1:
                continue

            box_w, box_h = x2 - x1, y2 - y1
            mask = cv2.resize(m28, (box_w, box_h), cv2.INTER_LINEAR)
            mask_bin = (mask > mask_thr).astype(np.uint8)
            if mask_bin.sum() == 0:
                continue

            keep_boxes.append([x1, y1, x2, y2])
            keep_scores.append(float(score))
            keep_classes.append(int(cls))

            if overlay is not None:
                color = np.array(self.color, dtype=np.uint8)
                roi = overlay[y1:y2, x1:x2]
                roi[mask_bin == 1] = (
                    roi[mask_bin == 1] * (1 - alpha) + color * alpha
                ).astype(np.uint8)

        if overlay is not None:
            out_name = Path(self.debug_dir) / Path(image_path).name
            cv2.imwrite(str(out_name), overlay)

        return {"boxes": keep_boxes,
                "scores": keep_scores,
                "classes": keep_classes}
