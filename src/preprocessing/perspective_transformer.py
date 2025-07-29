from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
class PerspectiveTransformer:
    def __init__(self, cfg: Dict):
        pts_cfg = cfg["perspective_points"]

        # ---------------------------------------------------- 4-точ. режим ---
        if _is_four_points(pts_cfg):
            self.mode = "4pt"
            order = ["TL", "BL", "TR", "BR"]
            self.pts_src = np.float32([pts_cfg[k] for k in order])

            self.real_width  = float(cfg["real_width"])
            self.real_height = float(cfg["real_height"])

        # ---------------------------------------------------- 6-точ. режим ---
        else:
            self.mode = "6pt"
            order = [f"p{i}" for i in range(1, 7)]
            self.pts_src = np.float32([pts_cfg[k] for k in order])

            rd: Dict[str, float] = cfg["real_distances"]

            # --- реальные координаты по оси Y (мм)
            y1 = 0.0
            y2 = rd["p1-p2"] if "p1-p2" in rd else rd["p2-p1"]
            y3 = rd["p1-p3"] if "p1-p3" in rd else rd["p3-p1"]

            # --- ширина между столбами (мм)
            w12 = rd["p1-p4"] if "p1-p4" in rd else rd["p4-p1"]
            w25 = rd["p2-p5"] if "p2-p5" in rd else rd["p5-p2"]
            w36 = rd["p3-p6"] if "p3-p6" in rd else rd["p6-p3"]
            self.real_width = (w12 + w25 + w36) / 3.0          # усредняем
            self.real_height = y3                              # самый дальний маркер

            # --- реальные координаты 6 маркеров (мм)
            self._dst_metric = np.float32(
                [
                    [0,            y1],   # p1
                    [0,            y2],   # p2
                    [0,            y3],   # p3
                    [self.real_width, y1],   # p4
                    [self.real_width, y2],   # p5
                    [self.real_width, y3],   # p6
                ]
            )

            # пары, по которым считаем масштаб
            vert_pairs = [("p1", "p2"), ("p2", "p3"), ("p1", "p3"),
                          ("p4", "p5"), ("p5", "p6"), ("p4", "p6")]

            scales = []
            for a, b in vert_pairs:
                key = f"{a}-{b}"
                key_rev = f"{b}-{a}"
                if key in rd or key_rev in rd:
                    real_mm = rd.get(key, rd.get(key_rev))
                    ia, ib = order.index(a), order.index(b)
                    pix = np.linalg.norm(self.pts_src[ia] - self.pts_src[ib])
                    scales.append(pix / real_mm)
            if not scales:
                raise ValueError("Не удаётся вычислить масштаб — проверьте real_distances")
            self._scale_mean = float(np.mean(scales))

            # матрица гомографии (по 6 соответствиям)
            dst_px = self._dst_metric * self._scale_mean
            self._M, _ = cv2.findHomography(self.pts_src, dst_px, method=0)

    # ------------------------------------------------------------------ #
    def transform(self, frame):
        """
        :param frame: исходный BGR-кадр
        :return: warped_img, scale_px_per_mm
        """
        if self.mode == "4pt":
            return self._transform_4pt(frame)
        return self._transform_6pt(frame)

    # ========================= private impl =========================== #
    def _transform_4pt(self, frame):
        # масштаб по левому ребру
        pix_h = np.linalg.norm(self.pts_src[0] - self.pts_src[1])
        scale = pix_h / self.real_height

        out_h = int(round(self.real_height * scale))
        out_w = int(round(self.real_width  * scale))

        dst = np.float32([[0, 0], [0, out_h], [out_w, 0], [out_w, out_h]])
        M = cv2.getPerspectiveTransform(self.pts_src, dst)
        warped = cv2.warpPerspective(frame, M, (out_w, out_h))
        return warped, scale

    def _transform_6pt(self, frame):
        out_w = int(round(self.real_width  * self._scale_mean))
        out_h = int(round(self.real_height * self._scale_mean))
        warped = cv2.warpPerspective(frame, self._M, (out_w, out_h))
        return warped, self._scale_mean

# --------------------------------------------------------------------------- #
def _is_four_points(pts_cfg) -> bool:
    """True, если конфиг описывает 4-точечную схему."""
    if isinstance(pts_cfg, dict):
        return set(pts_cfg.keys()) >= {"TL", "BL", "TR", "BR"}
    return False


# if __name__ == "__main__":
#     import yaml
#     cfg = yaml.safe_load(open('config.yaml', 'r'))
#     persp = PerspectiveTransformer(cfg)
    
#     jpg_path = "/Users/maratsher/Work/DCS/Egida/Egida_MVP/ref.jpg"
#     frame = cv2.imread(str(jpg_path))
#     warped, scale = persp.transform(frame)
#     warped_path = "./ref_warped.jpg"
#     cv2.imwrite(str(warped_path), warped)
#     print(scale)
    