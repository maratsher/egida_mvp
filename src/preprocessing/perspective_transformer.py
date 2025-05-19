import numpy as np
import cv2

class PerspectiveTransformer:
    """
    Class to apply perspective correction and compute pixel-to-real scale.

    Config schema:
      perspective_points: either dict with keys ['TL','BL','TR','BR'] or list of four [x, y] pairs
      real_width: real-world width (in same units as height)
      real_height: real-world height
    """
    def __init__(self, cfg: dict):
        pts = cfg['perspective_points']
        # Support both dict and list input
        if isinstance(pts, dict):
            order = ['TL', 'BL', 'TR', 'BR']
            self.pts_src = np.float32([pts[k] for k in order])
        else:
            # assume list of four [x, y]
            self.pts_src = np.float32(pts)

        self.real_width = float(cfg['real_width'])
        self.real_height = float(cfg['real_height'])

    def transform(self, frame: np.ndarray):
        """
        Apply perspective warp to the frame and compute pixel-to-real scale.
        :param frame: BGR image
        :return: tuple(warped_image, scale_pixels_per_unit)
        """
        # Compute pixel distance between top-left and bottom-left
        pix_h = np.linalg.norm(self.pts_src[0] - self.pts_src[1])
        scale = pix_h / self.real_height

        # Compute output image size in pixels
        out_h = int(round(self.real_height * scale))
        out_w = int(round(self.real_width * scale))

        # Destination rectangle
        pts_dst = np.float32([
            [0,      0],        # TL
            [0,      out_h],    # BL
            [out_w,  0],        # TR
            [out_w,  out_h]     # BR
        ])

        # Compute perspective matrix and warp
        M = cv2.getPerspectiveTransform(self.pts_src, pts_dst)
        warped = cv2.warpPerspective(frame, M, (out_w, out_h))
        return warped, scale