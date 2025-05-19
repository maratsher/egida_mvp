import numpy as np
import cv2

class DistortionCorrector:
    """
    Class to correct lens distortion using camera calibration parameters.

    Config schema:
      camera_matrix: 3x3 list of lists
      dist_coefs: list of distortion coefficients (k1, k2, p1, p2, k3)
    """
    def __init__(self, cfg: dict):
        self.camera_matrix = np.array(cfg['camera_matrix'], dtype=np.float32)
        # Flatten nested lists into 1D array
        self.dist_coefs = np.array(cfg['dist_coefs'], dtype=np.float32).reshape(-1)

    def correct(self, frame: np.ndarray) -> np.ndarray:
        """
        Undistort the input frame.
        :param frame: BGR image as numpy array
        :return: undistorted image
        """
        h, w = frame.shape[:2]
        # Compute optimal new camera matrix to minimize black regions
        new_cam_mtx, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coefs, (w, h), 1, (w, h)
        )
        undistorted = cv2.undistort(
            frame, self.camera_matrix, self.dist_coefs, None, new_cam_mtx
        )
        return undistorted