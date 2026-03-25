"""ChArUco board detection and camera intrinsic calibration."""

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np


@dataclass
class BoardConfig:
    """ChArUco board parameters. Lengths are in meters."""

    cols: int = 5
    rows: int = 7
    square_length: float = 0.040
    marker_length: float = 0.030
    dict_id: int = cv2.aruco.DICT_4X4_50


class BoardDetector:
    """Detects a ChArUco board and estimates its 6-DoF pose."""

    def __init__(self, config: BoardConfig | None = None) -> None:
        self.config = config or BoardConfig()
        dictionary = cv2.aruco.getPredefinedDictionary(self.config.dict_id)
        self.board = cv2.aruco.CharucoBoard(
            (self.config.cols, self.config.rows),
            self.config.square_length,
            self.config.marker_length,
            dictionary,
        )
        self._detector = cv2.aruco.CharucoDetector(self.board)

    # ------------------------------------------------------------------

    def detect(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return *(charuco_corners, charuco_ids)* or ``None``."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners, ids, _, _ = self._detector.detectBoard(gray)
        if ids is None or len(ids) < 4:
            return None
        return corners, ids.astype(np.int32, copy=False)

    def estimate_pose(
        self,
        corners: np.ndarray,
        ids: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Estimate board pose → *(rvec, tvec)* or ``None``."""
        obj_pts, img_pts = self.board.matchImagePoints(corners, ids)  # type: ignore[arg-type]
        if len(obj_pts) < 4:
            return None
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs)
        if not ok:
            return None
        return rvec.astype(np.float64, copy=False), tvec.astype(np.float64, copy=False)

    def generate_image(self, dpi: int = 150) -> np.ndarray:
        """Render a board image at exact physical scale.

        When printed at *dpi* with 100% scaling (no "fit to page"),
        the squares will measure exactly ``square_length`` meters.
        """
        cfg = self.config
        m_to_px = dpi / 0.0254  # meters → pixels
        width = int(cfg.cols * cfg.square_length * m_to_px)
        height = int(cfg.rows * cfg.square_length * m_to_px)
        return self.board.generateImage((width, height)).astype(np.uint8, copy=False)


def calibrate_camera(
    detector: BoardDetector,
    images: list[np.ndarray],
    image_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Calibrate camera intrinsics from ChArUco images.

    Returns *(camera_matrix, dist_coeffs, rms_error)*.
    """
    all_obj: list[np.ndarray] = []
    all_img: list[np.ndarray] = []

    for img in images:
        result = detector.detect(img)
        if result is None:
            continue
        corners, ids = result
        obj_pts, img_pts = detector.board.matchImagePoints(corners, ids)  # type: ignore[arg-type]
        if len(obj_pts) >= 6:
            all_obj.append(obj_pts)
            all_img.append(img_pts)

    if len(all_obj) < 3:
        raise ValueError(
            f"Need ≥3 valid detections (≥6 corners each), got {len(all_obj)}"
        )

    if image_size is None:
        h, w = images[0].shape[:2]
        image_size = (w, h)

    _no_mat = cast(np.ndarray, None)
    try:
        rms, K, D, _, _ = cv2.calibrateCamera(
            all_obj,
            all_img,
            image_size,
            _no_mat,
            _no_mat,
        )
    except cv2.error:
        # initIntrinsicParams2D can fail when corners are sparse or
        # nearly collinear.  Fall back to a rough focal-length guess
        # so OpenCV skips its own homography-based initialisation.
        w, h = image_size
        f = float(max(w, h))
        K0 = np.array([[f, 0, w / 2.0],
                        [0, f, h / 2.0],
                        [0, 0, 1.0]], dtype=np.float64)
        D0 = np.zeros((5, 1), dtype=np.float64)
        rms, K, D, _, _ = cv2.calibrateCamera(
            all_obj,
            all_img,
            image_size,
            K0,
            D0,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS,
        )
    return K.astype(np.float64, copy=False), D.astype(np.float64, copy=False), rms
