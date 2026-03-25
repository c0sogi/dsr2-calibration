"""ChArUco board detection and camera intrinsic calibration."""

from dataclasses import dataclass
from typing import Sequence, cast

import cv2
import numpy as np
from numpy import typing as npt


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
        self, image: npt.NDArray[np.uint8]
    ) -> tuple[Sequence[npt.NDArray[np.float32]], npt.NDArray[np.int32]] | None:
        """Return *(charuco_corners, charuco_ids)* or ``None``."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners, ids, _, _ = self._detector.detectBoard(gray)
        corners = cast(Sequence[npt.NDArray[np.float32]], corners)
        if len(ids) < 4:
            return None
        return corners, ids.astype(np.int32, copy=False)

    def estimate_pose(
        self,
        corners: Sequence[npt.NDArray[np.float32]],
        ids: npt.NDArray[np.int32],
        camera_matrix: npt.NDArray[np.float64],
        dist_coeffs: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
        """Estimate board pose → *(rvec, tvec)* or ``None``."""
        obj_pts, img_pts = self.board.matchImagePoints(corners, ids)
        if len(obj_pts) < 4:
            return None
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs)
        if not ok:
            return None
        return rvec.astype(np.float64, copy=False), tvec.astype(np.float64, copy=False)

    def generate_image(self, width: int = 800, height: int = 1100) -> npt.NDArray[np.uint8]:
        """Render a printable board image."""
        return self.board.generateImage((width, height)).astype(np.uint8, copy=False)


def calibrate_camera(
    detector: BoardDetector,
    images: list[npt.NDArray[np.uint8]],
    image_size: tuple[int, int] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
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
        obj_pts, img_pts = detector.board.matchImagePoints(corners, ids)
        if len(obj_pts) >= 4:
            all_obj.append(obj_pts)
            all_img.append(img_pts)

    if len(all_obj) < 3:
        raise ValueError(f"Need ≥3 valid detections, got {len(all_obj)}")

    if image_size is None:
        h, w = images[0].shape[:2]
        image_size = (w, h)

    _no_mat = cast(np.ndarray, None)
    rms, K, D, _, _ = cv2.calibrateCamera(
        all_obj,
        all_img,
        image_size,
        _no_mat,
        _no_mat,
    )
    return K.astype(np.float64, copy=False), D.astype(np.float64, copy=False), rms
