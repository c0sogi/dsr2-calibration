"""Hand-eye calibration solver and automated pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .detector import BoardDetector

# ── Doosan posx helpers ──────────────────────────────────────────────


def _zyz_to_rotmat(a_deg: float, b_deg: float, g_deg: float) -> np.ndarray:
    """Intrinsic ZYZ Euler angles (degrees) → 3×3 rotation matrix."""
    a, b, g = np.radians([a_deg, b_deg, g_deg])
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cg, sg = np.cos(g), np.sin(g)
    return np.array([
        [ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb],
        [sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb],
        [-sb * cg, sb * sg, cb],
    ])


def posx_to_matrix(posx: list[float] | np.ndarray) -> np.ndarray:
    """Convert Doosan ``posx [x,y,z,w,p,r]`` to a 4×4 matrix.

    * x/y/z in **mm** → converted to **meters**.
    * w/p/r are ZYZ Euler angles in degrees.
    """
    x, y, z, w, p, r = posx
    T = np.eye(4)
    T[:3, :3] = _zyz_to_rotmat(w, p, r)
    T[:3, 3] = [x / 1000.0, y / 1000.0, z / 1000.0]
    return T


# ── Calibration result ───────────────────────────────────────────────


@dataclass
class CalibrationResult:
    T_cam2gripper: np.ndarray  # 4×4
    n_samples: int = 0
    rms: float | None = None

    def save(self, path: str | Path) -> None:
        np.savez(
            Path(path),
            T_cam2gripper=self.T_cam2gripper,
            n_samples=self.n_samples,
            rms=self.rms if self.rms is not None else np.nan,
        )

    @classmethod
    def load(cls, path: str | Path) -> CalibrationResult:
        d = np.load(path)
        rms = float(d["rms"])
        return cls(
            T_cam2gripper=d["T_cam2gripper"],
            n_samples=int(d["n_samples"]),
            rms=None if np.isnan(rms) else rms,
        )


# ── Hand-eye calibrator ─────────────────────────────────────────────


class HandEyeCalibrator:
    """Collects (image, robot_pose) pairs and solves eye-in-hand AX=XB."""

    def __init__(
        self,
        detector: BoardDetector,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> None:
        self.detector = detector
        self.K = camera_matrix
        self.D = dist_coeffs
        self._Rg: list[np.ndarray] = []
        self._tg: list[np.ndarray] = []
        self._Rt: list[np.ndarray] = []
        self._tt: list[np.ndarray] = []

    @property
    def n_samples(self) -> int:
        return len(self._Rg)

    def add_sample(
        self,
        image: np.ndarray,
        robot_pose: np.ndarray | list[float],
    ) -> bool:
        """Add one sample. *robot_pose* is a 4×4 matrix **or** a ``posx`` list.

        Returns ``True`` if the board was detected and the sample was stored.
        """
        if not isinstance(robot_pose, np.ndarray) or robot_pose.ndim == 1:
            robot_pose = posx_to_matrix(robot_pose)

        det = self.detector.detect(image)
        if det is None:
            return False
        pose = self.detector.estimate_pose(det[0], det[1], self.K, self.D)
        if pose is None:
            return False

        rvec, tvec = pose
        R, _ = cv2.Rodrigues(rvec)
        self._Rg.append(robot_pose[:3, :3])
        self._tg.append(robot_pose[:3, 3].reshape(3, 1))
        self._Rt.append(R.astype(np.float64, copy=False))
        self._tt.append(tvec.reshape(3, 1))
        return True

    def calibrate(
        self,
        method: int = cv2.CALIB_HAND_EYE_TSAI,
    ) -> CalibrationResult:
        """Solve the hand-eye problem. Requires ≥ 3 samples."""
        if self.n_samples < 3:
            raise ValueError(f"Need ≥3 samples, have {self.n_samples}")

        R, t = cv2.calibrateHandEye(self._Rg, self._tg, self._Rt, self._tt, method=method)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return CalibrationResult(T_cam2gripper=T, n_samples=self.n_samples)

    def reset(self) -> None:
        self._Rg.clear()
        self._tg.clear()
        self._Rt.clear()
        self._tt.clear()


# ── Pose generation ──────────────────────────────────────────────────


# Doosan A0509 joint limits (degrees)
_A0509_JOINT_LIMITS: list[tuple[float, float]] = [
    (-360, 360),   # J1
    (-125, 125),   # J2
    (-150, 150),   # J3
    (-360, 360),   # J4
    (-125, 125),   # J5
    (-360, 360),   # J6
]


def generate_calibration_poses(
    center_joints: list[float],
    n_poses: int = 20,
    wrist_range: float = 20.0,
    arm_range: float = 8.0,
    seed: int = 42,
    joint_limits: list[tuple[float, float]] | None = None,
) -> list[list[float]]:
    """Generate *n_poses* joint configurations around *center_joints*.

    Wrist joints (4-6) are perturbed by ±*wrist_range*° and arm joints
    (1-3) by ±*arm_range*° to produce diverse camera viewpoints.
    Poses that violate joint limits are discarded.
    """
    limits = joint_limits or _A0509_JOINT_LIMITS
    rng = np.random.default_rng(seed)
    center = np.asarray(center_joints, dtype=float)
    poses: list[list[float]] = [center.tolist()]
    attempts = 0
    while len(poses) < n_poses and attempts < n_poses * 10:
        attempts += 1
        offset = np.zeros(6)
        offset[:3] = rng.uniform(-arm_range, arm_range, 3)
        offset[3:] = rng.uniform(-wrist_range, wrist_range, 3)
        candidate = center + offset
        if all(lo <= j <= hi for j, (lo, hi) in zip(candidate, limits)):
            poses.append(candidate.tolist())
    return poses


# ── Automated pipeline ───────────────────────────────────────────────


def auto_calibrate(
    detector: BoardDetector,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    calibration_joints: list[list[float]],
    move_fn: Callable[[list[float]], None],
    get_pose_fn: Callable[[], list[float] | np.ndarray],
    capture_fn: Callable[[], np.ndarray],
    settle_time: float = 1.0,
    method: int = cv2.CALIB_HAND_EYE_TSAI,
) -> CalibrationResult:
    """Fully-automated hand-eye calibration.

    Parameters
    ----------
    calibration_joints : list of joint-angle lists (degrees)
    move_fn            : movej — move robot to given joint angles
    get_pose_fn        : get_current_posx — returns posx or 4×4 matrix
    capture_fn         : returns a BGR image from the camera
    """
    cal = HandEyeCalibrator(detector, camera_matrix, dist_coeffs)

    for i, joints in enumerate(calibration_joints):
        move_fn(joints)
        time.sleep(settle_time)
        image = capture_fn()
        pose = get_pose_fn()
        ok = cal.add_sample(image, pose)
        status = "ok" if ok else "board not detected, skipped"
        print(f"  [{i + 1}/{len(calibration_joints)}] {status}")

    return cal.calibrate(method=method)
