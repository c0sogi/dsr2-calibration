"""Example: automated hand-eye calibration on Doosan A0509.

Run inside the DSR2 Docker container where DSR_ROBOT2 is available.
"""

from __future__ import annotations

import cv2
import numpy as np

from dsr2_calibration import (
    BoardConfig,
    BoardDetector,
    auto_calibrate,
    calibrate_camera,
    generate_calibration_poses,
)

CAMERA_ID = 0

# Board you printed (edit to match your actual board)
BOARD = BoardConfig(cols=5, rows=7, square_length=0.040, marker_length=0.030)

# A joint pose where the camera clearly sees the board.
# ★ Teach this on the real robot first, then paste the values here.
CENTER_JOINTS = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]


def capture(cap: cv2.VideoCapture) -> np.ndarray:
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Camera capture failed")
    return frame


def main() -> None:
    detector = BoardDetector(BOARD)

    # -- 0. Save printable board image ------------------------------------
    cv2.imwrite("charuco_board.png", detector.generate_image())
    print("Saved charuco_board.png — print and fix to a flat surface.\n")

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAMERA_ID}")

    try:
        # -- 1. Camera intrinsic calibration ------------------------------
        print("[Step 1] Intrinsic calibration — capture ~15-20 board images.")
        images: list[np.ndarray] = []
        while len(images) < 15:
            key = input(f"  Press Enter to capture ({len(images)}/15), 'q' to finish early: ")
            if key.strip().lower() == "q" and len(images) >= 3:
                break
            frame = capture(cap)
            if detector.detect(frame) is not None:
                images.append(frame)
                print(f"    ✓ Detected ({len(images)} total)")
            else:
                print("    ✗ Board not found — adjust and retry")

        K, D, rms = calibrate_camera(detector, images)
        print(f"  Intrinsic RMS error: {rms:.4f}")
        np.savez("camera_intrinsics.npz", K=K, D=D)
        print("  Saved camera_intrinsics.npz\n")

        # -- 2. Hand-eye calibration (automatic) --------------------------
        print("[Step 2] Hand-eye calibration — robot moves automatically.")

        # ---- DSR_ROBOT2 imports (ROS 2 container only) ----
        from DSR_ROBOT2 import get_current_posx, movej  # type: ignore[import-not-found]

        poses = generate_calibration_poses(CENTER_JOINTS, n_poses=15)

        result = auto_calibrate(
            detector=detector,
            camera_matrix=K,
            dist_coeffs=D,
            calibration_joints=poses,
            move_fn=lambda j: movej(j, vel=30, acc=30),
            get_pose_fn=lambda: list(get_current_posx()[0]),
            capture_fn=lambda: capture(cap),
        )

        result.save("calibration_result.npz")
        print(f"\nDone — {result.n_samples} samples used.")
        print(f"T_cam2gripper:\n{result.T_cam2gripper}")
        print("Saved calibration_result.npz")

    finally:
        cap.release()


if __name__ == "__main__":
    main()
