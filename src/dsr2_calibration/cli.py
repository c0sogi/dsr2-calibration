"""CLI entry point for dsr2-calibration.

Usage (no robot):
    dsr2-calibration generate-charuco
    dsr2-calibration calibrate-camera --images-dir ./images

Usage (with robot — Docker container must be running):
    dsr2-calibration calibrate -j 0,0,90,0,90,0
    dsr2-calibration calibrate                        # uses current robot pose
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .calibration import auto_calibrate, generate_calibration_poses
from .detector import BoardConfig, BoardDetector, calibrate_camera
from .robot import DSR2Robot

_DELTA_PREFIX = "d:"


# -- helpers ---------------------------------------------------------------


def _board_from_args(args: argparse.Namespace) -> BoardConfig:
    if args.marker_length >= args.square_length:
        sys.exit("--marker-length must be smaller than --square-length")
    return BoardConfig(
        cols=args.cols,
        rows=args.rows,
        square_length=args.square_length,
        marker_length=args.marker_length,
    )


def _add_board_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("board")
    g.add_argument("--cols", type=int, default=5,
                   help="board columns (default: 5)")
    g.add_argument("--rows", type=int, default=7,
                   help="board rows (default: 7)")
    g.add_argument("--square-length", type=float, default=0.040,
                   help="black/white checkerboard square side length "
                        "in meters (default: 0.040 = 40mm)")
    g.add_argument("--marker-length", type=float, default=0.030,
                   help="ArUco marker side length in meters, must be smaller "
                        "than square-length (default: 0.030 = 30mm)")


def _add_pose_args(p: argparse.ArgumentParser, required: bool = False) -> None:
    g = p.add_mutually_exclusive_group(required=required)
    g.add_argument("-j", "--joints",
                   help="joint angles in degrees, or d:delta "
                        "(e.g. 0,0,90,0,90,0 or d:5,0,-5,0,0,0)")
    g.add_argument("-x", "--posx",
                   help="Cartesian [x,y,z,w,p,r] in mm/deg, or d:delta "
                        "(e.g. -300,0,500,0,180,0 or d:100,0,0,0,0,0)")


def _add_robot_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("robot")
    g.add_argument("--container", default="ros-control-real", help="Docker container name")
    g.add_argument("--vel", type=float, default=30.0,
                   help="joint velocity in deg/s (default: 30)")
    g.add_argument("--acc", type=float, default=30.0,
                   help="joint acceleration in deg/s^2 (default: 30)")
    g.add_argument("--settle-time", type=float, default=1.0,
                   help="wait time after each move in seconds (default: 1.0)")
    g.add_argument("--wrist-range", type=float, default=20.0,
                   help="wrist joint perturbation in degrees (default: 20)")
    g.add_argument("--arm-range", type=float, default=8.0,
                   help="arm joint perturbation in degrees (default: 8)")


def _make_capture(camera_id: int):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        sys.exit(
            f"Cannot open camera {camera_id}. "
            "If using RealSense, try different --camera IDs (0, 1, 2, ...) "
            "or check that the camera is connected."
        )

    def capture() -> np.ndarray:
        # Flush buffered frames to get the latest one
        for _ in range(5):
            cap.grab()
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Camera capture failed")
        return frame

    return capture, cap


def _parse_floats(s: str, expected: int = 6) -> list[float]:
    vals = [float(v) for v in s.split(",")]
    if len(vals) != expected:
        sys.exit(f"Expected {expected} comma-separated values, got {len(vals)}: {s}")
    return vals


def _parse_pose(s: str) -> tuple[list[float], bool]:
    """Parse '0,0,90,0,90,0' or 'd:10,0,-5,0,0,0'. Returns (values, is_delta)."""
    if s.startswith(_DELTA_PREFIX):
        return _parse_floats(s[len(_DELTA_PREFIX):]), True
    return _parse_floats(s), False


def _resolve_center_joints(args: argparse.Namespace, robot: DSR2Robot) -> list[float]:
    """Resolve center pose to joint angles from -j, -x, or current position."""
    if getattr(args, "joints", None):
        vals, delta = _parse_pose(args.joints)
        if delta:
            cur = robot.get_posj()
            return [c + d for c, d in zip(cur, vals)]
        return vals
    if getattr(args, "posx", None):
        vals, delta = _parse_pose(args.posx)
        if delta:
            cur = robot.get_posx()
            posx = [c + d for c, d in zip(cur, vals)]
        else:
            posx = vals
        return robot.ikin(posx)
    return robot.get_posj()


# -- commands --------------------------------------------------------------


def cmd_preview(args: argparse.Namespace) -> None:
    board = _board_from_args(args)
    detector = BoardDetector(board)
    capture_fn, cap = _make_capture(args.camera)

    print("Press 'q' to quit.")
    while True:
        frame = capture_fn()
        result = detector.detect(frame)
        if result is not None:
            corners, ids = result
            cv2.aruco.drawDetectedCornersCharuco(frame, corners, ids, (0, 255, 0))
            cv2.putText(
                frame, f"Detected: {len(ids)} corners",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
        else:
            cv2.putText(
                frame, "Board not detected",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
            )

        cv2.imshow("dsr2-calibration preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def cmd_dry_run(args: argparse.Namespace) -> None:
    board = _board_from_args(args)
    detector = BoardDetector(board)
    capture_fn, cap = _make_capture(args.camera)

    # Use low speed for safety
    safe_vel = min(args.vel, 10.0)
    safe_acc = min(args.acc, 10.0)

    with DSR2Robot(container=args.container, vel=safe_vel, acc=safe_acc) as robot:
        center = _resolve_center_joints(args, robot)
        poses = generate_calibration_poses(
            center, n_poses=args.n_poses,
            wrist_range=args.wrist_range, arm_range=args.arm_range,
        )

        print(f"Dry run: {len(poses)} poses at {safe_vel} deg/s")
        print("Watch the robot and camera feed. Press 'q' to abort.\n")

        detected = 0
        for i, joints in enumerate(poses):
            robot.move_to_joints(joints)
            time.sleep(args.settle_time)

            frame = capture_fn()
            result = detector.detect(frame)
            posx = robot.get_posx()

            if result is not None:
                detected += 1
                corners, ids = result
                cv2.aruco.drawDetectedCornersCharuco(frame, corners, ids, (0, 255, 0))
                status = f"[{i + 1}/{len(poses)}] OK ({len(ids)} corners)"
                color = (0, 255, 0)
            else:
                status = f"[{i + 1}/{len(poses)}] Board not visible"
                color = (0, 0, 255)

            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(
                frame,
                f"posx: [{posx[0]:.0f}, {posx[1]:.0f}, {posx[2]:.0f}, "
                f"{posx[3]:.0f}, {posx[4]:.0f}, {posx[5]:.0f}]",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
            cv2.imshow("dsr2-calibration dry-run", frame)
            print(f"  {status}  posx=[{', '.join(f'{v:.1f}' for v in posx)}]")

            if cv2.waitKey(500) & 0xFF == ord("q"):
                print("\nAborted by user.")
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nResult: {detected}/{len(poses)} poses with board visible")
    if detected < 3:
        print("Not enough detections. Adjust board position or center pose.")
    else:
        print("Ready for calibration.")


def cmd_generate_charuco(args: argparse.Namespace) -> None:
    board = _board_from_args(args)
    detector = BoardDetector(board)
    dpi = args.dpi
    img = detector.generate_image(dpi=dpi)

    # Write with DPI metadata via PIL if available, else fall back to OpenCV
    try:
        from PIL import Image  # type: ignore[import-not-found]

        Image.fromarray(img).save(args.output, dpi=(dpi, dpi))
    except ImportError:
        cv2.imwrite(args.output, img)

    sq_mm = board.square_length * 1000
    print(f"Saved {args.output}  ({img.shape[1]}x{img.shape[0]}px, {dpi} DPI)")
    print(f"Print at 100% scale - each square should measure {sq_mm:.0f}mm")


def cmd_calibrate_camera(args: argparse.Namespace) -> None:
    board = _board_from_args(args)
    detector = BoardDetector(board)

    if args.images_dir:
        img_dir = Path(args.images_dir)
        paths = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
        if not paths:
            sys.exit(f"No images found in {img_dir}")
        images = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                print(f"  Warning: skipping unreadable file {p}")
                continue
            images.append(img)
        if not images:
            sys.exit("No valid images found")
        print(f"Loaded {len(images)} images from {img_dir}")
    else:
        capture_fn, cap = _make_capture(args.camera)
        images = []
        with DSR2Robot(container=args.container, vel=args.vel, acc=args.acc) as robot:
            center = _resolve_center_joints(args, robot)
            poses = generate_calibration_poses(
                center, n_poses=args.n_images,
                wrist_range=args.wrist_range, arm_range=args.arm_range,
            )
            for i, joints in enumerate(poses):
                robot.move_to_joints(joints)
                time.sleep(args.settle_time)
                frame = capture_fn()
                if detector.detect(frame) is not None:
                    images.append(frame)
                    print(f"  [{i + 1}/{len(poses)}] detected ({len(images)} total)")
                else:
                    print(f"  [{i + 1}/{len(poses)}] board not found, skipped")
        cap.release()

    K, D, rms = calibrate_camera(detector, images)
    np.savez(args.output, K=K, D=D)
    print(f"RMS reprojection error: {rms:.4f}")
    print(f"Saved {args.output}")


def cmd_calibrate_transform(args: argparse.Namespace) -> None:
    try:
        intrinsics = np.load(args.intrinsics)
    except FileNotFoundError:
        sys.exit(f"{args.intrinsics} not found. Run calibrate-camera first.")
    K, D = intrinsics["K"], intrinsics["D"]

    board = _board_from_args(args)
    detector = BoardDetector(board)
    capture_fn, cap = _make_capture(args.camera)
    try:
        with DSR2Robot(container=args.container, vel=args.vel, acc=args.acc) as robot:
            center = _resolve_center_joints(args, robot)
            poses = generate_calibration_poses(
                center, n_poses=args.n_poses,
                wrist_range=args.wrist_range, arm_range=args.arm_range,
            )
            result = auto_calibrate(
                detector=detector,
                camera_matrix=K,
                dist_coeffs=D,
                calibration_joints=poses,
                move_fn=robot.move_to_joints,
                get_pose_fn=robot.get_posx,
                capture_fn=capture_fn,
                settle_time=args.settle_time,
            )
    finally:
        cap.release()

    result.save(args.output)
    print(f"\nT_cam2gripper:\n{result.T_cam2gripper}")
    print(f"Saved {args.output}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    from .calibration import HandEyeCalibrator, posx_to_matrix

    board = _board_from_args(args)
    detector = BoardDetector(board)
    capture_fn, cap = _make_capture(args.camera)
    try:
        with DSR2Robot(container=args.container, vel=args.vel, acc=args.acc) as robot:
            center = _resolve_center_joints(args, robot)
            poses = generate_calibration_poses(
                center, n_poses=args.n_poses,
                wrist_range=args.wrist_range, arm_range=args.arm_range,
            )

            # Single pass: collect images + robot poses together
            print(f"Collecting data ({len(poses)} poses)...")
            images: list[np.ndarray] = []
            robot_poses: list[np.ndarray] = []
            for i, joints in enumerate(poses):
                robot.move_to_joints(joints)
                time.sleep(args.settle_time)
                frame = capture_fn()
                if detector.detect(frame) is not None:
                    images.append(frame)
                    robot_poses.append(posx_to_matrix(robot.get_posx()))
                    print(f"  [{i + 1}/{len(poses)}] detected ({len(images)} total)")
                else:
                    print(f"  [{i + 1}/{len(poses)}] board not found, skipped")
    finally:
        cap.release()

    if len(images) < 3:
        sys.exit(
            f"Only {len(images)} board detections (need >= 3). "
            "Run 'dsr2-calibration preview' to check board visibility, "
            "or adjust board position / center pose."
        )

    # Step 1: camera intrinsics from collected images
    print(f"\n[1/2] Camera intrinsics ({len(images)} images)...")
    K, D, rms = calibrate_camera(detector, images)
    intrinsics_path = args.output.replace(".npz", "_intrinsics.npz")
    np.savez(intrinsics_path, K=K, D=D)
    print(f"  RMS reprojection error: {rms:.4f}")
    print(f"  Saved {intrinsics_path}")

    # Step 2: hand-eye transform from same data
    print("\n[2/2] Camera-to-gripper transform...")
    calibrator = HandEyeCalibrator(detector, K, D)
    for img, pose in zip(images, robot_poses):
        calibrator.add_sample(img, pose)

    result = calibrator.calibrate()
    result.save(args.output)
    print(f"\nT_cam2gripper:\n{result.T_cam2gripper}")
    print(f"Saved {args.output} ({result.n_samples} samples)")


# -- CLI -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dsr2-calibration",
        description="Doosan A0509 hand-eye calibration",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # preview
    p = sub.add_parser("preview", help="Live camera feed with board detection overlay")
    _add_board_args(p)
    p.add_argument("--camera", type=int, default=0)
    p.set_defaults(func=cmd_preview)

    # dry-run
    p = sub.add_parser("dry-run",
                        help="Test all poses at low speed with camera feed before calibrating")
    _add_board_args(p)
    _add_pose_args(p)
    _add_robot_args(p)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("-n", "--n-poses", type=int, default=20)
    p.set_defaults(func=cmd_dry_run)

    # generate-charuco
    p = sub.add_parser("generate-charuco", help="Save a printable ChArUco board image")
    _add_board_args(p)
    p.add_argument("--dpi", type=int, default=150, help="image resolution (default: 150)")
    p.add_argument("-o", "--output", default="charuco_board.png")
    p.set_defaults(func=cmd_generate_charuco)

    # calibrate-camera
    p = sub.add_parser("calibrate-camera", help="Compute camera lens parameters (K, D)")
    _add_board_args(p)
    p.add_argument("--images-dir", help="Offline: directory of board images")
    _add_pose_args(p)
    _add_robot_args(p)
    p.add_argument("-n", "--n-images", type=int, default=20)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("-o", "--output", default="camera_intrinsics.npz")
    p.set_defaults(func=cmd_calibrate_camera)

    # calibrate-transform
    p = sub.add_parser("calibrate-transform", help="Compute camera-to-gripper transform")
    _add_board_args(p)
    _add_pose_args(p)
    _add_robot_args(p)
    p.add_argument("-i", "--intrinsics", default="camera_intrinsics.npz")
    p.add_argument("-n", "--n-poses", type=int, default=15)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("-o", "--output", default="calibration_result.npz")
    p.set_defaults(func=cmd_calibrate_transform)

    # calibrate
    p = sub.add_parser("calibrate", help="Full auto-calibration (camera + transform)")
    _add_board_args(p)
    _add_pose_args(p)
    _add_robot_args(p)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("-n", "--n-poses", type=int, default=20)
    p.add_argument("-o", "--output", default="calibration_result.npz")
    p.set_defaults(func=cmd_calibrate)

    args = parser.parse_args()
    args.func(args)
