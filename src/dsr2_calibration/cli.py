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
import atexit
from datetime import datetime
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .calibration import auto_calibrate, generate_calibration_poses
from .detector import BoardConfig, BoardDetector, calibrate_camera
from .robot import DSR2Robot

# Global camera registry for guaranteed cleanup on exit / signal.
_active_caps: list[cv2.VideoCapture] = []


def _cleanup_cameras() -> None:
    for cap in _active_caps:
        try:
            cap.release()
        except Exception:
            pass
    _active_caps.clear()


atexit.register(_cleanup_cameras)


def _signal_cleanup(signum: int, _frame: object) -> None:
    _cleanup_cameras()
    sys.exit(128 + signum)


# SIGINT is already KeyboardInterrupt in Python, but SIGTERM needs a handler.
# On Windows SIGTERM is never delivered externally, so skip registration.
if sys.platform != "win32":
    signal.signal(signal.SIGTERM, _signal_cleanup)

_DELTA_PREFIX = "d:"


def _timestamped_name(base: str) -> str:
    """Generate a filename with a timestamp, e.g. 'calibration_result_20260325_143012.npz'."""
    stem, _, ext = base.rpartition(".")
    if not stem:
        stem, ext = ext, ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{ts}.{ext}" if ext else f"{stem}_{ts}"


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


def _release_capture(cap: cv2.VideoCapture) -> None:
    """Release a capture and remove it from the global registry."""
    try:
        cap.release()
    except Exception:
        pass
    try:
        _active_caps.remove(cap)
    except ValueError:
        pass


def _make_capture(camera_id: int, retries: int = 3):
    cap: cv2.VideoCapture | None = None
    for attempt in range(retries):
        if cap is not None:
            cap.release()
            time.sleep(1.0)
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Camera {camera_id}: open failed (attempt {attempt + 1}/{retries})")
            continue
        # Validate with a real read — isOpened() alone can be a false positive.
        ret, _ = cap.read()
        if ret:
            break
        print(f"Camera {camera_id}: open ok but read failed "
              f"(attempt {attempt + 1}/{retries})")
    else:
        if cap is not None:
            cap.release()
        hint = "Try unplugging/replugging the camera."
        if sys.platform == "linux":
            hint += (" Or run 'sudo usbreset /dev/bus/usb/...' "
                     "to reset the USB device.")
        sys.exit(
            f"Cannot open camera {camera_id} after {retries} attempts. "
            + hint
        )

    _active_caps.append(cap)

    def capture() -> np.ndarray:
        # Flush buffered frames to get the latest one
        for _ in range(5):
            cap.grab()  # type: ignore[union-attr]
        ret, frame = cap.read()  # type: ignore[union-attr]
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
    try:
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
    finally:
        _release_capture(cap)
        cv2.destroyAllWindows()


def cmd_dry_run(args: argparse.Namespace) -> None:
    board = _board_from_args(args)
    detector = BoardDetector(board)
    capture_fn, cap = _make_capture(args.camera)

    # Use low speed for safety
    safe_vel = min(args.vel, 10.0)
    safe_acc = min(args.acc, 10.0)

    try:
        with DSR2Robot(container=args.container, vel=safe_vel, acc=safe_acc) as robot:
            initial_joints = robot.get_posj()
            center = _resolve_center_joints(args, robot)
            poses = generate_calibration_poses(
                center, n_poses=args.n_poses,
                wrist_range=args.wrist_range, arm_range=args.arm_range,
            )

            print(f"Dry run: {len(poses)} poses at {safe_vel} deg/s")
            print("Watch the robot and camera feed. Press 'q' to abort.\n")

            try:
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
            finally:
                print("Returning to initial position...")
                robot.move_to_joints(initial_joints)
    finally:
        _release_capture(cap)
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
        try:
            with DSR2Robot(container=args.container, vel=args.vel, acc=args.acc) as robot:
                initial_joints = robot.get_posj()
                center = _resolve_center_joints(args, robot)
                poses = generate_calibration_poses(
                    center, n_poses=args.n_images,
                    wrist_range=args.wrist_range, arm_range=args.arm_range,
                )
                try:
                    for i, joints in enumerate(poses):
                        robot.move_to_joints(joints)
                        time.sleep(args.settle_time)
                        frame = capture_fn()
                        if detector.detect(frame) is not None:
                            images.append(frame)
                            print(f"  [{i + 1}/{len(poses)}] detected ({len(images)} total)")
                        else:
                            print(f"  [{i + 1}/{len(poses)}] board not found, skipped")
                finally:
                    print("Returning to initial position...")
                    robot.move_to_joints(initial_joints)
        finally:
            _release_capture(cap)

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
            initial_joints = robot.get_posj()
            center = _resolve_center_joints(args, robot)
            poses = generate_calibration_poses(
                center, n_poses=args.n_poses,
                wrist_range=args.wrist_range, arm_range=args.arm_range,
            )
            try:
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
                print("Returning to initial position...")
                robot.move_to_joints(initial_joints)
    finally:
        _release_capture(cap)

    result.save(args.output)
    print(f"\nT_cam2gripper:\n{result.T_cam2gripper}")
    print(f"Saved {args.output}")


_JOINT_LABELS = ["J1", "J2", "J3", "J4", "J5", "J6"]
_TASK_LABELS = ["X", "Y", "Z", "W", "P", "R"]
_TASK_UNITS = ["mm", "mm", "mm", "deg", "deg", "deg"]


def _clear_screen() -> None:
    """Clear terminal screen (cross-platform)."""
    if sys.platform == "win32":
        import os
        os.system("cls")
    else:
        print("\033[2J\033[H", end="")


def _jog_print_state(
    task_mode: bool, selected_axis: int, step: float,
    joints: list[float], posx: list[float],
) -> None:
    """Print current jog state to terminal."""
    mode_text = "TASK" if task_mode else "JOINT"
    step_unit = "mm/deg" if task_mode else "deg"
    labels = _TASK_LABELS if task_mode else _JOINT_LABELS
    units = _TASK_UNITS if task_mode else ["deg"] * 6
    values = posx if task_mode else joints

    _clear_screen()
    print(f"=== Jog Mode [{mode_text}] ===  Step: {step:.1f} {step_unit}\n")
    for i in range(6):
        marker = " >" if i == selected_axis else "  "
        print(f"{marker} {labels[i]}: {values[i]:+8.2f} {units[i]}")
    print()
    if task_mode:
        print(f"  J: [{', '.join(f'{v:.1f}' for v in joints)}]")
    else:
        print(f"  X:{posx[0]:.1f} Y:{posx[1]:.1f} Z:{posx[2]:.1f}"
              f"  W:{posx[3]:.1f} P:{posx[4]:.1f} R:{posx[5]:.1f}")
    print("\n[Tab] joint/task  [1-6] axis  [a/d] jog  [w/s] step"
          "  [Enter] accept  [Esc] cancel")


def cmd_jog(args: argparse.Namespace) -> None:
    """Interactive jog mode: move robot with keyboard while watching camera."""
    use_camera = args.camera is not None

    cap = None
    if use_camera:
        board = _board_from_args(args)
        detector = BoardDetector(board)
        capture_fn, cap = _make_capture(args.camera)

    joint_step_sizes = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    task_step_sizes = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    joint_step_idx = 2   # 2.0 deg
    task_step_idx = 3    # 5.0 mm/deg

    selected_axis = 0    # 0-based index
    task_mode = False     # False=joint, True=task space

    try:
        with DSR2Robot(container=args.container, vel=args.vel, acc=args.acc) as robot:
            joints = _resolve_center_joints(args, robot)
            posx = robot.get_posx()

            if use_camera:
                _jog_loop_camera(
                    args, robot, detector, capture_fn, cap,  # type: ignore[possibly-undefined]
                    joints, posx,
                    joint_step_sizes, task_step_sizes,
                    joint_step_idx, task_step_idx,
                    selected_axis, task_mode,
                )
            else:
                _jog_loop_terminal(
                    robot, joints, posx,
                    joint_step_sizes, task_step_sizes,
                    joint_step_idx, task_step_idx,
                    selected_axis, task_mode,
                )
    finally:
        if cap is not None:
            _release_capture(cap)
            cv2.destroyAllWindows()


def _jog_loop_camera(
    args: argparse.Namespace,
    robot: DSR2Robot,
    detector: BoardDetector,
    capture_fn,
    cap,
    joints: list[float],
    posx: list[float],
    joint_step_sizes: list[float],
    task_step_sizes: list[float],
    joint_step_idx: int,
    task_step_idx: int,
    selected_axis: int,
    task_mode: bool,
) -> None:
    print("=== Jog Mode (camera) ===")
    print("Tab: joint/task | 1-6: select axis | a/d: jog -/+ | w/s: step")
    print("Enter: accept pose | Esc: cancel\n")

    moving = False
    move_error: str | None = None

    def _do_move(move_fn, move_arg, sync_fn, sync_target):
        """Run a blocking move in a background thread."""
        nonlocal moving, move_error
        try:
            move_fn(move_arg)
            sync_target[:] = sync_fn()
        except Exception as e:
            move_error = str(e)
        finally:
            moving = False

    while True:
        frame = capture_fn()
        result = detector.detect(frame)

        h_img, w_img = frame.shape[:2]
        if result is not None:
            corners, ids = result
            cv2.aruco.drawDetectedCornersCharuco(frame, corners, ids, (0, 255, 0))
            det_text = f"Detected: {len(ids)} corners"
            det_color = (0, 255, 0)
        else:
            det_text = "Board not detected"
            det_color = (0, 0, 255)

        if task_mode:
            labels, units, values = _TASK_LABELS, _TASK_UNITS, posx
            step = task_step_sizes[task_step_idx]
            mode_text, step_unit = "TASK", "mm/deg"
        else:
            labels, units, values = _JOINT_LABELS, ["deg"] * 6, joints
            step = joint_step_sizes[joint_step_idx]
            mode_text, step_unit = "JOINT", "deg"

        cv2.putText(frame, det_text,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, det_color, 2)

        status_text = f"[{mode_text}]  Step: {step:.1f} {step_unit}"
        if moving:
            status_text += "  ** Moving... **"
        cv2.putText(frame, status_text,
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        for i in range(6):
            color = (0, 255, 255) if i == selected_axis else (200, 200, 200)
            marker = ">" if i == selected_axis else " "
            txt = f"{marker} {labels[i]}: {values[i]:+8.2f} {units[i]}"
            cv2.putText(frame, txt,
                        (10, 95 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        if task_mode:
            sec = f"J:[{', '.join(f'{v:.1f}' for v in joints)}]"
        else:
            sec = (f"X:{posx[0]:.1f} Y:{posx[1]:.1f} Z:{posx[2]:.1f}  "
                   f"W:{posx[3]:.1f} P:{posx[4]:.1f} R:{posx[5]:.1f}")
        cv2.putText(frame, sec,
                    (10, h_img - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(frame,
                    "[Tab] joint/task  [1-6] axis  [a/d] jog  [w/s] step  [Enter] ok",
                    (10, h_img - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        cv2.imshow("dsr2-calibration jog", frame)
        key = cv2.waitKey(50) & 0xFF

        # Check for move errors from the background thread
        if move_error is not None:
            print(f"Move error: {move_error}")
            move_error = None

        if key == 27:
            print("Cancelled.")
            return
        elif key == 13 or key == ord("q"):
            break
        elif key == 9:
            task_mode = not task_mode
            selected_axis = 0
        elif ord("1") <= key <= ord("6"):
            selected_axis = key - ord("1")
        elif key in (ord("a"), ord("d")) and not moving:
            sign = -1.0 if key == ord("a") else 1.0
            moving = True
            if task_mode:
                posx[selected_axis] += sign * step
                threading.Thread(
                    target=_do_move,
                    args=(robot.move_to_posx, list(posx),
                          robot.get_posj, joints),
                    daemon=True,
                ).start()
            else:
                joints[selected_axis] += sign * step
                threading.Thread(
                    target=_do_move,
                    args=(robot.move_to_joints, list(joints),
                          robot.get_posx, posx),
                    daemon=True,
                ).start()
        elif key == ord("w"):
            if task_mode:
                task_step_idx = min(task_step_idx + 1, len(task_step_sizes) - 1)
            else:
                joint_step_idx = min(joint_step_idx + 1, len(joint_step_sizes) - 1)
        elif key == ord("s"):
            if task_mode:
                task_step_idx = max(task_step_idx - 1, 0)
            else:
                joint_step_idx = max(joint_step_idx - 1, 0)

    _jog_print_result(joints, posx)


def _get_key_unix() -> str:
    """Read a single keypress on Unix (handles escape sequences)."""
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        seq = sys.stdin.read(1)
        if seq == "[":
            return "\x1b[" + sys.stdin.read(1)
        return "\x1b"
    return ch


if sys.platform == "win32":
    import msvcrt

    def _get_key_win() -> str:
        """Read a single keypress on Windows via msvcrt."""
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            # Extended key — read second byte but we don't use arrow keys
            msvcrt.getwch()
            return ""
        if ch == "\x1b":
            return "\x1b"
        return ch

    def _jog_loop_terminal_win(
        robot: DSR2Robot,
        joints: list[float],
        posx: list[float],
        joint_step_sizes: list[float],
        task_step_sizes: list[float],
        joint_step_idx: int,
        task_step_idx: int,
        selected_axis: int,
        task_mode: bool,
    ) -> None:
        # msvcrt.getwch() already reads single keys without echo — no raw mode needed.
        accepted = _jog_terminal_mainloop(
            robot, joints, posx,
            joint_step_sizes, task_step_sizes,
            joint_step_idx, task_step_idx,
            selected_axis, task_mode,
            _get_key_win,
        )
        if accepted:
            _jog_print_result(joints, posx)
        else:
            print("\nCancelled.")


def _jog_loop_terminal(
    robot: DSR2Robot,
    joints: list[float],
    posx: list[float],
    joint_step_sizes: list[float],
    task_step_sizes: list[float],
    joint_step_idx: int,
    task_step_idx: int,
    selected_axis: int,
    task_mode: bool,
) -> None:
    if sys.platform == "win32":
        _jog_loop_terminal_win(
            robot, joints, posx,
            joint_step_sizes, task_step_sizes,
            joint_step_idx, task_step_idx,
            selected_axis, task_mode,
        )
    else:
        _jog_loop_terminal_unix(
            robot, joints, posx,
            joint_step_sizes, task_step_sizes,
            joint_step_idx, task_step_idx,
            selected_axis, task_mode,
        )


def _jog_terminal_mainloop(
    robot: DSR2Robot,
    joints: list[float],
    posx: list[float],
    joint_step_sizes: list[float],
    task_step_sizes: list[float],
    joint_step_idx: int,
    task_step_idx: int,
    selected_axis: int,
    task_mode: bool,
    get_key_fn,
) -> bool:
    """Shared jog loop logic. Returns True if accepted, False if cancelled."""
    step = joint_step_sizes[joint_step_idx]
    _jog_print_state(task_mode, selected_axis, step, joints, posx)

    while True:
        step = (task_step_sizes[task_step_idx] if task_mode
                else joint_step_sizes[joint_step_idx])

        key = get_key_fn()

        if key == "\x1b":  # Esc
            return False
        elif key in ("\r", "\n", "q"):  # Enter or q
            return True
        elif key == "\t":  # Tab
            task_mode = not task_mode
            selected_axis = 0
        elif key in "123456":
            selected_axis = int(key) - 1
        elif key in ("a", "d"):
            sign = -1.0 if key == "a" else 1.0
            if task_mode:
                posx[selected_axis] += sign * step
                robot.move_to_posx(posx)
                joints[:] = robot.get_posj()
            else:
                joints[selected_axis] += sign * step
                robot.move_to_joints(joints)
                posx[:] = robot.get_posx()
        elif key == "w":
            if task_mode:
                task_step_idx = min(task_step_idx + 1, len(task_step_sizes) - 1)
            else:
                joint_step_idx = min(joint_step_idx + 1, len(joint_step_sizes) - 1)
        elif key == "s":
            if task_mode:
                task_step_idx = max(task_step_idx - 1, 0)
            else:
                joint_step_idx = max(joint_step_idx - 1, 0)
        else:
            continue

        _jog_print_state(task_mode, selected_axis, step, joints, posx)


def _jog_loop_terminal_unix(
    robot: DSR2Robot,
    joints: list[float],
    posx: list[float],
    joint_step_sizes: list[float],
    task_step_sizes: list[float],
    joint_step_idx: int,
    task_step_idx: int,
    selected_axis: int,
    task_mode: bool,
) -> None:
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        accepted = _jog_terminal_mainloop(
            robot, joints, posx,
            joint_step_sizes, task_step_sizes,
            joint_step_idx, task_step_idx,
            selected_axis, task_mode,
            _get_key_unix,
        )
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    if accepted:
        _jog_print_result(joints, posx)
    else:
        print("\nCancelled.")


def _jog_print_result(joints: list[float], posx: list[float]) -> None:
    joints_str = ",".join(f"{v:.2f}" for v in joints)
    posx_str = ",".join(f"{v:.1f}" for v in posx)
    print(f"\nAccepted pose:")
    print(f"  joints: {joints_str}")
    print(f"  posx:   {posx_str}")
    print(f"\nUse with calibrate:")
    print(f"  dsr2-calibration calibrate -j {joints_str}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    from .calibration import HandEyeCalibrator, posx_to_matrix

    board = _board_from_args(args)
    detector = BoardDetector(board)
    capture_fn, cap = _make_capture(args.camera)

    # Check if GUI display is available
    show_gui = True
    try:
        cv2.namedWindow("dsr2-calibration", cv2.WINDOW_AUTOSIZE)
    except cv2.error:
        show_gui = False

    try:
        with DSR2Robot(container=args.container, vel=args.vel, acc=args.acc) as robot:
            initial_joints = robot.get_posj()
            center = _resolve_center_joints(args, robot)
            poses = generate_calibration_poses(
                center, n_poses=args.n_poses,
                wrist_range=args.wrist_range, arm_range=args.arm_range,
            )

            # Single pass: collect images + robot poses together
            print(f"Collecting data ({len(poses)} poses)...")
            images: list[np.ndarray] = []
            robot_poses: list[np.ndarray] = []
            aborted = False

            try:
                for i, joints in enumerate(poses):
                    robot.move_to_joints(joints)
                    time.sleep(args.settle_time)
                    frame = capture_fn()
                    display = frame.copy()
                    result = detector.detect(frame)
                    if result is not None:
                        corners, ids = result
                        images.append(frame)
                        robot_poses.append(posx_to_matrix(robot.get_posx()))
                        status = f"[{i + 1}/{len(poses)}] detected ({len(images)} total)"
                        if show_gui:
                            cv2.aruco.drawDetectedCornersCharuco(display, corners, ids, (0, 255, 0))
                            cv2.putText(display, status, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        print(f"  {status}")
                    else:
                        status = f"[{i + 1}/{len(poses)}] board not found, skipped"
                        if show_gui:
                            cv2.putText(display, status, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        print(f"  {status}")
                    if show_gui:
                        cv2.imshow("dsr2-calibration", display)
                        if cv2.waitKey(500) & 0xFF == ord("q"):
                            print("\nAborted by user.")
                            aborted = True
                            break
            finally:
                print("Returning to initial position...")
                robot.move_to_joints(initial_joints)
    finally:
        _release_capture(cap)
        if show_gui:
            cv2.destroyAllWindows()

    if aborted:
        sys.exit("Calibration aborted.")

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
    p.add_argument("-o", "--output", default=_timestamped_name("camera_intrinsics.npz"))
    p.set_defaults(func=cmd_calibrate_camera)

    # calibrate-transform
    p = sub.add_parser("calibrate-transform", help="Compute camera-to-gripper transform")
    _add_board_args(p)
    _add_pose_args(p)
    _add_robot_args(p)
    p.add_argument("-i", "--intrinsics", default="camera_intrinsics.npz")
    p.add_argument("-n", "--n-poses", type=int, default=15)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("-o", "--output", default=_timestamped_name("calibration_result.npz"))
    p.set_defaults(func=cmd_calibrate_transform)

    # jog
    p = sub.add_parser("jog", help="Interactive jog mode to find a good center pose")
    _add_board_args(p)
    _add_pose_args(p)
    _add_robot_args(p)
    p.add_argument("--camera", type=int, default=None,
                   help="camera ID (omit for terminal-only jog without camera)")
    p.set_defaults(func=cmd_jog)

    # calibrate
    p = sub.add_parser("calibrate", help="Full auto-calibration (camera + transform)")
    _add_board_args(p)
    _add_pose_args(p)
    _add_robot_args(p)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("-n", "--n-poses", type=int, default=20)
    p.add_argument("-o", "--output", default=_timestamped_name("calibration_result.npz"))
    p.set_defaults(func=cmd_calibrate)

    args = parser.parse_args()
    args.func(args)
