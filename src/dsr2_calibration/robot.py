"""Doosan A0509 adapter.

Deploys a JSON-RPC bridge into the ROS 2 Docker container via
``docker cp`` + ``docker exec -i``.  No volume mounts or port
mappings required.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import posx_to_matrix

_BRIDGE_SRC = Path(__file__).parent / "bridge.py"
_BRIDGE_DEST = "/tmp/dsr2_calibration_bridge.py"

_ROS_SETUP = (
    "source /opt/ros/humble/setup.bash && "
    "source /ros2_ws/install/setup.bash && "
)


class DSR2Robot:
    """Controls the Doosan robot via a JSON-RPC bridge inside Docker."""

    def __init__(
        self,
        container: str = "ros-control",
        robot_id: str = "dsr01",
        robot_model: str = "a0509",
        vel: float = 30.0,
        acc: float = 30.0,
    ) -> None:
        self.vel = vel
        self.acc = acc

        # Deploy bridge script into the container
        try:
            cp = subprocess.run(
                ["docker", "cp", str(_BRIDGE_SRC), f"{container}:{_BRIDGE_DEST}"],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "docker not found. Is Docker installed and on PATH?"
            ) from None
        if cp.returncode != 0:
            raise RuntimeError(
                f"Cannot reach Docker container '{container}'. "
                f"Is it running?  (docker cp failed: {cp.stderr.strip()})"
            )

        # Start bridge process
        self._proc: subprocess.Popen[str] = subprocess.Popen(
            [
                "docker", "exec", "-i", container,
                "bash", "-c",
                f"{_ROS_SETUP}python3 -u {_BRIDGE_DEST}"
                f" --robot-id {robot_id} --robot-model {robot_model}",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
        )

        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        self._stdin = self._proc.stdin
        self._stdout = self._proc.stdout

        resp = self._readline()
        if not resp.get("ready"):
            raise RuntimeError(f"Bridge failed to start: {resp}")

    # ── low-level ─────────────────────────────────────────────────────

    def _readline(self) -> dict[str, Any]:
        line = self._stdout.readline()
        if not line:
            raise RuntimeError("Bridge process terminated unexpectedly")
        return json.loads(line)

    def _call(self, method: str, **params: object) -> dict[str, Any]:
        payload = json.dumps({"method": method, **params}) + "\n"
        self._stdin.write(payload)
        self._stdin.flush()
        resp = self._readline()
        if "error" in resp:
            raise RuntimeError(resp["error"])
        return resp

    # ── public API ────────────────────────────────────────────────────

    def move_to_joints(self, joints: list[float]) -> None:
        self._call("movej", joints=joints, vel=self.vel, acc=self.acc)

    def move_to_posx(self, posx: list[float]) -> None:
        self._call("movel", posx=posx, vel=self.vel, acc=self.acc)

    def get_posx(self) -> list[float]:
        return self._call("get_posx")["posx"]

    def get_posj(self) -> list[float]:
        return self._call("get_posj")["posj"]

    def ikin(self, posx: list[float]) -> list[float]:
        """Inverse kinematics: posx -> joint angles."""
        return self._call("ikin", posx=posx)["posj"]

    def get_pose_matrix(self) -> np.ndarray:
        return posx_to_matrix(self.get_posx())

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._call("exit")
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()

    def __enter__(self) -> DSR2Robot:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
