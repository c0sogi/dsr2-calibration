#!/usr/bin/env python3
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportMissingTypeArgument=false
"""DSR_ROBOT2 stdin/stdout JSON-RPC bridge.

Automatically deployed and started by ``DSR2Robot``.
Runs inside the Doosan ROS 2 Docker container — no external dependencies
beyond what the container already provides.

Protocol: one JSON object per line on stdin → one JSON object per line on
stdout.  All ROS / DSR_ROBOT2 log output is redirected to stderr so it
never corrupts the protocol stream.
"""

import json
import os
import sys

# ── Redirect stdout → stderr BEFORE any ROS / DSR imports ───────────
_proto_out = os.fdopen(os.dup(sys.stdout.fileno()), "w")
sys.stdout = sys.stderr


def _respond(data: dict) -> None:
    _proto_out.write(json.dumps(data) + "\n")
    _proto_out.flush()


def main(robot_id: str = "dsr01", robot_model: str = "a0509") -> None:
    # ── ROS 2 + DSR_ROBOT2 initialisation ────────────────────────────
    import rclpy  # type: ignore[import-not-found]
    import DR_init  # type: ignore[import-not-found]

    DR_init.__dsr__id = robot_id
    DR_init.__dsr__model = robot_model
    rclpy.init()
    node = rclpy.create_node("calibration_bridge", namespace=robot_id)
    DR_init.__dsr__node = node

    from DSR_ROBOT2 import (  # type: ignore[import-not-found]
        ROBOT_MODE_AUTONOMOUS,
        get_current_posj,
        get_current_posx,
        ikin,
        movej,
        movel,
        set_robot_mode,
    )

    set_robot_mode(ROBOT_MODE_AUTONOMOUS)
    _respond({"ready": True})

    # ── Command loop ─────────────────────────────────────────────────
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            method = req.get("method")

            if method == "movej":
                movej(req["joints"], vel=req.get("vel", 30), acc=req.get("acc", 30))
                _respond({"ok": True})
            elif method == "movel":
                movel(req["posx"], vel=req.get("vel", 30), acc=req.get("acc", 30))
                _respond({"ok": True})
            elif method == "get_posx":
                posx_val, sol = get_current_posx()
                _respond({"posx": list(posx_val), "solution_space": sol})
            elif method == "get_posj":
                posj_val = get_current_posj()
                _respond({"posj": list(posj_val)})
            elif method == "ikin":
                posj_val = ikin(req["posx"], req.get("sol_space", 0), req.get("ref", 0))
                _respond({"posj": list(posj_val)})
            elif method == "ping":
                _respond({"pong": True})
            elif method == "exit":
                _respond({"ok": True})
                break
            else:
                _respond({"error": f"unknown method: {method}"})

        except Exception as e:
            _respond({"error": str(e)})

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--robot-id", default="dsr01")
    p.add_argument("--robot-model", default="a0509")
    a = p.parse_args()
    main(a.robot_id, a.robot_model)
