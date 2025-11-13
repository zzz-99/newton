from __future__ import annotations

import warp as wp
import newton


def vec3(v):
    if isinstance(v, (list, tuple)) and len(v) == 3:
        return wp.vec3(float(v[0]), float(v[1]), float(v[2]))
    return wp.vec3(0.0, 0.0, 0.0)


def axis_from_str(s: str | None):
    name = (s or "Y").upper()
    if name == "X":
        return newton.Axis.X
    if name == "Z":
        return newton.Axis.Z
    return newton.Axis.Y


__all__ = ["vec3", "axis_from_str"]
