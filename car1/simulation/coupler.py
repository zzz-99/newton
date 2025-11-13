from __future__ import annotations

import warp as wp
import newton


def _vec3(v):
    if isinstance(v, (list, tuple)) and len(v) == 3:
        return wp.vec3(float(v[0]), float(v[1]), float(v[2]))
    return wp.vec3(0.0, 0.0, 0.0)


def _axis_from_str(s: str | None):
    name = (s or "Y").upper()
    if name == "X":
        return newton.Axis.X
    if name == "Z":
        return newton.Axis.Z
    return newton.Axis.Y


def add_constraint(builder: newton.ModelBuilder, bodies_by_key: dict[str, int], spec: dict) -> int | None:
    t = (spec.get("type") or "").lower()
    parent_key = spec.get("parent")
    child_key = spec.get("child")
    if not isinstance(parent_key, str) or not isinstance(child_key, str):
        return None
    parent = bodies_by_key.get(parent_key, -1)
    child = bodies_by_key.get(child_key, -1)

    parent_xform = wp.transform(p=_vec3(spec.get("parent_pos", (0.0, 0.0, 0.0))))
    child_xform = wp.transform(p=_vec3(spec.get("child_pos", (0.0, 0.0, 0.0))))
    key = spec.get("key") or spec.get("id")

    if t == "fixed":
        return builder.add_joint_fixed(parent=parent, child=child, parent_xform=parent_xform, child_xform=child_xform, key=key)

    if t == "revolute":
        axis = _axis_from_str(spec.get("axis"))
        mode = newton.JointMode.TARGET_VELOCITY
        target = float(spec.get("target", 0.0))
        target_ke = float(spec.get("target_ke", 50.0))
        target_kd = float(spec.get("target_kd", 0.0))
        friction = float(spec.get("friction", 0.0))
        cfg = newton.ModelBuilder.JointDofConfig(axis=axis, target=target, target_ke=target_ke, target_kd=target_kd, mode=mode, friction=friction)
        return builder.add_joint_revolute(parent=parent, child=child, parent_xform=parent_xform, child_xform=child_xform, axis=cfg, key=key)

    return None


__all__ = ["add_constraint"]
