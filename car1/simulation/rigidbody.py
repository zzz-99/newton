"""
rigidbody.py

提供 `create_rigid(builder, spec)`，根据 `SimulationConfig.rigid_bodies` 中的条目
创建刚体和其几何形状。当前支持：box、cylinder、sphere。
"""

from __future__ import annotations

import warp as wp
import newton


def _vec3(v):
    if isinstance(v, (list, tuple)) and len(v) == 3:
        return wp.vec3(float(v[0]), float(v[1]), float(v[2]))
    return wp.vec3(0.0, 0.0, 0.0)


def _quat_from_spec(spec):
    # 支持 'quat': [x, y, z, w] 或 'rpy': [roll, pitch, yaw]，否则单位四元数
    q = spec.get("quat")
    if isinstance(q, (list, tuple)) and len(q) == 4:
        return wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    rpy = spec.get("rpy")
    if isinstance(rpy, (list, tuple)) and len(rpy) == 3:
        return wp.quat_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    return wp.quat_identity()


def _shape_cfg_from_materials(spec: dict, materials: dict | None) -> newton.ModelBuilder.ShapeConfig | None:
    name = (spec.get("material") or "default").lower()
    m = (materials or {}).get(name) or (materials or {}).get("default")
    if not isinstance(m, dict):
        return None
    cfg = newton.ModelBuilder.ShapeConfig()
    if "density" in m:
        cfg.density = float(m["density"])  # type: ignore[arg-type]
    if "ke" in m:
        cfg.ke = float(m["ke"])  # type: ignore[arg-type]
    if "kd" in m:
        cfg.kd = float(m["kd"])  # type: ignore[arg-type]
    if "mu" in m:
        cfg.mu = float(m["mu"])  # type: ignore[arg-type]
    if "restitution" in m:
        cfg.restitution = float(m["restitution"])  # type: ignore[arg-type]
    return cfg


def create_rigid(builder: newton.ModelBuilder, spec: dict, materials: dict | None = None) -> int:
    """
    根据规范字典 `spec` 创建刚体及其形状，返回刚体索引。

    期望字段示例：
    - id: 可选，字符串，作为 body key
    - pos: [x, y, z]
    - quat 或 rpy: 姿态
    - shape: 'box' | 'cylinder' | 'sphere'
      - box: size: [sx, sy, sz]
      - cylinder: radius: float, height: float
      - sphere: radius: float
    - mass: 可选，float
    """

    key = spec.get("id")
    pos = _vec3(spec.get("pos", (0.0, 0.0, 0.0)))
    rot = _quat_from_spec(spec)
    mass = spec.get("mass")

    # 创建刚体
    if mass is None:
        body = builder.add_body(xform=wp.transform(p=pos, q=rot), key=key)
    else:
        body = builder.add_body(xform=wp.transform(p=pos, q=rot), key=key, mass=float(mass))

    # 添加形状
    shape = (spec.get("shape") or "").lower()
    shape_cfg = _shape_cfg_from_materials(spec, materials)
    if shape == "box":
        size = spec.get("size", [1.0, 1.0, 1.0])
        hx, hy, hz = float(size[0]) / 2.0, float(size[1]) / 2.0, float(size[2]) / 2.0
        builder.add_shape_box(body, hx=hx, hy=hy, hz=hz, cfg=shape_cfg)
    elif shape == "cylinder":
        radius = float(spec.get("radius", 0.1))
        height = float(spec.get("height", 0.2))
        builder.add_shape_cylinder(body, radius=radius, half_height=height / 2.0, cfg=shape_cfg)
    elif shape == "sphere":
        radius = float(spec.get("radius", 0.1))
        builder.add_shape_sphere(body, radius=radius, cfg=shape_cfg)
    else:
        # 未知形状，创建一个小立方体占位
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=shape_cfg)

    return body


__all__ = ["create_rigid"]
