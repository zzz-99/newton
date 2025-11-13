"""
softbody.py

提供 `create_soft(builder, spec)` 的占位实现。
后续可根据 `spec` 添加软体（如网格、体素、粒子）。
"""

from __future__ import annotations

import newton


def create_soft(builder: newton.ModelBuilder, spec: dict):
    """
    占位：当前不创建软体实体，仅保留接口。
    可在后续根据 spec 结构调用 `builder.add_soft_*` 等 API。
    """
    # TODO: 根据 spec 增强软体创建，如 add_soft_grid / add_soft_mesh 等
    return None


__all__ = ["create_soft"]