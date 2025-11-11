"""
配置加载器（Newton + Warp 刚柔耦合小车）

该模块用于从指定目录读取以下配置文件：
- scene.json：包含刚体、软体及约束定义
- material.json：包含物理参数
- sim_params.json：仿真步长与重力

并汇总为一个 `SimulationConfig` 数据类对象返回。
文件读取使用 `pathlib.Path`。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union


# 类型别名，便于阅读
JSONDict = Dict[str, Any]
JSONList = List[JSONDict]


@dataclass
class SimulationConfig:
    """
    汇总仿真所需的全部配置。

    属性：
    - rigid_bodies: 刚体定义列表
    - soft_bodies: 软体定义列表
    - constraints: 约束定义列表
    - materials: 物理参数字典
    - sim_params: 仿真参数字典（如步长、重力等）
    """

    rigid_bodies: JSONList = field(default_factory=list)
    soft_bodies: JSONList = field(default_factory=list)
    constraints: JSONList = field(default_factory=list)
    materials: JSONDict = field(default_factory=dict)
    sim_params: JSONDict = field(default_factory=dict)


def _read_json(path: Path) -> JSONDict:
    """读取并解析 JSON 文件，返回字典。

    说明：
    - 若文件不存在或为目录，将抛出异常。
    - 若 JSON 解析失败，将抛出 ValueError。
    """
    if not path.exists():
        raise FileNotFoundError(f"未找到配置文件: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"期望文件但得到文件夹: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败: {path}: {e}") from e

    if not isinstance(data, dict):
        # 为保持接口一致性，仅接受字典根；若需要列表可在上层定义
        raise TypeError(f"{path} 根元素需为对象（JSON 字典），当前为: {type(data).__name__}")

    return data


def _ensure_list(obj: Any, name: str) -> JSONList:
    """确保对象为列表；None 返回空列表，其他不合法类型抛错。"""
    if obj is None:
        return []
    if isinstance(obj, list):
        # 允许列表元素为任意 JSON；此处不做更细校验
        return obj  # type: ignore[return-value]
    raise TypeError(f"scene.json 字段 '{name}' 需为数组（list），当前为: {type(obj).__name__}")


def _ensure_dict(obj: Any, name: str) -> JSONDict:
    """确保对象为字典；None 返回空字典，其他不合法类型抛错。"""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"字段 '{name}' 需为对象（dict），当前为: {type(obj).__name__}")


def load_all_configs(config_dir: Union[str, Path]) -> SimulationConfig:
    """
    从配置目录读取 `scene.json`, `material.json`, `sim_params.json` 并返回 `SimulationConfig`。

    参数：
    - config_dir: 包含三个 JSON 的目录路径（str 或 Path）

    返回：
    - SimulationConfig 数据类对象
    """
    base = Path(config_dir)

    scene_path = base / "scene.json"
    material_path = base / "material.json"
    sim_params_path = base / "sim_params.json"

    scene_data = _read_json(scene_path)
    materials = _read_json(material_path)
    sim_params = _read_json(sim_params_path)

    rigid_bodies = _ensure_list(scene_data.get("rigid_bodies"), "rigid_bodies")
    soft_bodies = _ensure_list(scene_data.get("soft_bodies"), "soft_bodies")
    constraints = _ensure_list(scene_data.get("constraints"), "constraints")

    materials = _ensure_dict(materials, "materials")
    sim_params = _ensure_dict(sim_params, "sim_params")

    return SimulationConfig(
        rigid_bodies=rigid_bodies,
        soft_bodies=soft_bodies,
        constraints=constraints,
        materials=materials,
        sim_params=sim_params,
    )


__all__ = ["SimulationConfig", "load_all_configs"]