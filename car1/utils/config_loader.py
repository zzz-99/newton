"""
配置加载器（Newton + Warp 刚柔耦合小车 / car1）

读取三个 JSON 文件：
- scene.json：包含刚体、软体及（可选）约束定义
- constraints.json（或 constraint.json）：包含 XPBD 约束
- sim_params.json：仿真步长与重力

返回 `SimulationConfig` 数据类，属性：
- rigid_bodies, soft_bodies, constraints, sim_params

文件读取使用 `pathlib.Path`。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

JSONDict = Dict[str, Any]
JSONList = List[JSONDict]


@dataclass
class SimulationConfig:
    """汇总仿真所需的全部配置。"""

    rigid_bodies: JSONList = field(default_factory=list)
    soft_bodies: JSONList = field(default_factory=list)
    constraints: JSONList = field(default_factory=list)
    sim_params: JSONDict = field(default_factory=dict)
    materials: JSONDict = field(default_factory=dict)


def _read_json_object(path: Path) -> JSONDict:
    """读取并解析 JSON 文件，要求根为对象（dict）。"""
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
        raise TypeError(f"{path} 根元素需为对象（dict），当前为: {type(data).__name__}")
    return data


def _read_json_any(path: Path) -> Any:
    """读取并解析 JSON 文件，返回任意类型（允许 dict 或 list）。"""
    if not path.exists():
        raise FileNotFoundError(f"未找到配置文件: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"期望文件但得到文件夹: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败: {path}: {e}") from e


def _ensure_list(obj: Any, name: str) -> JSONList:
    """确保对象为列表；None 返回空列表，其他类型抛错。"""
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj  # type: ignore[return-value]
    raise TypeError(f"字段 '{name}' 需为数组（list），当前为: {type(obj).__name__}")


def _ensure_dict(obj: Any, name: str) -> JSONDict:
    """确保对象为字典；None 返回空字典，其他类型抛错。"""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"字段 '{name}' 需为对象（dict），当前为: {type(obj).__name__}")


def _resolve_constraints_path(base: Path) -> Optional[Path]:
    """在目录中查找约束文件，优先 `constraints.json`，其次 `constraint.json`。"""
    c1 = base / "constraints.json"
    c2 = base / "constraint.json"
    if c1.exists() and c1.is_file():
        return c1
    if c2.exists() and c2.is_file():
        return c2
    return None


def load_all_configs(config_dir: Union[str, Path]) -> SimulationConfig:
    """
    从配置目录读取 `scene.json`, `constraints.json`（或 `constraint.json`）, `sim_params.json`
    并返回 `SimulationConfig`。
    """
    base = Path(config_dir)

    scene_path = base / "scene.json"
    sim_params_path = base / "sim_params.json"
    material_path = base / "material.json"
    constraints_path = _resolve_constraints_path(base)

    # 读取 scene.json
    scene = _read_json_object(scene_path)
    rigid_bodies = _ensure_list(scene.get("rigid_bodies"), "rigid_bodies")
    soft_bodies = _ensure_list(scene.get("soft_bodies"), "soft_bodies")

    # 读取约束：优先独立 constraints 文件；没有则回退到 scene.json 中的字段
    if constraints_path is not None:
        constraints_data = _read_json_any(constraints_path)
        if isinstance(constraints_data, list):
            constraints = constraints_data  # type: ignore[assignment]
        elif isinstance(constraints_data, dict):
            constraints = _ensure_list(constraints_data.get("constraints"), "constraints")
        else:
            raise TypeError(
                f"约束文件根类型需为 list 或 dict，当前为: {type(constraints_data).__name__}"
            )
    else:
        constraints = _ensure_list(scene.get("constraints"), "constraints")

    # 读取仿真参数
    sim_params = _read_json_object(sim_params_path)
    sim_params = _ensure_dict(sim_params, "sim_params")

    # 读取材料库（可选）
    materials: JSONDict = {}
    if material_path.exists() and material_path.is_file():
        materials = _read_json_object(material_path)
        materials = _ensure_dict(materials, "materials")

    return SimulationConfig(
        rigid_bodies=rigid_bodies,
        soft_bodies=soft_bodies,
        constraints=constraints,
        sim_params=sim_params,
        materials=materials,
    )


__all__ = ["SimulationConfig", "load_all_configs"]
if __name__ == "__main__":
    cfg = load_all_configs("config/")
    print(cfg.rigid_bodies)
