"""
car_builder.py

功能：
- 接收 SimulationConfig 对象
- 创建 newton.ModelBuilder()
- 调用 rigidbody.create_rigid() 创建刚体
- 调用 softbody.create_soft() 创建软体（当前占位）
- 添加地面平面
- 返回 (builder, model)

并在 main 中提供示例运行：
from newton import examples
viewer, args = examples.init()
builder, model = build_car(cfg)
viewer.set_model(model)
examples.run(example_like_loop, args)
"""

from __future__ import annotations

import warp as wp
import newton

# 允许脚本直接运行时解析到 car1/utils
try:
    from utils.config_loader import SimulationConfig, load_all_configs
except ModuleNotFoundError:  # pragma: no cover - 运行脚本时的后备路径
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from utils.config_loader import SimulationConfig, load_all_configs

# 同目录内模块直接导入
from .rigidbody import create_rigid
from .softbody import create_soft
from .coupler import add_constraint


def build_car(cfg: SimulationConfig) -> tuple[newton.ModelBuilder, newton.Model]:
    builder = newton.ModelBuilder()

    builder.add_ground_plane()

    bodies_by_key: dict[str, int] = {}
    for rb in cfg.rigid_bodies:
        b = create_rigid(builder, rb, cfg.materials)
        k = rb.get("id")
        if isinstance(k, str):
            bodies_by_key[k] = b

    for sb in cfg.soft_bodies:
        create_soft(builder, sb)

    for c in cfg.constraints:
        add_constraint(builder, bodies_by_key, c)

    model = builder.finalize()
    return builder, model


class ExampleLike:
    def __init__(self, viewer: newton.viewer.ViewerBase, model: newton.Model):
        # 仿真参数
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.model = model

        # 求解器与状态
        self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # 初始化 FK（不在此调用 set_model，遵循 main 的使用方式）
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

    def step(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            # 允许查看器交互力
            self.viewer.apply_forces(self.state_0)
            # 碰撞与步进
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            # 状态交换
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    from newton import examples

    # 加载配置
    cfg = load_all_configs("config/")

    # 初始化查看器
    viewer, args = examples.init()

    # 构建小车
    builder, model = build_car(cfg)

    # 设置模型并运行示例循环
    viewer.set_model(model)
    example_like_loop = ExampleLike(viewer, model)
    examples.run(example_like_loop, args)
