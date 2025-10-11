# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer):
        # timing
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 20
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        builder = newton.ModelBuilder()

        # ground
        builder.add_ground_plane()

        # 基础接触参数（SemiImplicit 的软接触模型）
        # 提高滚动稳定性，减少落地弹跳
        builder.default_shape_cfg.mu = 0.8

        # chassis dimensions and placement
        chassis_hx = 0.6
        chassis_hy = 0.25
        chassis_hz = 0.15

        wheel_radius = 0.30
        wheel_half_height = 0.05  # wheel thickness along its axis

        # chassis positioned above ground so wheels touch ground
        chassis_pos = wp.vec3(0.0, 0.0, wheel_radius + chassis_hz + 0.05)

        body_chassis = builder.add_body(xform=wp.transform(p=chassis_pos, q=wp.quat_identity()), key="chassis")

        # chassis collider & visual (box)
        builder.add_shape_box(
            body_chassis,
            hx=chassis_hx,
            hy=chassis_hy,
            hz=chassis_hz,
        )

        # wheel placement (four corners)
        # forward axis: +X, left-right: Y, up: Z
        x_offset = chassis_hx - 0.05
        y_offset = chassis_hy + 0.05
        # wheel centers should be at ground height (z = wheel_radius)
        # place wheels relative to chassis center with negative z offset
        z_wheel_rel = -(chassis_hz + 0.05)

        wheel_centers = [
            wp.vec3(+x_offset, +y_offset, z_wheel_rel),  # front-left
            wp.vec3(+x_offset, -y_offset, z_wheel_rel),  # front-right
            wp.vec3(-x_offset, +y_offset, z_wheel_rel),  # rear-left
            wp.vec3(-x_offset, -y_offset, z_wheel_rel),  # rear-right
        ]

        # orient cylinder so its axis aligns with Y (default cylinder axis is Z)
        cyl_rot_to_y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi * 0.5)

        wheel_bodies = []
        for i, rel in enumerate(wheel_centers):
            world_pos = chassis_pos + rel
            body_wheel = builder.add_body(xform=wp.transform(p=world_pos, q=wp.quat_identity()), key=f"wheel_{i}")
            wheel_bodies.append(body_wheel)

            # visual cylinder (no collision)
            vis_cfg = builder.default_shape_cfg.copy()
            vis_cfg.has_shape_collision = False
            vis_cfg.is_visible = True
            builder.add_shape_cylinder(
                body_wheel,
                xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=cyl_rot_to_y),
                radius=wheel_radius,
                half_height=wheel_half_height,
                cfg=vis_cfg,
                key=f"wheel_vis_{i}",
            )

            # collision cylinder for rolling contact
            col_cfg = builder.default_shape_cfg.copy()
            col_cfg.has_shape_collision = True
            col_cfg.is_visible = False
            col_cfg.mu = 0.9  # higher tire-ground friction for traction
            builder.add_shape_cylinder(
                body_wheel,
                xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=cyl_rot_to_y),
                radius=wheel_radius,
                half_height=wheel_half_height,
                cfg=col_cfg,
                key=f"wheel_col_{i}",
            )

            # 轮轴关节：父局部坐标系 Y 轴
            # 采用 TARGET_VELOCITY 驱动；初速度为 0，后续由键盘输入控制
            omega = 0.0  # 初始角速度为 0
            axis_cfg = newton.ModelBuilder.JointDofConfig(
                axis=newton.Axis.Y,
                mode=newton.JointMode.TARGET_VELOCITY,
                target=omega,
                target_ke=50.0,
                target_kd=0.0,
                friction=0.0,
            )

            parent_anchor = wp.transform(p=rel, q=wp.quat_identity())
            child_anchor = wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity())
            builder.add_joint_revolute(
                parent=body_chassis,
                child=body_wheel,
                parent_xform=parent_anchor,
                child_xform=child_anchor,
                axis=axis_cfg,
                collision_filter_parent=True,
                enabled=True,
                key=f"hinge_{i}",
            )

        # finalize model
        self.model = builder.finalize()

        # 软接触参数调优：提高阻尼、降低反弹系数，抑制落地弹跳
        self.model.soft_contact_ke = 2.0e3
        self.model.soft_contact_kd = 5.0e1
        self.model.soft_contact_mu = 0.8
        self.model.soft_contact_restitution = 0.02

        # solver：提高角阻尼，进一步降低弹跳与抖动
        self.solver = newton.solvers.SolverSemiImplicit(self.model, angular_damping=0.12)

        # states & control
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # viewer and initial FK
        # ---- 键盘控制与关节映射（需在预运行前初始化）----
        # 记录四个轮子关节的 DOF 起始索引，便于向 control.joint_target 写入
        qd_start = self.model.joint_qd_start.numpy()
        self.wheel_joint_ids = [self.model.joint_key.index(f"hinge_{i}") for i in range(4)]
        self.wheel_dof_indices = [int(qd_start[jid]) for jid in self.wheel_joint_ids]

        # 总 DOF 数量（最后一个 sentinel 即为总长度）
        self.joint_dof_count = int(qd_start[-1])

        # 车轮速度控制量（rad/s），初始为 0
        self.wheel_speed = 0.0
        self.max_wheel_speed = 30.0  # 上限（约 286 RPM）
        self.accel_rate = 6.0        # I 加速（rad/s^2）
        self.brake_rate = 8.0        # K 减速（rad/s^2）
        self.turn_gain = 4.0         # J/L 差速偏置（rad/s）

        self.viewer.set_model(self.model)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # 交互示例不使用 Warp 图捕获：需要每帧读取键盘输入
        # 如果启用图捕获，Python 侧的输入与控制更新会被绕过，导致按键无效
        self.graph = None

        # 注册一个简单的 UI 面板，显示按键状态与当前车轮速度
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.gui, position="stats")
        # 若 UI 后端不可用，提示安装依赖并说明切换热键
        try:
            if hasattr(self.viewer, "ui") and not getattr(self.viewer.ui, "is_available", False):
                print(
                    "提示: imgui-bundle 未安装，UI 面板不可用。安装: 'uv add imgui-bundle' 或 'pip install imgui-bundle'。按 H 切换 UI 显示。"
                )
        except Exception:
            pass

        # 控制台按键状态打印的节流计时器（UI 不可用时生效）
        self._last_key_print_t = -1.0

    def capture(self):
        # 说明：为保持交互性，本示例默认禁用图捕获。
        # 若需要离线/确定性回放，可手动启用并注意不要依赖键盘输入。
        self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # allow interactive forces (mouse picking, etc.)
            self.viewer.apply_forces(self.state_0)

            # ---- 键盘输入：IJKL ----
            # I：加速，K：减速（可反向），J：左转（左慢右快），L：右转（左快右慢）
            i_down = self.viewer.is_key_down("i")
            k_down = self.viewer.is_key_down("k")
            j_down = self.viewer.is_key_down("j")
            l_down = self.viewer.is_key_down("l")

            if i_down:
                self.wheel_speed += self.accel_rate * self.sim_dt
            if k_down:
                self.wheel_speed -= self.brake_rate * self.sim_dt

            # 限幅
            self.wheel_speed = max(-self.max_wheel_speed, min(self.wheel_speed, self.max_wheel_speed))

            turn = 0.0
            if j_down:
                turn -= self.turn_gain
            if l_down:
                turn += self.turn_gain

            left = self.wheel_speed - turn
            right = self.wheel_speed + turn

            # 控制台回退：在 UI 不可用时，周期性打印按键状态与速度，便于确认事件捕获
            try:
                ui_available = hasattr(self.viewer, "ui") and getattr(self.viewer.ui, "is_available", False)
            except Exception:
                ui_available = False
            if not ui_available:
                if self.sim_time - (self._last_key_print_t if self._last_key_print_t is not None else -1.0) > 0.5:
                    if i_down or k_down or j_down or l_down:
                        print(
                            f"[Keys] I={i_down} K={k_down} J={j_down} L={l_down} speed={self.wheel_speed:.2f}"
                        )
                    self._last_key_print_t = self.sim_time

            # 将目标速度写入四个轮子的 DOF 目标
            targets = [0.0] * self.joint_dof_count
            targets[self.wheel_dof_indices[0]] = left   # front-left
            targets[self.wheel_dof_indices[1]] = right  # front-right
            targets[self.wheel_dof_indices[2]] = left   # rear-left
            targets[self.wheel_dof_indices[3]] = right  # rear-right
            self.control.joint_target.assign(targets)

            # explicit collision update for non-MuJoCo solvers
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

            self.sim_time += self.sim_dt

    def step(self):
        # 始终逐步仿真以读取实时按键与更新控制
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def gui(self, ui):
        # 依赖 imgui_bundle，若不可用则跳过
        if not getattr(ui, "is_available", False):
            return
        imgui = ui.imgui

        imgui.begin("Car Controls & Status", flags=imgui.WindowFlags_.always_auto_resize)

        # 实时按键状态指示
        i_down = self.viewer.is_key_down("i")
        k_down = self.viewer.is_key_down("k")
        j_down = self.viewer.is_key_down("j")
        l_down = self.viewer.is_key_down("l")
        imgui.text(f"Keys: I={i_down} K={k_down} J={j_down} L={l_down}")

        # 当前参数与速度
        imgui.text(f"wheel_speed: {self.wheel_speed:.2f} rad/s")
        imgui.text(f"accel_rate: {self.accel_rate:.2f}  brake_rate: {self.brake_rate:.2f}")
        imgui.text(f"turn_gain: {self.turn_gain:.2f}  max: {self.max_wheel_speed:.1f}")

        # 可调滑块（便于现场调试）
        changed, val = imgui.slider_float("accel_rate", self.accel_rate, 0.0, 20.0, "%.2f")
        if changed:
            self.accel_rate = float(val)
        changed, val = imgui.slider_float("brake_rate", self.brake_rate, 0.0, 20.0, "%.2f")
        if changed:
            self.brake_rate = float(val)
        changed, val = imgui.slider_float("turn_gain", self.turn_gain, 0.0, 20.0, "%.2f")
        if changed:
            self.turn_gain = float(val)
        changed, val = imgui.slider_float("max_wheel_speed", self.max_wheel_speed, 5.0, 60.0, "%.1f")
        if changed:
            self.max_wheel_speed = float(val)

        imgui.end()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer)
    newton.examples.run(example, args)