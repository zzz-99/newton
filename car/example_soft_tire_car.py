import math
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.examples


def ring_indices(n_theta: int, n_phi: int, base_idx: int):
    idx = [[0] * n_phi for _ in range(n_theta)]
    c = base_idx
    for i in range(n_theta):
        for j in range(n_phi):
            idx[i][j] = c
            c += 1
    return idx


def add_tire_tet_ring(
    builder: newton.ModelBuilder,
    center: wp.vec3,
    rot: wp.quat,
    R_major: float,
    half_width: float,
    radial_thickness: float,
    n_theta: int = 48,
    n_radial: int = 6,
    n_width: int = 6,
    density: float = 500.0,
    particle_radius: float = 0.02,
    k_mu: float = 3.0e4,
    k_lambda: float = 3.0e4,
    k_damp: float = 8.0e2,
):
    """
    构建一个围绕局部 Y 轴的“矩形截面”环形四面体软体轮胎：
    - 通过 (theta, radial, width) 规则网格将每个六面体划分为 5 个四面体
    - 使用 Lame 参数 (k_mu, k_lambda) 与阻尼 k_damp 控制材料硬度与阻尼
    - `density` 用于近似每个粒子的质量（总体积/粒子数）
    """

    def local_to_world(p):
        v = wp.vec3(p[0], p[1], p[2])
        w = wp.quat_rotate(rot, v)
        return [center[0] + w[0], center[1] + w[1], center[2] + w[2]]

    # 构建节点网格：theta 环向、radial 径向厚度、width 轮胎宽度方向（Y）
    # 采用矩形截面（radial × width），中心线半径为 R_major
    pos = []
    vel = []
    node_index = [[[0 for _ in range(n_width + 1)] for _ in range(n_radial + 1)] for _ in range(n_theta)]

    for it in range(n_theta):
        theta = (2.0 * math.pi) * (it / n_theta)
        nx = math.cos(theta)
        nz = math.sin(theta)
        centerline = np.array([R_major * nx, 0.0, R_major * nz])
        b1 = np.array([nx, 0.0, nz])      # 径向
        b2 = np.array([0.0, 1.0, 0.0])    # 宽度方向（Y）

        for ir in range(n_radial + 1):
            r = (-radial_thickness * 0.5) + radial_thickness * (ir / n_radial)
            for iy in range(n_width + 1):
                y = (-half_width) + (2.0 * half_width) * (iy / n_width)
                lp = centerline + (b1 * r) + (b2 * y)
                node_index[it][ir][iy] = len(pos)
                pos.append(local_to_world(lp))
                vel.append([0.0, 0.0, 0.0])

    start_idx = builder.particle_count
    # 体积近似：矩形截面（2*half_width × radial_thickness） × 圆周长（2πR_major）
    approx_volume = (2.0 * half_width) * radial_thickness * (2.0 * math.pi * R_major)
    total_mass = density * approx_volume
    mass_per_particle = total_mass / len(pos)
    builder.add_particles(pos, vel, [mass_per_particle] * len(pos), radius=[particle_radius] * len(pos))

    # 将每个六面体单元划分为 5 个四面体（theta wrap-around）
    def add_cell_tets(a000, a100, a010, a110, a001, a101, a011, a111):
        builder.add_tetrahedron(a000, a100, a110, a111, k_mu=k_mu, k_lambda=k_lambda, k_damp=k_damp)
        builder.add_tetrahedron(a000, a110, a010, a111, k_mu=k_mu, k_lambda=k_lambda, k_damp=k_damp)
        builder.add_tetrahedron(a000, a010, a011, a111, k_mu=k_mu, k_lambda=k_lambda, k_damp=k_damp)
        builder.add_tetrahedron(a000, a011, a001, a111, k_mu=k_mu, k_lambda=k_lambda, k_damp=k_damp)
        builder.add_tetrahedron(a000, a001, a101, a111, k_mu=k_mu, k_lambda=k_lambda, k_damp=k_damp)

    for it in range(n_theta):
        it1 = (it + 1) % n_theta
        for ir in range(n_radial):
            for iy in range(n_width):
                v000 = start_idx + node_index[it][ir][iy]
                v100 = start_idx + node_index[it1][ir][iy]
                v010 = start_idx + node_index[it][ir + 1][iy]
                v110 = start_idx + node_index[it1][ir + 1][iy]
                v001 = start_idx + node_index[it][ir][iy + 1]
                v101 = start_idx + node_index[it1][ir][iy + 1]
                v011 = start_idx + node_index[it][ir + 1][iy + 1]
                v111 = start_idx + node_index[it1][ir + 1][iy + 1]
                add_cell_tets(v000, v100, v010, v110, v001, v101, v011, v111)

    return start_idx, len(pos)


@dataclass
class SoftTireCar:
    def __init__(self, viewer):
        self.viewer = viewer

        builder = newton.ModelBuilder()

        # ground
        builder.add_ground_plane()

        # parameters
        chassis_hx, chassis_hy, chassis_hz = 0.8, 0.5, 0.2
        wheel_radius_rim = 0.25
        # 轮胎几何与间隙：避免胎体粒子与轮辋初始相交导致巨大反推力
        tire_radius = 0.32
        tire_thickness = tire_radius - wheel_radius_rim
        rim_clearance = 0.02  # 与轮辋的径向间隙（内半径 - 轮辋外半径）
        wheel_half_height = 0.18 * 0.5
        wheel_mass = 10.0

        # chassis centered so tires can touch ground
        # 根据设定间隙计算实际胎体外半径，确保底盘高度匹配
        tire_outer_radius = (wheel_radius_rim + tire_thickness) + rim_clearance
        chassis_pos = wp.vec3(0.0, 0.0, tire_outer_radius + chassis_hz + 0.05)
        body_chassis = builder.add_body(xform=wp.transform(p=chassis_pos, q=wp.quat_identity()), key="chassis")
        builder.add_shape_box(body_chassis, hx=chassis_hx, hy=chassis_hy, hz=chassis_hz)

        # rim rotation to align cylinder axis with Y
        cyl_rot_to_y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi * 0.5)

        # wheel centers relative to chassis
        x_offset = chassis_hx - 0.05
        y_offset = chassis_hy + 0.05
        z_wheel_rel = -(chassis_hz + 0.05)
        wheel_centers = [
            wp.vec3(+x_offset, +y_offset, z_wheel_rel),
            wp.vec3(+x_offset, -y_offset, z_wheel_rel),
            wp.vec3(-x_offset, +y_offset, z_wheel_rel),
            wp.vec3(-x_offset, -y_offset, z_wheel_rel),
        ]

        self.wheel_bodies = []
        self.tires = []
        self.tire_particles_radius = 0.02

        # friction tuning
        mu_rim = 1.0
        mu_ground = 1.2

        for i, rel in enumerate(wheel_centers):
            world_pos = chassis_pos + rel
            body_wheel = builder.add_body(xform=wp.transform(p=world_pos, q=wp.quat_identity()), key=f"wheel_{i}")
            self.wheel_bodies.append(body_wheel)

            # rim collision (smaller than tire)
            rim_cfg = builder.default_shape_cfg.copy()
            rim_cfg.mu = mu_rim
            rim_cfg.is_visible = False
            builder.add_shape_cylinder(
                body_wheel,
                xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=cyl_rot_to_y),
                radius=wheel_radius_rim,
                half_height=wheel_half_height,
                cfg=rim_cfg,
                key=f"wheel_rim_{i}",
            )

            # rim visual
            vis_cfg = builder.default_shape_cfg.copy()
            vis_cfg.has_shape_collision = False
            vis_cfg.is_visible = True
            builder.add_shape_cylinder(
                body_wheel,
                xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=cyl_rot_to_y),
                radius=wheel_radius_rim,
                half_height=wheel_half_height,
                cfg=vis_cfg,
                key=f"wheel_vis_{i}",
            )

            # revolute joint on Y
            axis_cfg = newton.ModelBuilder.JointDofConfig(
                axis=newton.Axis.Y,
                mode=newton.JointMode.TARGET_VELOCITY,
                target=0.0,
                # 降低关节目标刚度/阻尼，避免过强驱动引发数值爆发
                target_ke=80.0,
                target_kd=10.0,
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

            # tetrahedral volumetric tire ring
            tire_start, tire_count = add_tire_tet_ring(
                builder,
                center=world_pos,
                rot=wp.quat_identity(),
                # 设置环中心半径，使内半径 = 轮辋半径 + 间隙
                # R_major - radial_thickness/2 = wheel_radius_rim + rim_clearance
                # => R_major = wheel_radius_rim + rim_clearance + radial_thickness/2
                R_major=(wheel_radius_rim + rim_clearance + (tire_thickness * 0.5)),
                half_width=wheel_half_height,
                radial_thickness=tire_thickness,
                # 将每个胎体的六面体单元数降至 ~100（每单元5个四面体），
                # 四个轮胎总计约 2000 个四面体
                n_theta=20,
                n_radial=5,
                n_width=1,
                density=450.0,
                particle_radius=self.tire_particles_radius,
                k_mu=1.2e4,
                k_lambda=1.2e4,
                k_damp=2.5e3,
            )
            self.tires.append((tire_start, tire_count))

        # finalize model and solver
        self.model = builder.finalize()
        # 使用 XPBD 求解器以增强软体与接触的稳定性
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=12,
            soft_body_relaxation=0.6,
            soft_contact_relaxation=0.7,
            joint_angular_relaxation=0.5,
            joint_linear_relaxation=0.7,
            angular_damping=0.12,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # contact params
        # XPBD 下以松弛参数为主；降低摩擦系数与接触半径，并抑制反弹
        self.model.soft_contact_mu = 0.6
        self.model.soft_contact_ke = 4.0e3
        self.model.soft_contact_kd = 4.0e2
        self.model.soft_contact_radius = max(0.75 * self.tire_particles_radius, 0.005)
        self.model.soft_contact_restitution = 0.0
        self.model.particle_max_velocity = 8.0

        # wheel drive
        self.wheel_speed = 0.0
        self.max_wheel_speed = 12.0
        self.accel_rate = 3.0
        self.brake_rate = 5.0

        # viewer
        self.viewer.set_model(self.model)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        # 明确禁用 Warp 图捕获，确保每帧读取键盘输入
        self.graph = None
        # 默认不让基础渲染管线显示粒子，由我们自定义颜色输出
        self.viewer.show_particles = False
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.gui, position="stats")

    def step(self, time=0.0):
        # keyboard I/K for accelerate/brake, symmetric for all wheels
        i_down = self.viewer.is_key_down("i")
        k_down = self.viewer.is_key_down("k")
        self.wheel_speed += (self.accel_rate if i_down else 0.0) * (1.0 / 60.0)
        self.wheel_speed -= (self.brake_rate if k_down else 0.0) * (1.0 / 60.0)
        self.wheel_speed = max(-self.max_wheel_speed, min(self.max_wheel_speed, self.wheel_speed))

        # set target velocity on each wheel DOF
        qd_start = self.model.joint_qd_start.numpy()
        wheel_joint_ids = [self.model.joint_key.index(f"hinge_{i}") for i in range(4)]
        wheel_dof_indices = [int(qd_start[jid]) for jid in wheel_joint_ids]
        targets = [0.0] * int(qd_start[-1])
        for dof_idx in wheel_dof_indices:
            targets[dof_idx] = self.wheel_speed
        self.control.joint_target.assign(targets)

        # substepping
        substeps = 30
        dt = 1.0 / 60.0 / substeps
        for _ in range(substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def render(self):
        # mirror the pattern used in rigid car example
        self.viewer.begin_frame(0.0)
        self.viewer.log_state(self.state_0)
        # 自定义胎体粒子为深色，增强与轮辋的区分度
        try:
            tire_color = wp.vec3(0.06, 0.06, 0.06)
            colors = wp.full(shape=self.model.particle_count, value=tire_color, device=self.viewer.device)
            self.viewer.log_points(
                name="/custom/tires",
                points=self.state_0.particle_q,
                radii=self.model.particle_radius,
                colors=colors,
                hidden=False,
            )
        except Exception:
            # 若后端不支持 points（如某些非 GL viewer），忽略自定义颜色
            pass
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def gui(self, ui):
        # imgui-bundle style UI
        if not getattr(ui, "is_available", False):
            return
        imgui = ui.imgui
        imgui.begin("Soft Tire Car", flags=imgui.WindowFlags_.always_auto_resize)
        imgui.text(f"Wheel speed: {self.wheel_speed:.2f} rad/s")
        imgui.text("I/K 加速/刹车；目前未实现转向")
        imgui.end()


def main():
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = SoftTireCar(viewer)
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()