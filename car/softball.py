#!/usr/bin/env python3

import warp as wp
import newton


class SoftballDemo:
    def __init__(self):
        # 仿真参数
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 30
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # 软体球参数（使用立方网格近似球体）
        self.ball_radius = 0.5
        self.ball_resolution = 16
        self.ball_density = 1000.0

        # 橡胶材质参数（拉梅参数）
        young_modulus = 1.5e5
        poisson_ratio = 0.3
        self.k_mu = 0.5 * young_modulus / (1.0 + poisson_ratio)
        self.k_lambda = young_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        self.k_damp = 0.06

        # 创建模型与求解器
        self.model = self._create_model()
        # 使用半隐式求解器，当前对软体支持更完备，数值稳定性更好
        self.solver = newton.solvers.SolverSemiImplicit(self.model)

        # 初始化状态与接触
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        # 首次接触生成（不使用过大的边距）
        self.contacts = self.model.collide(self.state_0, soft_contact_margin=0.05)

        # 查看器
        self.viewer = newton.viewer.ViewerGL(headless=False)
        self.viewer.set_model(self.model)
        # XPBD 属于最大坐标系求解器，需要先评估正向运动学以初始化形状/刚体位姿
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        # 可选：设置相机参数（如果支持）
        try:
            self.viewer.set_camera(pos=wp.vec3(6.0, -6.0, 4.0), pitch=-20.0, yaw=135.0)
        except Exception:
            pass

    def _create_model(self) -> newton.Model:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        builder.default_particle_radius = 0.03

        # 软体球位置（Z为竖直方向）
        pos = wp.vec3(0.0, 0.0, 2.0)
        rot = wp.quat_identity()
        vel = wp.vec3(0.0, 0.0, 0.0)

        # 使用立方体网格近似球体体积
        dim = self.ball_resolution
        cell = (self.ball_radius * 2.0) / float(dim)
        builder.add_soft_grid(
            pos=pos,
            rot=rot,
            vel=vel,
            dim_x=dim,
            dim_y=dim,
            dim_z=dim,
            cell_x=cell,
            cell_y=cell,
            cell_z=cell,
            density=self.ball_density,
            k_mu=self.k_mu,
            k_lambda=self.k_lambda,
            k_damp=self.k_damp,
            fix_bottom=False,
            fix_top=False,
        )

        # 地面
        ke, kf, kd, mu = 2.0e3, 0.0, 2.0e1, 0.4
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kf=kf, kd=kd, mu=mu))

        model = builder.finalize(requires_grad=False)

        # 软接触参数与反弹系数
        model.soft_contact_ke = ke
        model.soft_contact_kf = kf
        model.soft_contact_kd = kd
        model.soft_contact_mu = mu
        model.soft_contact_restitution = 0.4

        return model

    def step(self):
        # 每帧执行多个子步
        for _ in range(self.sim_substeps):
            # 清除上一子步累积的力，避免数值发散
            self.state_0.clear_forces()
            # 每子步重新生成软接触，避免接触穿透与不稳定
            self.contacts = self.model.collide(self.state_0, soft_contact_margin=0.1)
            # 允许查看器对状态施加交互力（拾取等），若无则安全为空操作
            self.viewer.apply_forces(self.state_0)
            # 步进求解器
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            # 交换状态
            self.state_0, self.state_1 = self.state_1, self.state_0

        # 渲染
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

        self.sim_time += self.frame_dt

    def run(self):
        print("开始仿真：软体球垂直下落并与地面反弹")
        while self.viewer.is_running():
            self.step()
        self.viewer.close()
        print("仿真结束")


def main():
    demo = SoftballDemo()
    demo.run()


if __name__ == "__main__":
    main()