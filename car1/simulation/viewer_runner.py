from __future__ import annotations

import warp as wp
import newton

from .solver import XPBDSolver
from .integrator import Integrator


class RigidCarRunner:
    def __init__(self, viewer: newton.viewer.ViewerBase, model: newton.Model, sim_params: dict):
        iterations = int(sim_params.get("solver_iterations", 10))
        substeps = int(sim_params.get("substeps", 4))
        self.dt = float(sim_params.get("time_step", 1.0 / 60.0))

        self.viewer = viewer
        self.model = model
        self.solver = XPBDSolver(model, iterations=iterations)
        self.integrator = Integrator(model, self.solver, substeps=substeps)

        # 轮关节 DoF 索引
        self.wheel_keys = [
            "hinge_fl",
            "hinge_fr",
            "hinge_rl",
            "hinge_rr",
        ]
        qd_start = self.model.joint_qd_start.numpy()
        joint_keys = list(self.model.joint_key)
        self.wheel_dof_indices: list[int] = []
        for k in self.wheel_keys:
            if k in joint_keys:
                jid = joint_keys.index(k)
                self.wheel_dof_indices.append(int(qd_start[jid]))

        # 控制参数
        self.drive_speed = 10.0
        self.turn_gain = 0.5
        self.time = 0.0

    def _update_control(self):
        forward = 0.0
        if self.viewer.is_key_down("i"):
            forward += 1.0
        if self.viewer.is_key_down("k"):
            forward -= 1.0

        turn = 0.0
        if self.viewer.is_key_down("j"):
            turn -= 1.0
        if self.viewer.is_key_down("l"):
            turn += 1.0

        left = self.drive_speed * forward * (1.0 - self.turn_gain * max(0.0, turn))
        right = self.drive_speed * forward * (1.0 - self.turn_gain * max(0.0, -turn))

        dof_count = int(self.model.joint_qd_start.numpy()[-1])
        targets = [0.0] * dof_count
        if len(self.wheel_dof_indices) == 4:
            # 前左、前右、后左、后右
            targets[self.wheel_dof_indices[0]] = left
            targets[self.wheel_dof_indices[1]] = right
            targets[self.wheel_dof_indices[2]] = left
            targets[self.wheel_dof_indices[3]] = right
        self.integrator.control.joint_target.assign(targets)

    def step(self):
        self._update_control()
        self.integrator.step(self.viewer, self.dt)
        self.time += self.dt

    def render(self):
        self.viewer.begin_frame(self.time)
        self.viewer.log_state(self.integrator.state_0)
        self.viewer.log_contacts(self.integrator.contacts, self.integrator.state_0)
        self.viewer.end_frame()


__all__ = ["RigidCarRunner"]
