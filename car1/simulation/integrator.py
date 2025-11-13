from __future__ import annotations

import newton


class Integrator:
    def __init__(self, model: newton.Model, solver, substeps: int = 4):
        self.model = model
        self.solver = solver
        self.substeps = substeps
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

    def step(self, viewer: newton.viewer.ViewerBase, dt: float):
        sub_dt = dt / float(max(1, self.substeps))
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            viewer.apply_forces(self.state_0)
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        return self.state_0, self.contacts


__all__ = ["Integrator"]
