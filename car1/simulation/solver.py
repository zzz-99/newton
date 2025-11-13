from __future__ import annotations

import newton


class XPBDSolver:
    def __init__(self, model: newton.Model, iterations: int = 10):
        self.solver = newton.solvers.SolverXPBD(model, iterations=iterations)

    def step(self, state_0: newton.State, state_1: newton.State, control: newton.Control, contacts: newton.Contacts, dt: float):
        self.solver.step(state_0, state_1, control, contacts, dt)


__all__ = ["XPBDSolver"]
