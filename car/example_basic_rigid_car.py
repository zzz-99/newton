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

        # basic contact parameters (soft contact model used by SemiImplicit)
        # tuned moderately for stable rolling without excessive bounce
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

            # collision box approximating cylinder's oriented bounding box
            # For cylinder axis along Y, bounding box extents should be:
            # local hx=r, hy=r, hz=half_height (after rotX(90°): hy -> world Z = r)
            col_cfg = builder.default_shape_cfg.copy()
            col_cfg.has_shape_collision = True
            col_cfg.is_visible = False
            col_cfg.mu = 0.9  # slightly higher tire-ground friction
            builder.add_shape_box(
                body_wheel,
                xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=cyl_rot_to_y),
                hx=wheel_radius,
                hy=wheel_radius,
                hz=wheel_half_height,
                cfg=col_cfg,
                key=f"wheel_col_{i}",
            )

            # revolute joint at wheel center, axis = Y in parent's local frame
            # fixed angular velocity via TARGET_VELOCITY
            omega = 8.0  # rad/s, roughly ~76 RPM
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

        # tune soft contact (used for rigid collisions in SemiImplicit)
        self.model.soft_contact_ke = 2.0e3
        self.model.soft_contact_kd = 2.0e1
        self.model.soft_contact_mu = 0.8
        self.model.soft_contact_restitution = 0.1

        # solver
        self.solver = newton.solvers.SolverSemiImplicit(self.model)

        # states & control
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # viewer and initial FK
        self.viewer.set_model(self.model)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # capture graph if cuda
        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # allow interactive forces (mouse picking, etc.)
            self.viewer.apply_forces(self.state_0)

            # explicit collision update for non-MuJoCo solvers
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

            self.sim_time += self.sim_dt

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer)
    newton.examples.run(example, args)