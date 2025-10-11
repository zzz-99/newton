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


def add_tire_ring(
    builder: newton.ModelBuilder,
    center: wp.vec3,
    rot: wp.quat,
    R_major: float,
    r_minor: float,
    n_theta: int = 32,
    n_phi: int = 8,
    mass: float = 0.1,
    particle_radius: float = 0.02,
    spring_ke: float = 5e3,
    spring_kd: float = 5e2,
):
    """
    Create a toroidal soft tire around Y axis of the wheel (using rot).

    - center, rot: wheel body transform in world space
    - R_major: ring centerline radius (target tire radius)
    - r_minor: ring thickness radius
    - n_theta: samples around main circle
    - n_phi: samples around cross-section
    """
    # local frame axes from rot
    # We consider the ring lies around local +Y axis
    # build parametric torus in local coords then rotate+translate
    def local_to_world(p):
        v = wp.vec3(p[0], p[1], p[2])
        w = wp.quat_rotate(rot, v)
        return [center[0] + w[0], center[1] + w[1], center[2] + w[2]]

    # particle positions
    pos = []
    vel = []
    for it in range(n_theta):
        theta = (2.0 * math.pi) * (it / n_theta)
        # center of tube (around Y)
        cx = R_major * math.cos(theta)
        cz = R_major * math.sin(theta)
        for ip in range(n_phi):
            phi = (2.0 * math.pi) * (ip / n_phi)
            # cross-section circle in plane normal to ring direction
            # construct local tangent and normal basis for the torus
            # around Y axis: tangent along -X/Z, normal plane is XZ; choose two orthonormal vectors
            # local normal from ring center to tube point in XZ plane rotated by phi
            nx = math.cos(theta)
            nz = math.sin(theta)
            # two basis in the cross section plane
            # b1: radial outward in XZ from ring center
            b1 = np.array([nx, 0.0, nz])
            # b2: binormal pointing +Y
            b2 = np.array([0.0, 1.0, 0.0])
            # point offset on cross-section
            off = (math.cos(phi) * b1 + math.sin(phi) * b2) * r_minor
            lp = np.array([cx, 0.0, cz]) + off
            pos.append(local_to_world(lp))
            vel.append([0.0, 0.0, 0.0])

    # add particles
    start_idx = builder.particle_count
    builder.add_particles(pos, vel, [mass] * len(pos), radius=[particle_radius] * len(pos))

    # connect springs (structural), wrap-around in both directions
    idx = ring_indices(n_theta, n_phi, start_idx)
    for it in range(n_theta):
        for ip in range(n_phi):
            a = idx[it][ip]
            b = idx[(it + 1) % n_theta][ip]
            c = idx[it][(ip + 1) % n_phi]
            builder.add_spring(a, b, spring_ke, spring_kd, 0.0)
            builder.add_spring(a, c, spring_ke, spring_kd, 0.0)

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
        tire_radius = 0.32
        tire_thickness = tire_radius - wheel_radius_rim
        wheel_half_height = 0.18 * 0.5
        wheel_mass = 10.0

        # chassis centered so tires can touch ground
        chassis_pos = wp.vec3(0.0, 0.0, tire_radius + chassis_hz + 0.05)
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
        mu_rim = 1.5
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
                target_ke=50.0,
                target_kd=0.0,
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

            # soft tire ring
            tire_start, tire_count = add_tire_ring(
                builder,
                center=world_pos,
                rot=wp.quat_identity(),
                R_major=(wheel_radius_rim + tire_thickness * 0.5),
                r_minor=tire_thickness * 0.5,
                n_theta=48,
                n_phi=10,
                mass=0.08,
                particle_radius=self.tire_particles_radius,
                spring_ke=8e3,
                spring_kd=6e2,
            )
            self.tires.append((tire_start, tire_count))

        # finalize model and solver
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverSemiImplicit(self.model, angular_damping=0.12)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # contact params
        self.model.soft_contact_mu = mu_ground
        self.model.soft_contact_ke = 1.2e4
        self.model.soft_contact_kd = 2.0e2
        self.model.soft_contact_radius = self.tire_particles_radius
        self.model.particle_max_velocity = 30.0

        # wheel drive
        self.wheel_speed = 0.0
        self.max_wheel_speed = 25.0
        self.accel_rate = 6.0
        self.brake_rate = 8.0

        # viewer
        self.viewer.set_model(self.model)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
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
        substeps = 20
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