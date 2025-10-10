# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Simple Car Demo
#
# Shows how to create a simple car with a box body and four cylindrical wheels
# connected using REVOLUTE joints.
#
# Command: uv run python simplecar.py
#
###########################################################################

import warp as wp
import newton
import newton.examples


class SimpleCar:
    def __init__(self, viewer):
        # setup simulation parameters
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        builder = newton.ModelBuilder()

        # add ground plane
        builder.add_ground_plane()

        # Car parameters
        car_height = 1.0
        car_body_size = wp.vec3(2.0, 1.0, 0.5)  # length, width, height
        wheel_radius = 0.3
        wheel_width = 0.2
        wheel_offset = 0.8  # distance from center to wheel

        # Create car body (main chassis)
        car_pos = wp.vec3(0.0, 0.0, car_height)
        self.car_body = builder.add_body(
            xform=wp.transform(p=car_pos, q=wp.quat_identity()),
            key="car_body"
        )
        builder.add_shape_box(
            self.car_body, 
            hx=car_body_size[0]/2, 
            hy=car_body_size[1]/2, 
            hz=car_body_size[2]/2
        )

        # Create four wheels (cylinders) - positioned at the sides of the car body
        # Wheels are placed at the edges of the car body, extending outward
        wheel_positions = [
            wp.vec3(wheel_offset, car_body_size[1]/2 + wheel_width/2, car_height - wheel_radius),    # front right
            wp.vec3(wheel_offset, -car_body_size[1]/2 - wheel_width/2, car_height - wheel_radius),   # front left  
            wp.vec3(-wheel_offset, car_body_size[1]/2 + wheel_width/2, car_height - wheel_radius),   # rear right
            wp.vec3(-wheel_offset, -car_body_size[1]/2 - wheel_width/2, car_height - wheel_radius)   # rear left
        ]

        self.wheels = []
        wheel_names = ["front_right", "front_left", "rear_right", "rear_left"]

        for i, (pos, name) in enumerate(zip(wheel_positions, wheel_names)):
            # Create wheel body - rotate cylinder 90 degrees around X-axis to make it vertical
            wheel_rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi/2)
            wheel = builder.add_body(
                xform=wp.transform(p=pos, q=wheel_rotation),
                key=f"wheel_{name}"
            )
            builder.add_shape_cylinder(wheel, radius=wheel_radius, half_height=wheel_width/2)
            self.wheels.append(wheel)

            # Connect wheel to car body with REVOLUTE joint
            # Joint axis is Y (for wheel rotation around horizontal Y-axis)
            joint_axis = wp.vec3(0.0, 1.0, 0.0)
            
            # Parent transform (car body connection point)
            parent_offset = wp.vec3(
                pos[0] - car_pos[0],  # x offset from car center
                pos[1] - car_pos[1],  # y offset from car center  
                -car_body_size[2]/2   # bottom of car body
            )
            
            # Child transform (wheel center) - apply same rotation as wheel body
            child_offset = wp.vec3(0.0, 0.0, 0.0)  # wheel center
            
            builder.add_joint_revolute(
                parent=self.car_body,
                child=wheel,
                axis=joint_axis,
                parent_xform=wp.transform(p=parent_offset, q=wp.quat_identity()),
                child_xform=wp.transform(p=child_offset, q=wheel_rotation),
                key=f"joint_{name}"
            )

        # finalize model
        self.model = builder.finalize()

        # Create solver
        self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)

        # Create states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # Set up viewer
        self.viewer.set_model(self.model)

        # Initialize forward kinematics
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

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

            # Apply forces from viewer (mouse interactions)
            self.viewer.apply_forces(self.state_0)

            # Apply car controls based on keyboard input
            self.apply_car_controls()

            # Collision detection and physics step
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def apply_car_controls(self):
        """Apply simple car controls based on keyboard input
        
        Controls (to avoid conflict with viewer camera controls):
        - I: Forward
        - K: Backward  
        - J: Turn left (differential steering)
        - L: Turn right (differential steering)
        """
        drive_force = 50.0
        turn_force = 30.0
        
        # Create a temporary array to hold the forces
        forces = wp.zeros(4, dtype=wp.float32)
        
        # Get input states
        forward = self.viewer.is_key_down("i")
        backward = self.viewer.is_key_down("k") 
        turn_left = self.viewer.is_key_down("j")
        turn_right = self.viewer.is_key_down("l")
        
        # Calculate base drive force
        base_force = 0.0
        if forward:
            base_force = drive_force
        elif backward:
            base_force = -drive_force
            
        # Apply differential steering for turning
        left_force = base_force
        right_force = base_force
        
        if turn_left:
            left_force -= turn_force
            right_force += turn_force
        elif turn_right:
            left_force += turn_force  
            right_force -= turn_force
            
        # Assign forces to wheels (front-left, front-right, rear-left, rear-right)
        forces.assign([left_force, right_force, left_force, right_force])
        
        # Copy forces to control
        wp.copy(self.control.joint_f, forces)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test(self):
        # Simple test to check if car body is above ground
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "car body is above ground",
            lambda q, qd: q[2] > 0.5,  # car body z position should be above 0.5
            [0]  # car body index
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create car demo and run
    car_demo = SimpleCar(viewer)

    newton.examples.run(car_demo, args)