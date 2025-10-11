import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import warp as wp

import newton
import newton.examples


def make_spring_sphere(
    builder: newton.ModelBuilder,
    center: wp.vec3,
    radius: float,
    grid_n: int = 9,
    particle_radius: float = 0.03,
    density: float = 600.0,
    spring_ke: float = 2.0e4,
    spring_kd: float = 1.0e3,
):
    """
    以规则体素网格近似球体：
    - 仅在半径内创建粒子
    - 在 6 邻域 + 面对角 + 体对角之间建立弹簧，增强稳定性
    - 粒子质量按球体总体积均分
    """
    h = (2.0 * radius) / float(grid_n - 1)
    origin = np.array([center[0] - radius, center[1] - radius, center[2] - radius])

    pos = []
    vel = []
    index_map: Dict[Tuple[int, int, int], int] = {}

    def inside(ix, iy, iz):
        p = origin + np.array([ix * h, iy * h, iz * h])
        d = np.linalg.norm(p - np.array(center))
        return d <= radius + 1e-6

    # create particles inside sphere
    for ix in range(grid_n):
        for iy in range(grid_n):
            for iz in range(grid_n):
                if inside(ix, iy, iz):
                    p = origin + np.array([ix * h, iy * h, iz * h])
                    index_map[(ix, iy, iz)] = len(pos)
                    pos.append([p[0], p[1], p[2]])
                    vel.append([0.0, 0.0, 0.0])

    start_idx = builder.particle_count
    approx_volume = (4.0 / 3.0) * math.pi * (radius ** 3)
    total_mass = density * approx_volume
    mass_per_particle = total_mass / max(1, len(pos))
    builder.add_particles(pos, vel, [mass_per_particle] * len(pos), radius=[particle_radius] * len(pos))

    # neighbor offsets: axis, face-diagonal, body-diagonal
    nbrs = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                # 只连接正半空间，避免重复
                if (dx, dy, dz) <= (0, 0, 0):
                    continue
                # 限制到 6 邻域 + 面/体对角
                nbrs.append((dx, dy, dz))

    # add springs between valid neighbor pairs
    for (ix, iy, iz), pi in index_map.items():
        for dx, dy, dz in nbrs:
            jx, jy, jz = ix + dx, iy + dy, iz + dz
            pj = index_map.get((jx, jy, jz))
            if pj is not None:
                builder.add_spring(start_idx + pi, start_idx + pj, ke=spring_ke, kd=spring_kd, control=0.0)

    return start_idx, len(pos)


def make_tet_sphere(
    builder: newton.ModelBuilder,
    center: wp.vec3,
    radius: float,
    grid_n: int = 9,
    particle_radius: float = 0.03,
    density: float = 600.0,
    k_mu: float = 1.5e4,
    k_lambda: float = 1.5e4,
    k_damp: float = 1.5e3,
):
    """
    以规则体素网格近似球体：
    - 体素角点在球内才创建粒子
    - 对每个完整体素（8角点均在球内）划分 5 个四面体
    - 将体积质量按四面体体积分摊到粒子
    """
    h = (2.0 * radius) / float(grid_n - 1)
    origin = np.array([center[0] - radius, center[1] - radius, center[2] - radius])

    pos = []
    vel = []
    index_map: Dict[Tuple[int, int, int], int] = {}

    def inside(ix, iy, iz):
        p = origin + np.array([ix * h, iy * h, iz * h])
        d = np.linalg.norm(p - np.array(center))
        return d <= radius + 1e-6

    for ix in range(grid_n):
        for iy in range(grid_n):
            for iz in range(grid_n):
                if inside(ix, iy, iz):
                    p = origin + np.array([ix * h, iy * h, iz * h])
                    index_map[(ix, iy, iz)] = len(pos)
                    pos.append([p[0], p[1], p[2]])
                    vel.append([0.0, 0.0, 0.0])

    start_idx = builder.particle_count
    builder.add_particles(pos, vel, [0.0] * len(pos), radius=[particle_radius] * len(pos))

    # 收集所有四面体索引，便于后续提取表面三角
    tet_indices = []

    def add_cell_tets(a000, a100, a010, a110, a001, a101, a011, a111):
        vol = 0.0
        # 将四面体加入构建器并收集索引
        t0 = (a000, a100, a110, a111)
        t1 = (a000, a110, a010, a111)
        t2 = (a000, a010, a011, a111)
        t3 = (a000, a011, a001, a111)
        t4 = (a000, a001, a101, a111)
        vol += builder.add_tetrahedron(*t0, k_mu=k_mu, k_lambda=k_lambda, k_damp=k_damp)
        vol += builder.add_tetrahedron(*t1, k_mu=k_mu, k_lambda=k_lambda, k_damp=k_damp)
        vol += builder.add_tetrahedron(*t2, k_mu=k_mu, k_lambda=k_lambda, k_damp=k_damp)
        vol += builder.add_tetrahedron(*t3, k_mu=k_mu, k_lambda=k_lambda, k_damp=k_damp)
        vol += builder.add_tetrahedron(*t4, k_mu=k_mu, k_lambda=k_lambda, k_damp=k_damp)
        tet_indices.extend([t0, t1, t2, t3, t4])
        return vol

    # 遍历体素并生成四面体
    for ix in range(grid_n - 1):
        for iy in range(grid_n - 1):
            for iz in range(grid_n - 1):
                corners = [
                    index_map.get((ix, iy, iz)),
                    index_map.get((ix + 1, iy, iz)),
                    index_map.get((ix, iy + 1, iz)),
                    index_map.get((ix + 1, iy + 1, iz)),
                    index_map.get((ix, iy, iz + 1)),
                    index_map.get((ix + 1, iy, iz + 1)),
                    index_map.get((ix, iy + 1, iz + 1)),
                    index_map.get((ix + 1, iy + 1, iz + 1)),
                ]
                if any(c is None for c in corners):
                    continue

                v000 = start_idx + corners[0]
                v100 = start_idx + corners[1]
                v010 = start_idx + corners[2]
                v110 = start_idx + corners[3]
                v001 = start_idx + corners[4]
                v101 = start_idx + corners[5]
                v011 = start_idx + corners[6]
                v111 = start_idx + corners[7]

                vol = add_cell_tets(v000, v100, v010, v110, v001, v101, v011, v111)

                # 将体积质量分配到 8 个角点（平均分摊）
                m = density * vol / 8.0
                builder.particle_mass[v000] += m
                builder.particle_mass[v100] += m
                builder.particle_mass[v010] += m
                builder.particle_mass[v110] += m
                builder.particle_mass[v001] += m
                builder.particle_mass[v101] += m
                builder.particle_mass[v011] += m
                builder.particle_mass[v111] += m

    # 提取表面三角：每个四面体的4个面，出现一次的是外表面
    # 保留原始朝向便于着色，使用去重后的唯一面
    face_map: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    for a, b, c, d in tet_indices:
        for tri in ((a, b, c), (a, c, d), (a, d, b), (b, d, c)):
            key = tuple(sorted(tri))
            if key in face_map:
                # 内部面出现两次（相反朝向），去掉
                del face_map[key]
            else:
                face_map[key] = tri

    surface_tris = np.array(list(face_map.values()), dtype=np.uint32)

    return start_idx, len(pos), surface_tris


@dataclass
class XPBDSpringSphere:
    def __init__(self, viewer, center=wp.vec3(0.0, 1.5, 0.0), radius=0.35):
        self.viewer = viewer
        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        # 构建弹簧质点球
        make_spring_sphere(
            builder,
            center=center,
            radius=radius,
            grid_n=11,
            particle_radius=0.03,
            density=600.0,
            spring_ke=2.0e4,
            spring_kd=1.2e3,
        )

        self.model = builder.finalize()
        # XPBD 的四面体约束在 solve_tetrahedra 中使用 soft_body_relaxation 作为“顺从度(compliance)”
        # 数值越小越“硬”，越能维持体积；适当提高迭代次数提升稳定性
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=30,
            soft_body_relaxation=0.02,
            soft_contact_relaxation=0.7,
            angular_damping=0.12,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # 接触略有弹性与阻尼
        self.model.soft_contact_mu = 0.9
        self.model.soft_contact_ke = 3.0e3
        self.model.soft_contact_kd = 3.0e2
        self.model.soft_contact_radius = 0.03
        self.model.soft_contact_restitution = 0.08
        self.model.particle_max_velocity = 12.0

        # 查看器
        self.viewer.set_model(self.model)
        # 默认显示粒子，便于观察软体
        self.viewer.show_particles = True
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.gui, position="stats")
        self.graph = None

    def step(self, time=0.0):
        # 更小的 dt（更多子步）可进一步增强稳定性和体积保持
        substeps = 30
        dt = 1.0 / 60.0 / substeps
        for _ in range(substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def render(self):
        self.viewer.begin_frame(0.0)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def gui(self, ui):
        if not getattr(ui, "is_available", False):
            return
        imgui = ui.imgui
        imgui.begin("XPBD Spring Sphere", flags=imgui.WindowFlags_.always_auto_resize)
        imgui.text("弹簧质点球下落与弹跳")
        imgui.end()


@dataclass
class XPBDTetSphere:
    def __init__(self, viewer, center=wp.vec3(0.0, 1.5, 0.0), radius=0.35):
        self.viewer = viewer
        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        # 构建四面体球
        start_idx, count, surface_tris = make_tet_sphere(
            builder,
            center=center,
            radius=radius,
            grid_n=11,
            particle_radius=0.03,
            density=600.0,
            k_mu=1.5e4,
            k_lambda=1.5e4,
            k_damp=1.5e3,
        )

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=14,
            soft_body_relaxation=0.65,
            soft_contact_relaxation=0.7,
            angular_damping=0.12,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # 接触参数
        self.model.soft_contact_mu = 0.9
        self.model.soft_contact_ke = 3.0e3
        self.model.soft_contact_kd = 3.0e2
        self.model.soft_contact_radius = 0.03
        self.model.soft_contact_restitution = 0.08
        self.model.particle_max_velocity = 12.0

        self.viewer.set_model(self.model)
        # 默认显示三角面（表面），同时保留粒子显示可切换
        self.viewer.show_triangles = True
        self.viewer.show_particles = True

        # 使用 trimesh 作为后备检查与法线（可选）
        self._mesh_normals_wp = None
        try:
            import trimesh as _tm
            # 构建局部顶点/面用于法线计算（局部索引从0开始）
            local_faces = surface_tris.astype(np.int32) - int(start_idx)
            # 从 builder 中的原始位置数组构造局部顶点
            # 注意：pos 在 make_tet_sphere 中创建并传入 add_particles，我们无法直接访问；
            # 这里用当前状态的全局顶点构建局部子集
            state_tmp = self.model.state()
            verts_np = state_tmp.particle_q.numpy()[start_idx : start_idx + count]
            mesh = _tm.Trimesh(vertices=verts_np, faces=local_faces, process=True)
            vnorm = mesh.vertex_normals.astype(np.float32)
            # 将局部法线回填至全局数组长度，其他位置置零
            vnorm_full = np.zeros_like(state_tmp.particle_q.numpy(), dtype=np.float32)
            vnorm_full[start_idx : start_idx + count] = vnorm
            self._mesh_normals_wp = wp.array(vnorm_full, dtype=wp.vec3, device=self.viewer.device)
        except Exception:
            self._mesh_normals_wp = None

        # 记录三角索引到设备，用于每帧 mesh 更新
        self._surface_tris_wp = wp.array(surface_tris.flatten(), dtype=wp.uint32, device=self.viewer.device)
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.gui, position="stats")
        self.graph = None

    def step(self, time=0.0):
        substeps = 20
        dt = 1.0 / 60.0 / substeps
        for _ in range(substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def render(self):
        self.viewer.begin_frame(0.0)
        # 先渲染模型（地面等），再叠加四面体表面
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/tet_surface",
            self.state_0.particle_q,
            self._surface_tris_wp,
            normals=self._mesh_normals_wp,
            hidden=not self.viewer.show_triangles,
            backface_culling=False,
        )
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def gui(self, ui):
        if not getattr(ui, "is_available", False):
            return
        imgui = ui.imgui
        imgui.begin("XPBD Tet Sphere", flags=imgui.WindowFlags_.always_auto_resize)
        imgui.text("四面体球下落与弹跳")
        imgui.end()


def main():
    # 为避免环境中对 newton.examples 的导入差异，提供本地解析器与初始化后备
    try:
        parser = newton.examples.create_parser()
    except Exception:
        import argparse
        parser = argparse.ArgumentParser(add_help=True)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--viewer", type=str, default="gl", choices=["gl", "usd", "rerun", "null"])
        parser.add_argument("--output-path", type=str, default="output.usd")
        parser.add_argument("--num-frames", type=int, default=200)
        parser.add_argument("--headless", action="store_true", default=False)
        parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--case", choices=["spring", "tet"], default="spring")
    # 初始化 viewer（优先使用 examples.init，失败时本地实现）
    try:
        viewer, args = newton.examples.init(parser)
    except Exception:
        import argparse as _argparse
        import newton.viewer as _viewer
        import warp as wp
        args = parser.parse_args()
        if args.device:
            wp.set_device(args.device)
        if args.viewer == "gl":
            viewer = _viewer.ViewerGL(headless=args.headless)
        elif args.viewer == "usd":
            viewer = _viewer.ViewerUSD(output_path=args.output_path, num_frames=args.num_frames)
        elif args.viewer == "rerun":
            viewer = _viewer.ViewerRerun()
        elif args.viewer == "null":
            viewer = _viewer.ViewerNull(num_frames=args.num_frames)
        else:
            raise ValueError(f"Invalid viewer: {args.viewer}")

    if args.case == "spring":
        example = XPBDSpringSphere(viewer)
    else:
        example = XPBDTetSphere(viewer)

    # 运行循环（优先使用 examples.run，失败则本地实现）
    try:
        newton.examples.run(example, args)
    except Exception:
        while viewer.is_running():
            if not viewer.is_paused():
                example.step()
            example.render()
        viewer.close()


if __name__ == "__main__":
    main()