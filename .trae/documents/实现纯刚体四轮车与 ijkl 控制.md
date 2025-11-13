## 目标
- 在 `d:\code\newton\car1` 下按指定结构补齐模块与配置文件，先实现“纯刚体四轮小车 + ijkl 键盘控制”。
- 使用现有 `newton.solvers.SolverXPBD` 与 Viewer 后端，完成构建、仿真步进、可视化与交互。

## 文件结构与新增模块
- config/
  - `scene.json`：保留并扩展为可描述刚体与关节。
  - `material.json`：新增材料库（密度/刚度/阻尼/摩擦/回弹等）。
  - `sim_params.json`：保留；包含 `time_step/substeps/solver_iterations`。
- simulation/
  - `rigidbody.py`：扩展创建车体与四轮，按材料设置形状属性；参考 `car1/simulation/rigidbody.py:31-74`。
  - `coupler.py`：保留软-刚耦合接口但本轮仅实现空壳/占位（纯刚体无需）。
  - `solver.py`：封装 `newton.solvers.SolverXPBD`，读取迭代次数；参考 `newton/_src/solvers/xpbd/solver_xpbd.py:40-99`。
  - `integrator.py`：统一时间步进（清力→碰撞→solver.step），支持 `substeps`。
  - `viewer_runner.py`：GL Viewer 集成，循环渲染并处理按键；参考 `newton/_src/viewer/viewer.py` 与 `viewer_gl.py` 的 `is_key_down`。
- utils/
  - `config_loader.py`：扩展加载 `material.json` 与 `constraints`（支持 `revolute/prismatic/fixed` 等类型）；参考 `car1/utils/config_loader.py:94-133`。
  - `geometry_utils.py`：通用几何工具（生成点云、变换），先提供基础变换工具以定位轮轴锚点。
- `run_sim.py`：主入口，加载配置→构建→运行 Viewer 循环。

## 配置约定
- scene.json（刚体与关节）：
  - `rigid_bodies`: 列表元素含 `id`, `shape`, `size/radius/height`, `pos`, `quat/rpy`, `material`；示例：`chassis`, `wheel_fl`, `wheel_fr`, `wheel_rl`, `wheel_rr`。
  - `constraints`: 列表元素，支持：
    - `type: "revolute"`，`parent`, `child`, `parent_pos`, `child_pos`, `axis: "X|Y|Z"`，可选 `limit_*`, `target_mode`, `target_ke`, `target_kd`。
    - 可选 `type: "fixed"` 作为占位；后续如需悬挂可增 `prismatic`。
- material.json（材料库）：
  - 以键为材料名，值为属性对象：`density`, `kd`, `ke`, `mu`, `restitution`。
  - 车体与轮胎形状通过 `material` 字段引用；缺省走全局默认。
- sim_params.json：
  - `time_step`（如 1/60）、`substeps`（如 4）、`solver_iterations`（如 10）。

## 关键实现
### 刚体构建（rigidbody.py）
- 读取 `rigid_bodies`，用 `ModelBuilder.add_body/add_shape_box|cylinder|sphere` 构造；按 `material.json` 设置形状的 `mu/kd/restitution` 与 `mass/density`。
- 现有函数 `create_rigid(builder, spec)` 已具备形状分派与位姿创建；在此基础上加材料/属性注入；见 `car1/simulation/rigidbody.py:31-74`。

### 关节添加（车轮旋转/转向）
- 使用 `ModelBuilder.add_joint_revolute(...)` 创建 4 个轮-轴铰链；参考 `newton/_src/sim/builder.py:1378-1460`。
- 轴向：以示例为准用 `Axis.Y` 作为轮自转；如需更严格与几何对齐可通过 `geometry_utils` 旋转子体局部坐标。
- 运行时控制通道：通过 `Control.joint_target` 写入目标速度（`JointMode.TARGET_VELOCITY`）；DOF 索引映射见 `newton/_src/sim/model.py:231-280`。
- 差速转向：`J/L` 左右偏置；`I/K` 前后速度；做成参数化的 `drive_speed` 与 `turn_gain`。

### 求解器/步进（solver.py, integrator.py）
- 构造 `SolverXPBD(model, iterations=sim_params.solver_iterations)`。
- 步进：每帧执行 `substeps` 次：`state.clear_forces` → `model.collide` → `solver.step(model, state, dt/substeps, control, contacts)`。
- `viewer_runner` 在每帧调用 `begin_frame/log_state/log_contacts/end_frame`。

### Viewer 与 ijkl 控制（viewer_runner.py）
- 使用 GL 后端：`newton.viewer.ViewerGL`（或仓库缺省 `ViewerBase` 选择器）。
- 键盘：
  - `i`：增加前进目标速度，`k`：反向；`j`/`l`：左右差速偏置；参照 `car/example_basic_rigid_car.py:199-241`、`simplecar.py:145-189`。
  - 相机与鼠标交互无需自实现，沿用 `viewer_gl.py:526-749`。

### run_sim.py 主流程
- 从 `config_loader.load_all_configs` 读取 `scene/material/sim_params/constraints`。
- 构建 `ModelBuilder`：先创建刚体，再按 `constraints` 添加关节。
- 初始化 `Model/State/Control/Contacts/Solver`，启动 `viewer_runner` 循环。

## 验证策略
- 场景：平面地面 + `chassis` + 4 轮；轮与地面接触稳定、可滚动。
- 运行：按 `I/K/J/L` 改变速度与差速，车辆前后移动并左/右转；监视 `contacts` 正常。
- 数值：调 `target_ke/kd` 与 `mu/restitution` 保持稳定；初始 `solver_iterations=10`，`substeps=4`。

## 参考现有实现（便于对齐风格）
- 求解器与抽象：`newton/_src/solvers/xpbd/solver_xpbd.py:40-99`，`newton/_src/solvers/solver.py`。
- Viewer 键盘：`newton/_src/viewer/viewer_gl.py:526-581`；渲染：`newton/_src/viewer/viewer.py:97-140, 407`。
- 刚体构建：`car1/simulation/rigidbody.py:31-74`。
- 车轮控制示例：`car/example_basic_rigid_car.py:104-126, 199-241`；`car/simplecar.py:77-99, 145-189`。

## 交付物清单
- 新增/扩展：`simulation/coupler.py, solver.py, integrator.py, viewer_runner.py`；`utils/geometry_utils.py`；`run_sim.py`。
- 配置：`config/material.json`；完善 `scene.json` 的 `constraints` 字段以声明四个 `revolute` 关节。
- 文档：简短 README 说明键位与参数（可在 `run_sim.py` 顶部打印提示）。