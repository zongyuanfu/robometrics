# Robometrics 数据、CPU ESDF 与公共评估增量扩展计划

日期：2026-10-14。状态：阶段 2 的 CPU 实现已完成；Git 提交、旧锁文件重建
及 GPU 联合 smoke 待解决当前环境限制。

## 1. 开发位置与目标

**目标仓库就是当前工作目录 `/home/fzy/codes/robometrics`。**
Planigo 位于 `/home/fzy/Desktop/planigo`，作为实现参考和后续调用方。
不另建 `/home/fzy/Desktop/robot-benchkit`，也不把本库放进 Planigo 的
`third_party/`。本计划直接维护在当前仓库的 `plans/` 中。

近期继续使用项目名和 import 名 `robometrics`。`robot-benchkit` 只作为旧计划的
暂定名称保留在本文件名中；是否改名留到发布准备，不与功能迁移一起进行。

沿用已有数据和简洁模块，逐步实现：

- 数据加载不依赖 Planigo，NumPy 依赖声明允许经验证的 1.26 / 2 环境。
- 四类解析障碍物的 NumPy CPU ESDF，随后增加可选 C++ 加速。
- Franka Panda 模型和公共轨迹评估，保留迁移前后的数据与判定一致性。
- Planigo、cuRobo 等调用方共享数据、模型和评估约定；规划器编排仍在调用方。

本库采用 NumPy / CPU C++ 路径，FK、距离查询和轨迹评估也在 CPU 上完成。
不引入 JAX 依赖、可选 extra、JIT 或 JAX 后端。Planigo 联合使用围绕现有
host 数据与 CUDA 上传接口设计，并随各阶段做小规模接入验证。

首轮不扩展到 GPU ESDF、mesh/voxel、任意机器人自动适配、连续碰撞检测或
真实机器人控制。先完成可独立使用的 CPU 路径，再处理发布工程。

## 2. 保持平铺目录，按需增加文件

现有代码约 800 行，已经有数据、机器人和评估模块，没有必要先搭建多层框架。
保留根目录包布局，不迁到 `src/`，不预建 `datasets/geometry/robots/esdf/`
等子包，也不增加通用后端注册器、插件系统或大量数据包装类。

```text
/home/fzy/codes/robometrics/
├── pyproject.toml
├── README.md
├── LICENSE
├── Licenses
├── robometrics/
│   ├── __init__.py
│   ├── file_structure.py       # 保留资源定位接口
│   ├── datasets.py             # 原始数据读取与现有兼容接口
│   ├── robometrics_types.py    # 保留现有类型
│   ├── robot.py                # 保留现有 Robot
│   ├── evaluator.py            # 保留现有 Evaluator
│   ├── metrics.py
│   ├── statistics.py
│   ├── content/
│   │   ├── dataset/            # 已有三套 YAML，原样保留
│   │   └── robot/              # 已有资源，后续按需补充
│   ├── geometry.py             # 阶段 2：primitive 与 Scene
│   └── esdf.py                 # 阶段 2：host context、构建；后续加入查询
├── tests/                     # 随功能补充必要测试
└── plans/
    └── robot_benchkit_extraction.md
```

图中的 `geometry.py` 和 `esdf.py` 已在阶段 2 实现。
阶段 3 再增加 `cpp/esdf.cpp` 与必要构建文件；性能脚本只需一个
`benchmarks/esdf.py`。阶段 4 确有需要时再增加 `franka.py`、`sampling.py`、
`trajectory.py`、`success.py` 和 NumPy `kinematics.py`，不提前占位。

单个模块的职责明显分离、难以阅读时再拆分。常量、版本和少量配置留在所属
模块或结果记录中，不为每一种元信息单建模块。

## 3. 职责与兼容边界

| 本库负责 | Planigo / 其他调用方负责 |
|---|---|
| 原始数据、资源定位、数据来源 | 规划器、IK 策略、实验编排 |
| 解析障碍物、CPU ESDF、距离查询 | CUDA 上传、GPU kernel、求解预算 |
| Franka 模型、FK、碰撞球与规则 | B-spline、MPC 执行、warm start |
| 轨迹采样、公共指标与成功判定 | 规划器原生结果、可视化与展示 |

本库的实现、依赖和测试不使用 JAX，也不得导入 `planigo`、Torch 或 cuRobo。
Planigo 通过单向依赖和少量转发接入本库；本库不为此增加 integrations 子包。

保留 `get_dataset_path()`、`demo_raw()`、`motion_benchmaker_raw()`、
`mpinets_raw()` 以及现有结构化数据接口。几何数据仍可通过原有
`geometrout` 路径使用，新 CPU 路径不要求它。

当前 `Evaluator`、`Robot` 与 Planigo 的模型和评估并非同一个接口。
特别是现有指标使用厘米、角度，不能直接改成米、弧度或换成新的成功标准。
新增公共评估先采用独立函数和明确单位，不覆盖旧接口的含义。

## 4. 只冻结当前需要的约定

暂不一次定义 `PlanningProblem`、通用 `RobotModel`、`TrajectorySamples` 等
整套公共对象。数据阶段继续返回原始字典；几何和 ESDF 阶段只引入现有实现
已经需要的简单 dataclass，并尽量保留 `Scene`、`EsdfContext` 等名称。

需要保持的约定：

- 原始 YAML 和障碍物 pose 保留 `xyz + wxyz`；转换后的目标位姿、FK 接口
  按原实现使用 `xyzw`。字段名和文档说明顺序，不在迁移中统一重编码全部 pose。
- raw 场景与 `canonical_obb_v1` 明确区分。第一版 ESDF 使用输入的解析几何；
  OBB 转换先留在调用方，后续有实际共用需求时再迁移。
- workspace 必须由调用方显式传入。Franka 范围和按目标补足 margin 的策略
  不进入通用 YAML 读取函数。
- ESDF 是四类 primitive SDF 的逐点最小值，内部为负；保留当前有限正值
  上限和空场景值 `10.0`，不换成 occupancy distance transform。
- 输出为有明确所有权的 C-contiguous float32 host 数组；保留首个 voxel
  中心、`ceil(extent / voxel_size)` shape、原始物理边界和只读 snapshot 语义。
- 非整除范围中，物理边界与网格覆盖边界不能混淆。查询保留三线性插值、
  边缘复制和显式 OOB mask，不能把 OOB 返回的零距离当作有效查询。
- 指标计算不修改轨迹；采样网格、导数来源和成功阈值显式记录。

CPU ESDF 接口优先沿用易读的形式，不额外套 evaluator 工厂：

```python
from robometrics.geometry import Scene
from robometrics.esdf import build_esdf_from_scene

scene = Scene.create(problem["obstacles"])
context = build_esdf_from_scene(
    scene,
    workspace_center=center,
    workspace_extent=extent,
    voxel_size=0.02,
)
```

以上为阶段 2 的接口方向，签名在该阶段验收时固定。`backend="cpp"` 和
`threads` 到阶段 3 才引入；显式指定 C++ 而扩展不可用时直接报错。
context 保留 `as_mapping()`，包含 `grid`、`origin`、`voxel_size`、
`grid_shape`、`lower_bound`、`upper_bound` 和 `version`。布局与字段对齐
Planigo 现有 `EsdfContext.from_mapping()` 和 `CudaBsplineSolver`；由 Planigo
直接上传 host 网格到 CUDA，CPU 构建结果不经过设备数组中转。

## 5. 分阶段执行，每次只做一个可验收的小改动

### 阶段 0：核对现状、调整计划（本轮已完成）

已阅读当前包、测试和依赖声明，并对照 Planigo 的数据、几何、ESDF 和模型实现。
本地 YAML 核对结果：demo 为 1 组 5 题，MB 为 8 组各 100 题，MPINets 为
12 组共 1800 题。已有 `franka_spheres.yml` 也有 61 个球，但数量相同不代表
与 Planigo 模型的 link、FK、自碰撞规则和 fingerprint 已经等价。

阶段 0 只调整计划与保存基线，完整数值基线仍需在对应迁移阶段记录。

当前仓库已由用户初始化，23 个源码、资源和计划文件在本仓库冻结：

- Git commit：`d7ec62dca070323df739dfc716098d164020803f`。
- 基线 tag：`robometrics-baseline-20261014`。

三套 YAML 的 SHA256 与来源记录见 README。后续验证使用该本仓库基线，
此前的独立 bundle 不作为本仓库历史。

### 阶段 1：数据入口与 NumPy 安装兼容（实现已完成，旧锁文件待重建）

改动集中在 `datasets.py`、`pyproject.toml`、数据测试及 README：

1. 在已有模块增加统一的 `load_raw_dataset(name, data_dir=None)` 和
   `list_problem_groups(...)`，已有三个 raw 函数复用读取逻辑。
2. 默认仍读包内 YAML；允许显式数据目录，未知名称和缺失文件给出清楚错误。
   不自动下载、不改题目顺序，不引入共享可变缓存。
3. 原始读取不需要几何对象，将结构化接口使用的可选依赖延迟到调用时导入。
4. 调整 NumPy 声明以允许 1.26 / 2，不照搬 Planigo 的固定版本；暂时沿用
   Poetry 打包方式，依赖变化时同步现有 `poetry.lock`，不迁移 lock 格式或
   整体重写项目元数据。
5. 记录三个 YAML 的 SHA256、可取得的源码版本及现有来源/许可说明。
   数据文件和原有许可文本保持原样，安装包应继续包含这些资源。

验收：旧 raw 入口与新入口的全部原始字段一致，题目集合和顺序不变；
自定义目录和错误路径可用；纯核心环境能够安装并读取打包数据。
先在 Python 3.11 上分别验证 NumPy 1.26 / 2；旧 evaluator 的 NumPy 2
兼容性单独验证，不能由数据加载通过推断。Python 支持范围另行按实测明确。

**本阶段结束即可独立交付。暂不迁入 ESDF、模型或评估代码。**

### 阶段 2：解析几何与 NumPy CPU ESDF（CPU 实现与数值对照已完成）

只增加 `geometry.py`、`esdf.py` 及对应测试：

- 从 Planigo 的 `geometry/primitives.py` 迁入四类 primitive 和 `Scene`。
- 从 `geometry/esdf_builder.py` 迁入 NumPy 计算部分，移除末尾 JAX 转换；
  合并 `geometry/context.py` 的 host snapshot 校验到 `esdf.py`。
- 先保持表达式、遍历顺序和中间精度，最终输出 float32；不要同时重写算法。
- workspace 显式传入，先不迁入 Franka 问题包装和 workspace 策略。

验收：四类 primitive、旋转场景、空场景、无效输入、dtype/strides、
非整除边界、网格中心和 snapshot 所有权通过验证；固定场景的逐 voxel
结果对齐 Planigo 冻结基线。验收时再验证 `as_mapping()` 可由现有 CUDA
facade 接收，优先替换 ESDF 构建入口，不等待全部模型和评估迁移完成。

### 阶段 3a：单线程 C++ 后端

新增一个 `cpp/esdf.cpp`，Python 侧继续使用阶段 2 的同一入口和 context。
在本阶段选择最小构建方式，不先铺设 `include/src/backends` 多层目录。
纯 Python 安装路径继续可用，C++ 构建和运行不需要 CUDA。

预计算障碍物常数，按 voxel 融合坐标、距离和最小值归约，避免完整坐标中间
数组和 voxel × 障碍物距离矩阵。通过 pybind11 绑定，计算时释放 GIL；
保留参考精度，不启用激进 fast-math。

验收：NumPy / C++ 的逐 voxel 对照、输入与边界验证以及 NumPy 1.26 / 2
绑定测试通过；用单个性能脚本记录构建时间与峰值内存。先交付串行版本。

### 阶段 3b：按性能结果增加 OpenMP

串行版本通过后再增加 1/2/4/8 线程对照，输出位置互不重叠，不修改调用
进程的全局线程设置。无 OpenMP 的构建仍支持串行；并行能力和实际线程数
写入报告。只有实测确有收益才将并行路径作为推荐配置。

每题重新构建 ESDF，不使用场景/网格结果缓存。YAML 读取和障碍物准备成本
单独记录。300 ms 中位数仅为初始工程目标，历史约 1.25 s 需要在同一设备、
范围、分辨率和统计口径下复测，不能当作已验证加速结果。

### 阶段 4：模型与公共评估，继续拆成小步

顺序为 **4a 模型 → 4b 查询/FK → 4c 采样与指标 → 4d 成功策略**。
每一步先对照已保存轨迹，再迁下一部分，不一次覆盖现有 evaluator。

| Planigo 来源 | 当前库中的目标 |
|---|---|
| `robot/franka.py` 与必要资源 | `franka.py` 与 `content/robot/` |
| `robot/kinematics_jax.py` 的 FK 数学与既有基线输出 | `kinematics.py`，用 NumPy 实现链式变换与碰撞球变换 |
| `geometry/collision.py` 的插值与边界规则 | `esdf.py`，用 NumPy 实现距离查询与有效 mask |
| `evaluation/sampling.py` | `sampling.py` |
| `evaluation/trajectory.py` 的指标定义与既有基线输出 | `trajectory.py`，NumPy CPU 评估，复用兼容的现有函数 |
| `evaluation/success.py` | `success.py` |

模型保持 61 球、11 links、694 个 self-collision pairs、固定 0.04 m
手指变换、ignore pairs、相加 padding 及 fingerprint。先比对当前资源，
需要不同资源时明确命名，不覆盖旧资源冒充等价模型。

保持 Planigo 调用形态：`FrankaModel.load()`、`TrajectoryEvaluator.end_effector()`、
`evaluate()` 与 `SuccessPolicy.evaluate()`。轨迹输入为 `(T, 7)` 的
q/dq/ddq/jerk host 数组，目标 pose 为 xyz + xyzw；指标字段与 failure codes
保持一致。Planigo 的 orientation error 是 `1 - abs(quaternion_dot)`，
不能在迁移时换成旋转角。矩阵转四元数保留半转附近的稳定计算方法。

FK、插值和指标按数学定义转换为 NumPy，并对照已保存的基线，不迁入原实现的
运行框架。五次 Hermite 重建需要 SciPy 时，在该阶段显式声明评估依赖，
不让数据和 ESDF 路径强制安装它。

评估保留 endpoint-inclusive 物理时间网格，以及 cuRobo q/dq/ddq 的五次
Hermite 重建。完整时域最大 0.025 s 判定网格和 MPC 0.05 s 执行记录分别
标注；native/common success 分开，规划失败仍留在分母。

验收：FK、碰撞球、clearance、OOB、关节与动力学限制、末端误差、采样、
指标和 failure codes 与冻结基线对齐；NumPy 1.26 / 2 分别验证。
对 NumPy FK/查询使用解析案例和保存轨迹做独立验证，测试不安装或运行 JAX。
评估性能单独测量，避免把 CPU FK 重写的成本混入 ESDF 加速收益。

### 阶段 5：调用方接入与发布准备

各阶段对应功能验收后，可穿插处理 Planigo 接入；阶段 5 汇总整体验收。
每次只替换一条路径：
数据 → ESDF → 模型 → 评估。旧 import 可用薄转发层暂时保留，最终每份
实现只有一个 owner。接入顺序不要求等全部公共评估完成后才能验证 CPU ESDF。

联合使用的最小检查随功能阶段推进：

| 阶段 | Planigo 接入点 | 联合验收 |
|---|---|---|
| 1：数据 | `benchmarks/datasets/robometrics_dataset.py` | 默认资源路径可用，全部原始题目与 case ID 不变 |
| 2 / 3：ESDF | `geometry/esdf_builder.py` 的薄转发与 `backend/cuda_bspline_solver.py` 上传入口 | shape、dtype、边界与版本字段兼容，小场景网格上传和求解 smoke 通过 |
| 4a：模型 | 模型加载与 CUDA 模型生成/校验工具 | fingerprint 和编译模型校验不变 |
| 4b–4d：评估 | `evaluation/` 的薄转发 | NumPy 评估接收现有轨迹数组，MPC、完整时域、cuRobo 的指标与失败码对齐 |

先替换 CPU ESDF 路径，再替换公共评估。Planigo 保留 workspace/geometry profile
选择、IK、MPC 和实验配置；本库只接收显式输入。接入时可以保留现有类名和
方法，避免调用方同时重写数据结构和规划流程。

本机 Planigo 的 editable 来源应指向当前库的实际位置，例如：

```toml
[tool.uv.sources]
robometrics = { path = "../../codes/robometrics", editable = true }
```

该路径相对 `/home/fzy/Desktop/planigo/pyproject.toml`，不能使用旧计划中的
`../robot-benchkit`。同步处理 dependency 声明，并检查 cuRobo benchmark extra
是否重新引入上游包；不修改 `third_party/curobo` 或 `third_party/MGIGO`。
只有依赖解析和对应 smoke 通过后，才移除仅为上游约束存在的 override。

模型生成工具经适配层读取新 owner；记录迁移前后 fingerprint，保持 CUDA ABI。
接入过程中不调整规划器参数、GPU kernel、求解预算或执行采样密度。

最后再补 CI、sdist/wheel 资源验证、最小示例、来源说明和实测性能文档。
先支持验证过的平台，不把多平台 wheel、自动下载和发布工作塞入前几阶段。
保留 `LICENSE` / `Licenses`；迁入代码与新增模型资源补足来源、版本和许可说明。

## 6. 验证与阶段完成标准

每次只运行与本次改动相关的检查，上一阶段通过才推进下一阶段。

- 数据：三套 YAML 的全量字段、ID/顺序与文件 hash 对照。
- 几何/ESDF：小场景边界案例、随机合法场景逐 voxel 对照；C++ 与线程对照
  使用相同输入。容差根据实际精度确定，阈值附近的判定变化逐例解释。
- 评估：优先复用保存轨迹，避免每迁一个函数就重复全量规划。
- 依赖：安装后的数据、ESDF、FK 和评估路径均在无 JAX 环境中验证。
- 接入：先少量分层 smoke；正式结果按小样本 → MB800 单 seed → 必要时
  多 seed 推进。离散通过不宣称连续运动或真实机器人安全。
- 性能：记录打包、计算、转换/校验与端到端时间，给出中位数、P90、峰值
  内存和设备/编译/依赖信息。多线程对照明确记录 `OMP_NUM_THREADS` 的影响。

不提交生成的轨迹、性能 CSV、build outputs 或环境 traces。
源码版本、数据 hash、模型 fingerprint 和评估配置随运行记录保存，不先建立
独立 provenance 框架。

## 7. 当前交付与下一步

已确认本仓库 Git 基线，并实施阶段 1 的统一数据入口、旧接口转发、延迟依赖
导入、NumPy 声明与安装文档。原始数据和旧结构化数值逻辑保持不变。

本次验证结果：

- 源码环境：15 项测试通过，包括原有结构化数据读取。
- Python 3.11 干净安装环境：NumPy 1.26.4 / 2.4.6 各有 14 项测试通过；
  各跳过 1 个未安装可选几何依赖的结构化测试。
- 两个环境均从构建 wheel 安装，在仓库外执行测试；无 JAX、Torch、cuRobo、
  Planigo、geometrout 或 urchin。
- wheel / sdist 的三套 YAML 和两份许可文本与基线逐字节一致。

Poetry 构建后端保持不变。当前环境未安装完整 Poetry，PyPI 访问受网络限制；
已有 `poetry.lock` 与原声明不一致，重建锁文件仍是阶段 1 的剩余项，未手工
改写或伪造锁结果。保留该遗留项，按本次执行指令继续阶段 2。

阶段 2 已增加 `geometry.py`、`esdf.py` 和对应测试：

- NumPy 1.26.4 / 2.4.6 的 37 项几何/ESDF 测试均通过。
- 新 wheel / sdist 包含两个新增模块并保留数据与许可资源。两种 NumPy 的
  独立安装环境在仓库外执行完整测试，各有 51 项通过、1 项可选结构化测试
  跳过；安装环境没有 JAX、GPU runtime 或 Planigo。
- 对照 Planigo commit `ba59e35adcfc4a6048fe27e9c03e26c8ec7516d0` 的原始
  NumPy 计算部分与 float32 host 输出，不加载 JAX。90 个空/随机混合/真实
  数据场景共 69,300 个 voxel 的距离与原始 context 字段逐值一致。
- Planigo 原 `EsdfContext.from_mapping()` 接收新 context；实际 CUDA facade
  也通过 host 校验和模型指纹校验，但在设备初始化时返回
  `no CUDA-capable device is detected`。当前会话没有 `/dev/nvidia*`，
  GPU 上传/求解/执行/reset smoke 尚未通过，不能记为已验证。
- 本库仅增加 CPU 实现，workspace/profile 策略和 Planigo 代码未修改。

本次数值对照的 Planigo 源文件 SHA256：

| 文件 | SHA256 |
|---|---|
| `geometry/primitives.py` | `f83818a72807173ddefdb630f1a5082594329844d1564c8574249ad1fcda55ec` |
| `geometry/context.py` | `2f12cfcbfb8c746f94bc4c371dd8b75fcce0b8786be0583bca2de8700309fbf0` |
| `geometry/esdf_builder.py` | `e3c279825665a69d18c2309251831eb4ce6767babbe05360fc6a5ea8875a6774` |

已按指令尝试提交阶段 1；当前会话仍不能写入 `.git/index.lock`，提交未生成。
源码修改保留在本仓库工作区，没有再创建独立 Git 快照或改动既有基线历史。
下一实现阶段为单线程 C++；当前遗留验收项需继续保留和复核。
