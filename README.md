# Instrument-pose-opt

对私有仿真双器械做 **可微分渲染 + 渲染比较（render-and-compare）** 的位姿优化，不训练网络。优化器可选 **CMA-ES** 或 **Adam**。默认把每个视频第一帧的 GT 位姿加小扰动当作良好初值；不加 `--track` 时后续帧仍各自用该帧 GT 扰动。

可微分渲染器、正向运动学和位姿矩阵组合来自 `instrument-tracking` 的 `simulated_dual_arm_v1` 约定。当前损失为部件 mask + 夹爪端点位置 / 张角（已去掉 GNCC）。

## 数据

默认路径（对应 `\\10.70.70.85\data1\shena\...`）：

- 序列：`/data/data1/shena/data/simulated_data/data30`
- 网格：`/data/data1/shena/data/simulated_instrument`（`shaft.obj` / `wrist.obj` / `left_gripper.obj` / `right_gripper.obj`）

每个 `run_*` 使用左内镜 RGB、语义 mask（1–3 左器械，4–6 右器械）和 `poses.json` 中的 link 位姿。右器械 link 通过立体基线变到左相机后再解算关节。

## 优化变量与流程

每支器械 9 维，两支共 18 维：轴角（3）+ 腕部平移（3）+ 关节 `alpha, theta_left, theta_right`（3）。正向运动学把腕部位姿展开成杆身 / 腕部 / 左右夹爪，再由 nvdiffrast 渲染语义 mask。

```text
视频第一帧：在该帧 GT 腕部位姿上加小扰动（旋转 / 平移 / 关节）
    -> 渲染，检查左右器械是否在视野内
    -> CMA-ES 或 Adam：渲染并比较（mask + tip losses）
    -> 每代/每步记录左右部件 Dice
    -> 结束后统计耗时、Dice、相对 GT 的平移 / 旋转误差
```

默认扰动约为旋转 10°、平移 5 mm、关节 8°。`--optimizer adam` 使用四元数 + **毫米局部平移**（与 Instrument-Splatting 一致）：平移 LR `0.1` mm、旋转 `0.001`、关节 `0.01`，默认 300 步，并按 Dice 动态放大/缩小学习率（Dice&lt;0.8 时 ×3）。旋转 LR 不要用 `0.01`，容易在 Dice≈0.5 附近震荡。

## 环境

```bash
conda env create -f environment.yml
conda activate instrument-pose-opt
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install setuptools wheel ninja
python -m pip install git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae --no-build-isolation
```

## 运行

```bash
# 检查第一帧扰动 GT 初值（不迭代）；输出默认 runs/<run名>
python optimize_pose.py --runs run_1789376009_gpu1_env0 --limit 2 --maxiter 0

# 整份 data30 中取前 15 个样本，每样本等间隔抽 50 帧
python optimize_pose.py --num-samples 15 --limit 50 --save-history

# CMA-ES
python optimize_pose.py --runs run_1789376009_gpu1_env0 --limit 5 --save-history

# Adam
python optimize_pose.py --runs run_1789376009_gpu1_env0 --optimizer adam --limit 5 --save-history

# 第一帧扰动 GT，之后用上一帧结果
python optimize_pose.py --runs run_1789376009_gpu1_env0 --track --optimizer adam
```

未指定 `--output` 时：单样本写到 `runs/<样本文件夹名>`；跑整个 `data30`（或多个 run）时写到 `runs/data30/<样本文件夹名>/`，并在 `runs/data30/` 保留汇总 `summary.json`。也可显式指定：

```bash
python optimize_pose.py --output runs/data30
```

早停：相邻两代/两步 `|Δloss| < --early-stop-tol`（默认 `0.001`）**且** `loss < --early-stop-loss`（默认 `0.1`）时停止。

常用参数：

| 参数 | 含义 |
| --- | --- |
| `--num-samples` / `--max-runs` | 只用 data-root 下前 N 个 `run_*`（如 15）；`0` 表示全部 |
| `--limit` | 每样本抽帧数；`>0` 时在视频内**等间隔**抽取（如 50）；`0` 表示全部 |
| `--init {perturb-gt,identity,centroid,gt}` | 初始位姿，默认 `perturb-gt` |
| `--track` | 仅第一帧扰动 GT，之后用上一帧预测 |
| `--maxiter` / `--popsize` | 迭代次数；CMA 默认 60 / 12，Adam 默认 300 |
| `--early-stop-tol` | `|Δloss|` 阈值；默认 `0.001`；`<=0` 关闭 |
| `--early-stop-loss` | 早停还要求 `loss` 小于该值；默认 `0.1`；`<=0` 关闭损失门槛 |
| `--lr-translation` / `--lr-rotation` / `--lr-joints` | Adam：毫米平移 / 四元数 / 关节 LR，默认 `0.1` / `0.001` / `0.01` |
| `--no-dice-lr` | 关闭 Adam 的 Dice 驱动 LR（默认开启） |
| `--translation-xy-mm` / `--translation-z-mm` | 相对初值平移搜索半宽；默认 ±20 / ±30 mm |
| `--mask-weight` / `--tips-weight` / `--tips-gap-weight` | 损失权重；默认 `1.0` / `1.0` / `0.5` |
| `--save-history` | 保存每代/每步 loss / Dice 曲线 |

## 输出

`runs/<样本名>/`（批量时为 `runs/data30/<样本名>/`）：

- `summary.json` / `summary.csv`：平均每帧时间、左右各部件 Dice、腕部平移（mm）与旋转（deg）误差
- `frames.jsonl`：逐帧记录
- `overlays/`：初始与优化后叠加
- `history/`：可选曲线

批量跑多个样本时，根目录 `runs/data30/` 另有总 `summary.json` / `summary.csv` / `config.json`。
