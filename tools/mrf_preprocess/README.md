# mrf_preprocess — MRF 背景抑制预处理（真实数据验证版）

参考专利 **CN106335646A**《一种干扰式反无人机系统》说明书 S1 图像预处理段（公式 (7)~(13)）
实现的、纯 CPU 可跑的 MRF 类背景抑制工具箱，以及在 ARD-MAV 真实可见光数据上与
DoG / Top-Hat / LCN 的对照评测。

> 结论与取舍见 **`report.md`**（同目录）。本文件只讲怎么跑、参数含义、产出在哪。

## 目录

```
tools/mrf_preprocess/
├── filters.py        # MRF 三/四种变体 + 基线（全部 pure numpy/OpenCV，float32 灰度图）
├── data.py           # ARD-MAV 序列读取：mp4 抽帧 + VOC XML GT（帧号 N ↔ cv2 第 N-1 帧）
├── evaluate.py       # 对照评测：SNR 增益 / AUC / d' / 虚警数 / 耗时
├── quant_check.py    # uint8 + INT8 量化友好性估算
├── make_dataset.py   # 批量产出预处理后的 YOLO 数据集（不改训练脚本）
├── ind_infer.py      # 指示性推理（用现成 best.pt，仅说明输入分布被改动）
├── report.md         # 最终报告（结论先行）
├── results/          # 报告里所有数字的原始产出（*.md 表格 + *.json 明细）
└── README.md
```

## 快速开始

```powershell
$PY = "G:\0-xiandong\rk3576\train\torch\torch\python.exe"

# 1) 单序列冒烟（约 10 s）
& $PY evaluate.py --seqs phantom43 --frames 40

# 2) 三类背景全量对比（10 序列 × 250 帧 × 10 方法，约 25 min，产出 results/final.*）
& $PY evaluate.py --frames 250 --max-dim 12 `
    --grid "mrf_flux:lam=0.5,1" `
    --methods raw,mrf_res,mrf_res_r2,mrf_res_r3,mrf_bgsub_ms,`
mrf_flux_lam0.5,mrf_flux_lam1,dog,tophat,lcn `
    --out results/final.json --md results/final.md

# 3) ρ 加权门控变体（同一批帧，产出 results/gates.*）
& $PY evaluate.py --frames 250 --max-dim 12 `
    --methods raw,mrf_gate_ms `
    --out results/gates.json --md results/gates.md

# 4) 参数扫描（τ / 迭代次数，3 序列 × 80 帧，产出 results/sw_*.md）
& $PY evaluate.py --frames 80 --max-dim 12 --seqs phantom43,phantom56,phantom63 `
    --grid "mrf_gate_ms:tau=8,16" --methods raw,mrf_gate_ms_tau8,mrf_gate_ms_tau16 `
    --out results/sw_tau.json --md results/sw_tau.md
& $PY evaluate.py --frames 80 --max-dim 12 --seqs phantom43,phantom56,phantom63 `
    --grid "mrf_gate_ms:iters=3,12" --methods raw,mrf_gate_ms_iter3,mrf_gate_ms_iter12 `
    --out results/sw_it.json --md results/sw_it.md

# 5) 量化影响估算（产出 results/quant.md）
& $PY quant_check.py --frames 40 --methods raw,mrf_res,mrf_res_r2,mrf_bgsub_ms,mrf_gate_ms,dog,tophat,lcn

# 6) 产出可直接训练的数据集（先用 --dry-run / --limit 试跑）
& $PY make_dataset.py --src G:/0-rk3588/train/ARD-MAV/ARD-MAV-yolo `
    --dst G:/0-rk3588/train/ARD-MAV/ARD-MAV-yolo-mrf-gate `
    --method mrf_gate --set tau=8,iters=3,scale=4 --limit 200
```

> **坑**：`--methods` 只保留 `build_methods()` 里已存在的名字，或 `--grid` 当场生成的名字。
> `--grid` 的后缀是 `键名前 4 个字符 + 值`（`radius` → `_radi2`，`iters` → `_iter3`），
> 写错名字会被静默过滤掉（表现为结果表里少一行）。`mrf_gate_ms_tau16` 这类名字**必须**由
> `--grid "mrf_gate_ms:tau=16"` 生成，不能直接写在 `--methods` 里。

参数扫描用 `--grid "base:key=v1,v2"`（可重复，同一 base 按笛卡尔积展开，变体名自动加后缀），
单点改参用 `--set "base:key=value"`。

## 方法一览

| 名称 | 对应专利/文献 | 公式 | 输出 |
|---|---|---|---|
| `raw` | 不做预处理 | — | 原图（对照组） |
| `mrf_res` | 式 (10) 粗糙度 | `d = mean(N(c)) - x` | 残差图（有符号） |
| `mrf_bgsub` | 式 (7) `u = x - αKx` | 保边背景估计 `bg`，`u = x - bg` | 残差图 |
| `mrf_bgsub_ms` | 同上 + 多尺度 | 在 1/scale 分辨率估背景再上采样 | 残差图 |
| `mrf_gate` | 式 (11)+(12)(13) 势函数做门控 | `u = bg + w(d)·(x-bg)`，`w=1-exp(-(d/τ)²)` | 图像（背景被压平） |
| `mrf_gate_rho` | 同上，门控由专利 ρ 导出 | `w = ρ'(d)/d`，ρ 见下 | 图像 |
| `mrf_flux` | 式 (8) 能量最小化的不动点 | `u = x + λ(I-A)ρ'(d)` | 锐化后的图像 |
| `dog` | 基线 | `G(σ1) - G(σ2)` | 残差图 |
| `tophat` | 基线 | 白帽 − 黑帽 | 残差图 |
| `lcn` | 基线 | `(x-μ_local)/(σ_local+ε)` | 归一化图 |

### 专利势函数 ρ

专利只给了三条性质（式 (12)(13) + 凸/单调/可微），**没给具体函数**。本实现取

```
ρ'(t) = τ·(u - atan(u)),  u = t/τ      ⇒  ρ'(t)/t = 1 - atan(u)/u
```

- `t→0` 时 `ρ'(t)/t → 0`（满足性质 1：同质背景几乎不被处理 → 相对压制）；
- `t→∞` 时 `ρ'(t)/t → 1 = M`（满足性质 2：大粗糙度（目标）被保留/放大）；
- `ρ''(t) = u²/(1+u²) ≥ 0`，`ρ'` 单调递增且 `ρ(0)=0`，C^∞（满足性质 3）。

注意：Huber / Charbonnier / Tukey / Geman-McClure **不满足**性质 1、2——它们的
`ρ'(t)/t` 在 `t=0` 取最大值、随 `t` 增大衰减到 0，方向正好相反。它们只能用于
“前向扩散”的**背景估计**（`bg_smooth` 的 `weight` 参数），不能当作专利的那个 ρ。
详见报告第 1 节。

## 关键参数

| 参数 | 含义 | 建议 |
|---|---|---|
| `nbhd` | 4 / 8 邻域 | 8（专利为二阶邻域） |
| `radius` | 邻域半径，1=3×3，2=5×5，3=7×7 | **必须与目标尺寸匹配**：3~5 px 目标用 1，7~10 px 用 2~3 |
| `tau` | 粗糙度阈值（灰度级） | 约取目标局部对比度的 1/3~1/2；实测 8~16 较稳 |
| `iters` | 背景扩散迭代次数 | 3~6；实测 3 / 6 / 12 的检测指标几乎一致，3 更省算力且天空 SNR 更好 |
| `scale` | 背景估计的降采样倍数 | 2~4；**既提速又保护小目标**，推荐 |
| `weight` | 前向扩散权重 `charbonnier/geman/tukey` | `tukey`（硬截断，不让目标渗进背景） |
| `gate` | `exp` / `rho` | `exp` 更平滑，`rho` 更贴专利 |
| `polarity` | `signed` / `abs` | 训练用 `signed`（保留明暗极性） |

## 评测口径（`evaluate.py`）

所有方法输出都过**同一个、与模型无关的读数算子**，避免“给某个方法偷偷加了个高通”：

```
D(x) = |x - box_mean_11(x)|
L(x) = D(x) / (box_std_31(D) + 0.5)        # CFAR 式局部归一化检测分
```

- **SNR 增益**：GT 框内 `max|x - ring_median|` / 环绕环带（外扩 4 px 起）的稳健 σ，
  逐目标算，再取比值的中位数；`gain<0.8` 是“被抑制掉”的比例。
- **AUC / d'**：GT 峰值（正样本）vs 全帧背景局部极大值（负样本）。
- **rec@FA=1**：每帧只允许 1 个虚假峰值时的检出率（极严格工作点）。
- **FA@rec.9 / FA@rec.5**：达到 90% / 50% 检出率时每帧的虚假峰值数（对应“降阈值全是误检”的真实痛点）。
- **耗时**：只计预处理本身（不含读数算子），P50/P95，1920×1080、x86 单线程。

GT 采用 VOC XML，`Annotations\<seq>\<seq>_NNNN.xml` 的 `NNNN` 对应 `cv2` 第 `NNNN-1` 帧；
`--max-dim 12` 只保留目标长边 ≤12 px 的帧。

## 产出

- 报告：`report.md`（结论 + 专利数学形式 + 对比表 + 建议 + 局限）。
- 评测原始产出：`results/` 下的 `*.md`（表格）与 `*.json`（逐序列明细），
  包括 `final`（10 序列 × 250 帧 × 10 方法）、`gates`（门控变体）、`sw_tau`/`sw_it`（参数扫描）、
  `quant.md`（量化估算）、`sweep2.json`（早期 3 序列 × 60 帧扫描）、`ind_infer.txt`（指示性推理）。
- 数据集：`--dst` 下的 `images/{train,val}`、`labels/{train,val}`、`data-mrf.yaml`。
  `data-mrf.yaml` 里记录了 `preprocess.method/params`，便于复现；
  训练侧只需 `yolo train data=<dst>/data-mrf.yaml ...`，**不需要改任何训练脚本**。
- 抽帧：`evaluate.py` 直接流式解码，不落盘；只有 `make_dataset.py` 会写图。
  实测 1920×1080 JPEG q95 为 **863 KB/图**（原图 611 KB）、生成 **0.42 s/图**（3 通道）。

## 已知限制

- 3~6 px 目标在 ARD-MAV 里极少（≤6 px 占比 0.2%~10%，见报告），主要结论落在 7~9 px 档。
- 没有用预处理后的数据重新训练，评测是**无模型**的代理指标，不代表 mAP 收益。
- 耗时是 x86 实测，RK3576 板端未实测。
