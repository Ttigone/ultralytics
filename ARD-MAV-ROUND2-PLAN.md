# ARD-MAV 第二轮：按尺寸分桶重做基线

本文档面向「你自己在本机执行训练」的场景。所有命令都是 PowerShell，解释器固定为：

```powershell
$PY = "G:\0-xiandong\rk3576\train\torch\torch\python.exe"   # Py3.9.25 + torch 2.1.0+cu121 + RTX 3080 10GB
Set-Location G:\0-xiandong\rk3576\ultralytice_self\ultralytics
```

> 不要用 `yolo` 命令：site-packages 里的 ultralytics 8.4.15 会遮蔽本仓库（8.4.174），且不支持 `nwd` / DWR / 新参数。
> 需要 CLI 形式时用 `& $PY -c "from ultralytics.cfg import entrypoint; entrypoint(r'yolo ...')"`。

---

## 0. 结论速览

| 问题 | 结论 |
| --- | --- |
| 上一轮是否 `train = val = calib/dataset.txt` | **不成立**。训练用的是 `ARD-MAV-yolo/data.yaml`（`images/train` 8478 张 / `images/val` 2283 张），`calib/data.yaml` 的双清单只用于 INT8 校准时读图 |
| train / val 序列是否重叠 | **不重叠**，交集为空，且不存在「同序列相邻帧跨 split」（逐帧检查 = 0） |
| 过拟合 + 召回不足的根因是否来自划分 | 不是划分泄漏；问题在于 val 里极小目标占比过高、训练轮次/早停配置、以及 640 输入下 4–8px 目标的物理像素不足 |
| 新划分 | 序列级 70/15/15 = 42/9/9 序列，7605 / 1539 / 1617 帧，`seed=2026` |
| 训练方案 | 640 主力 + 960/1280 对照；`patience=20`；mosaic 0.5 且最后 20 epoch 关闭；mixup 0；`scale=0.3`；`multi_scale=0` |
| 评测 | 分桶 Recall / Precision / FP-per-frame + conf 扫描 + 序列连续性，单次推理产出全部阈值 |

### 用户产生「train=val」怀疑的合理解释

`calib/data.yaml` 里确实写着：

```yaml
train: calib/dataset.txt
val: calib/dataset.txt
```

但它是 **INT8 校准专用**（400 张极小目标抽样清单，判据「全框最长边 ≤16px」，候选 4830 → 抽样 400，seed 2026），不是训练入口。
如果只看这个文件，很容易误判成训练集=验证集。

---

## 1. 第一步的核查证据

### 1.1 数据集 YAML

[data.yaml](file:///G:/0-rk3588/train/ARD-MAV/ARD-MAV-yolo/data.yaml)：

```
path: g:/0-rk3588/train/ARD-MAV/ARD-MAV-yolo
train: images/train      # 8478 张
val:   images/val        # 2283 张
nc: 1
names: ['Drone']
```

### 1.2 实际训练参数（`runs/**/args.yaml` + 日志）

`runs\baseline_p2\args.yaml`：

```
4:data: g:/0-rk3588/train/ARD-MAV/ARD-MAV-yolo/data.yaml
5:epochs: 12
7:patience: 100
8:batch: 16
9:imgsz: 640
12:cache: false
14:workers: 2
18:pretrained: false
22:seed: 0
27:close_mosaic: 10
29:amp: true
33:multi_scale: 0.0
104:translate: 0.1
105:scale: 0.5
109:fliplr: 0.5
111:mosaic: 1.0
112:mixup: 0.0
```

`dwr_p2_e12` / `dwr_nwd_p2_e12` 同值；`dwr_nwd_p2_e200` / `p2_transfer_e200` 为 `epochs: 200`、`pretrained: true`。

`baseline_p2.log` 的关键行：

```
train: Scanning ...\labels\train.cache... 8478 images, 77 backgrounds
val:   Scanning ...\labels\val.cache...   2283 images, 27 backgrounds
12 epochs completed in 0.405 hours
AMP checks failed → 回退 FP32
```

→ 用的是标准 data.yaml，**不是 calib 清单**；12 epoch 全部跑完，`patience=100` 实际上从未触发（早停没生效过）。

### 1.3 序列核查（逐帧统计，非估算）

```
train imgs 8478 seqs 48
val   imgs 2283 seqs 12
seq overlap []                      # 交集为空
adjacent cross-split frames 0       # 同序列相邻帧跨 split 的帧数 = 0
train seqs: phantom04,05,06,08,10,12,13,14,19,20,22,23,29,30,32,36,39,40,41,42,44,45,46,47,
            50,51,54,55,57,58,59,60,63,64,65,66,70,71,73,74,76,77,78,79,81,84,85,86
val   seqs: phantom02,09,16,28,38,43,49,56,61,68,75,80
```

结论：**没有视频级泄漏**，用户在这一点上的怀疑不成立。

### 1.4 但确实存在两个真问题

1. **val 与 train 的尺度分布不一致**：旧 val 的 ≤16px 框占 54.2%，与 train 的 55.9% 接近，但 4–8px 占比是 train 的 2 倍（12.5% vs 5.7%），导致 val 指标被极小目标主导。
2. **旧模型与新 val 不可比**：新 val 里有 7 个序列（04/20/46/54/71/74/79）是旧 baseline 的训练序列。所以：
   - 旧 checkpoint 只能用来「验证脚本能跑」，**不能**作为新划分的基线数字；
   - 新一轮的 `pretrained=True` 只能用 COCO 的 `weights/yolo26n.pt`，**不要**用 `runs/p2_transfer_e200/weights/best.pt`。

---

## 2. 新的序列级 70/15/15 划分（已生成并跑通）

脚本：[make_splits.py](file:///g:/0-xiandong/rk3576/ultralytice_self/ultralytics/tools/ard_mav/make_splits.py)，产物在 `G:\0-rk3588\train\ARD-MAV\ARD-MAV-yolo\splits\`：

```
train: sequences=42 frames= 7605 (70.7%) boxes= 7514 tiny<=16px=55.2%
       buckets={'<4px': 2, '4-8px': 550, '8-16px': 3594, '16-32px': 1850, '>32px': 1518}
val:   sequences= 9 frames= 1539 (14.3%) boxes= 1541 tiny<=16px=76.9%
       buckets={'<4px': 1, '4-8px': 172, '8-16px': 1012, '16-32px': 173, '>32px': 183}
test:  sequences= 9 frames= 1617 (15.0%) boxes= 1604 tiny<=16px=44.8%
       buckets={'<4px': 2, '4-8px': 130, '8-16px': 587, '16-32px': 617, '>32px': 268}
```

序列名单（`split_manifest.json` 实际内容）：

| split | 序列（42 / 9 / 9） |
| --- | --- |
| train | phantom02, 05, 08, 09, 10, 13, 14, 16, 19, 22, 23, 28, 29, 30, 32, 36, 39, 40, 41, 42, 43, 44, 45, 47, 49, 50, 51, 55, 56, 58, 59, 60, 61, 63, 64, 65, 66, 73, 75, 77, 78, 84 |
| val | phantom04, 06, 20, 46, 54, 68, 71, 74, 79 |
| test | phantom12, 38, 57, 70, 76, 80, 81, 85, 86 |

旧 `images/val` 的 12 个序列里，02/09/16/28/43/49/56/61/75 被换进 train，06/68 留在 val；旧 train 的 04/20/46/54/71/74/79 被换进 val。这个交换是分层再平衡的结果，因此**旧 checkpoint 在新 val 上有 7 个序列属于其训练集**（见 §1.4）。

（全部按 `splits/*.txt` 绝对路径清单引用，**没有复制或移动任何图片**；`images/train` / `images/val` 目录只是图片的物理存放位置。）

设计说明：

- 先按帧数做贪心分配（42/9/9，帧份额 70.7/14.3/15.0%），再做**单序列交换的分层再平衡**：目标函数是三个 split 的尺寸桶分布与全局分布的 L1 距离，约束是每个 split 的帧份额偏离请求份额不超过 ±2.5 个百分点。交换前 val 的 tiny 占比是 88.6%、`>32px` 只有 28 个框；交换后降到 76.9%、`>32px` 183 个框。
- val 仍有 76.9% 的 ≤16px 目标，**低于/高于 train 的 55.2%**。这是 9 个序列的离散性决定的，再强行交换就会破坏帧份额。所以：**过程监控看 val（同一序列内可比），最终结论看 test（44.8%，最接近全局 56.7%）**。
- val/test 的序列来自不同拍摄（phantom 编号即不同架次/场景），彼此不共享背景；test 的 `>32px` 框 268 个也足够评估大目标是否被牺牲。

数据入口（新）：

```yaml
# G:\0-rk3588\train\ARD-MAV\ARD-MAV-yolo\splits\data.yaml
path: G:/0-rk3588/train/ARD-MAV/ARD-MAV-yolo
train: splits/train.txt
val: splits/val.txt
test: splits/test.txt
nc: 1
names: ['Drone']
```

重新生成（想看不同 seed 时）：

```powershell
& $PY tools\ard_mav\make_splits.py --root g:/0-rk3588/train/ARD-MAV/ARD-MAV-yolo --seed 2026
```

---

## 3. 其他 RGB 数据集盘点（`tools/ard_mav/inventory_datasets.py` 实跑，每 split 抽样 300 个标注）

| 数据集 | 存在 | 标注格式 | 类别 | 图像数 | 目标尺寸（≤16px 占比） | 视频序列 | 重复风险 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARD-MAV | 是 | YOLO txt | 1 类 `Drone` | 10761（1920×1080 为主） | 55.9%（80% 在 8–16px） | 是，60 序列 | 已按序列隔离 |
| DUT-Anti-UAV | 是 | VOC XML + 转换后的 YOLO txt | 1 类 `uav` | 10,000 组（yolo_dataset: 5200/2600/2200） | 2.3%（>32px 占 69%） | 否（独立帧，部分连拍） | 中：与 ARD 同一类目标，但尺度大 10 倍 |
| 3rd_anti_uav | 是 | YOLO txt（另有 COCO json/CSV） | 1 类 `uav` | 115,792（yolo: 86324/29468，640×512） | 36.4%（16–32px 19%, >32px 44%） | 是（大量连续帧） | 高：86k 训练帧，clips 高度相似 |
| uav-anti-uav | **是**（不是用户以为的「可能不存在」） | VOC XML(Train/Test) + 转换的 YOLO txt | 1 类 `uav` | 76,637 张 jpg（yolo: 52536/6213/16058 = 74,807，856×480） | 3.9%（>32px 87%） | 是（另有 UAV_Tracking 跟踪集，1840 mp4） | 中高：与 ARD 类别相同、尺度不同 |
| visiodet (VisioDECT) | 是 | VOC XML（+ 每机型/天气的 YOLO txt，class id 15） | **多类**：按机型+光照命名（`DJI_Phantom_Cloudy`…） | 19458（1920×1080/1080×1920/2400×1080 混合） | **0%**（97.2% 是 >32px） | 否 | 低（不同机型/天气），但语义与 `Drone` 不一致 |
| ant_uav | 是 | **无检测框**：636 mp4 + 属性 json（`["FM","SV","OV","TC"]`） | 无 | 0 张抽帧图 | — | 是（视频） | 不可用于检测训练 |

### 该不该合并：结论

**不要整体合并。** 推荐做法：

1. **训练侧可以增广，验证侧绝不混**：把 `3rd_anti_uav` 与 `uav-anti-uav` 的 `train` 帧作为**额外负样本/多尺度样本**并入 train（约 +20k 帧抽样，而不是 11 万全量），val/test 始终只用 ARD-MAV 的 9+9 个序列。
2. **DUT-Anti-UAV 优先级最低**：它的 69% 目标 >32px，会把模型推向「靠大目标学特征」，对 4–8px 毫无帮助；要并也只建议按 1:10 降采样并只保留 ≤16px 的框。
3. **VisioDECT 不合并**：类别是「机型_光照」的多类语义，直接映射成单一 `Drone` 会引入标注噪声，而且 97% 的框 >32px，与当前任务无关。
4. **ant_uav 不可用**：没有检测框，只适合做「有/无目标」的视频级验证，需先人工抽帧标注。
5. 合并后必须**重新计算尺寸桶分布**并重做 `make_splits` 之外的采样记录；合并只影响 train，评测口径不变，否则跨数据集指标无法比较。

理由一句话：合并能补充小目标的**外观多样性**，但会污染「尺度分布」这一核心评测维度；在当前召回不足的阶段，优先保证评测可信，再谈增广。

---

## 4. 训练参数最终取值（已落到 `tools/ard_mav/train.py`）

`train.py` 现在把每个影响小目标的增强都暴露成显式参数（默认值即第二轮方案）：

| 参数 | 默认（第二轮） | 上一轮实际 | 取值理由 |
| --- | --- | --- | --- |
| `--data` | `splits/data.yaml` | `data.yaml` | 序列级隔离 |
| `--epochs` | 100 | 12 / 200 | 12 不够收敛，200 在 `patience=20` 下会自动截断 |
| `--patience` | **20** | 100（从未触发） | 20 ep 无提升就停，best.pt 自动保存 |
| `--imgsz` | 640 | 640 | 见下方分辨率对照 |
| `--batch` | 16 | 16 | 640 下 10GB 可跑；960→8，1280→4 |
| `--workers` | 2 | 2 | 4 曾因物理内存耗尽导致吞吐崩塌 |
| `--cache` | none | false | `ram` 需 ~47GB（7605×1920×1080×3B）；`disk` 会生成同量级 `.npy`，本盘只剩 91GB，不建议 |
| `--mosaic` | **0.5** | 1.0 | 4 图拼接会把 8px 目标再缩一半 |
| `--close-mosaic` | **20** | 10 | 最后 20 ep 用单图尺度，贴近推理分布 |
| `--mixup` | **0.0** | 0.0 | 叠图会把极小目标彻底抹掉；对照组再试 0.1 |
| `--scale` | **0.3** | 0.5 | ±50% 缩放会把 8px 压到 4px，±30% 只到 5.6px |
| `--multi_scale` | **0.0** | 0.0 | 一旦开启（如 0.4），640 会退到 384，8px→4.8px |
| `--perspective` | 0.0 | 0.0 | 透视变形会改变小目标的像素形状 |
| `--degrees` | 0.0 | 0.0 | 旋转对小目标有效但会引入插值糊化，先不做 |
| `--translate` | 0.1 | 0.1 | 保持 |
| `--fliplr` | 0.5 | 0.5 | 保持 |
| `--erasing` | **0.0** | 0.4 | 随机擦除可能直接删掉唯一的 8px 目标 |
| `--amp` | true | true | 但本机 AMP 自检会失败并回退 FP32（见 `baseline_p2.log`） |
| `--max-det` | 300 | 300 | 小目标密集场景可设 500，但会增加后处理耗时 |

> 关于 `lr0`：本机实测 `optimizer=auto` 会忽略 `lr0=0.01`，改用 `AdamW(lr=0.002, weight_decay=0.0005)`。过拟合明显时用 `--weight-decay 0.001 --lr0 0.001` 做一次对照，而不是继续加 epoch。

### 4.1 imgsz 对照（640 / 960 / 1280）

| imgsz | batch | 显存（估） | 每 epoch（估，FP32） | 100 epoch | 说明 |
| --- | --- | --- | --- | --- | --- |
| 640 | 16 | ~4–6 GB | **2.0 min**（实测 12 ep = 0.405 h） | ~3.4 h | 8px 目标在 640 下只剩 2.7px（P2 stride 4 → 0.67 格） |
| 960 | 8 | ~7–8 GB | ~4.5 min（像素 2.25×） | ~7.5 h | 8px → 4px，P2 特征图上 1 格 |
| 1280 | 4 | ~8–9.5 GB（临界） | ~8 min（像素 4×） | ~13–14 h | 8px → 5.3px，小目标最友好 |

- 1280 若 OOM：先 `--batch 3`，再 `--batch 2`；不要为了 1280 把 `workers` 加到 4（内存会先崩）。
- 时间都建立在「AMP 回退 FP32」的实测吞吐上；如果 AMP 自检通过，可再快 1.3–1.8 倍。
- **判断是否值得上 1280**：只有 640→960 的 8–16px 分桶 recall 提升 ≥10% 相对值，才继续跑 1280；否则优先调增强与数据。

### 4.2 pretrained True / False 的 A/B（三组必须同划分、同 imgsz、同增强）

```powershell
# A：从零
& $PY tools\ard_mav\train.py --variant baseline --imgsz 640 --batch 16 --epochs 100 --patience 20 --name r2_base_fs --out G:\0-rk3588\train\ARD-MAV\runs

# B：COCO 预训练（只加载 backbone/neck 同名权重，P2 头随机初始化）
& $PY tools\ard_mav\train.py --variant baseline --imgsz 640 --batch 16 --epochs 100 --patience 20 --weights weights\yolo26n.pt --name r2_base_pt --out G:\0-rk3588\train\ARD-MAV\runs

# C（可选）：DWR+NWD 同条件
& $PY tools\ard_mav\train.py --variant dwr_nwd --imgsz 640 --batch 16 --epochs 100 --patience 20 --weights weights\yolo26n.pt --name r2_dwr_nwd_pt --out G:\0-rk3588\train\ARD-MAV\runs
```

### 4.3 什么时候该停

- `patience=20` 触发（`EarlyStopping: no improvement for 20 epochs`）→ 直接停，用 `best.pt`。
- val 的 8–16px recall 在第 40 epoch 后仍低于 0.20 → 停下做数据/标签核查，不要继续扫 imgsz。
- train loss 还在降、val mAP 已连降 8 epoch → 过拟合，缩短 epoch 或加 `weight_decay`，不要加数据量以外的花招。
- 每 epoch 打印的 `cls_loss` 若是 `nan`/爆炸 → 先关 `amp`（`--amp false`）复现。

---

## 5. 评测升级（`tools/ard_mav/eval_size_buckets.py`）

一次推理（`conf=0.001` 地板）即可产出全部阈值，避免重复推理：

```powershell
& $PY tools\ard_mav\eval_size_buckets.py --weights G:\0-rk3588\train\ARD-MAV\runs\r2_base_pt\weights\best.pt `
  --split val --imgsz 640 --experiment r2_base_pt --csv G:\0-rk3588\train\ARD-MAV\eval_round2.csv
```

新增能力：

1. **分桶 Precision**：TP 按 GT 尺寸归桶，FP 按预测框尺寸归桶（定义写在脚本末尾，避免歧义）。
2. **FP/frame**：`fp / images`，整体与分桶都给，便于直接换算「每 100 帧误报」。
3. **conf 扫描**：0.05 / 0.10 / 0.15 / 0.20 / 0.30 / 0.50，每个阈值同时给 Recall、Precision、FP/frame。
4. **序列连续性**（在 `--continuity-conf`，默认 0.15）：
   - `frame_hit`：该帧至少 1 个 GT 被 IoU≥0.5 命中的比例；
   - `adjacent_co_hit`：命中帧中「上一帧也命中」的比例（连续检出率）；
   - `breaks`：检出段数 - 有检出的序列数（每条序列一段连续检出 = 0 断裂）；
   - 简化点：同帧多目标被合并成「有/无」信号，不做跨帧 ID 关联（真正的 ID 级断裂需跟踪器）。
5. **CSV**：一行 =（实验 × 阈值 × 桶），追加写入，直接横向对比多个实验。

### 示例输出

用**旧** `baseline_p2/best.pt` 在**新 val** 上跑（仅用于展示脚本行为，数字不可作为基线，原因见 §1.4）：

```
experiment=baseline_p2_legacy weights=...\baseline_p2\weights\best.pt split=val images=1539 imgsz=640
  conf |  recall  precision  fp/frame | <4px R 4-8px R 8-16px R 16-32px R >32px R
  0.05 |  0.1979   0.2523      0.587 | 0.000 0.017 0.160 0.191 0.585
  0.10 |  0.1441   0.3581      0.259 | 0.000 0.012 0.103 0.133 0.508
  0.15 |  0.1110   0.4430      0.140 | 0.000 0.006 0.069 0.075 0.475
  0.20 |  0.0896   0.5188      0.083 | 0.000 0.006 0.047 0.052 0.437
  0.30 |  0.0610   0.6483      0.033 | 0.000 0.006 0.019 0.017 0.388
  0.50 |  0.0370   0.8906      0.004 | 0.000 0.000 0.004 0.006 0.284
continuity(conf=0.15): sequences=9 frame_hit=0.111 adjacent_co_hit=0.067 detected_segments=68 breaks=60
```

这张表已经说明问题：**降到 conf=0.05 时 FP/frame 是 0.587（每 2 帧 1 个误报），而 recall 也只有 0.19**。也就是说上一轮的模型在「可部署的误报预算」下召回极低，这与用户看到的「conf=0 时 recall 只有 0.57」的 P-R 曲线一致。

CSV 片段：

```
experiment,weights,imgsz,split,conf,iou,bucket,images,gt,tp,pred,fp,recall,precision,fp_per_frame
baseline_p2_legacy,G:/.../best.pt,640,val,0.05,0.5,8-16px,1539,1012,162,389,227,0.1601,0.4165,0.1475
baseline_p2_legacy,G:/.../best.pt,640,val,0.20,0.5,all,1539,1541,138,266,128,0.0896,0.5188,0.0832
```

---

## 6. 阶段计划

### 阶段 0：前置校验

```powershell
& $PY -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
& $PY tools\ard_mav\train.py --help
& $PY -c "from pathlib import Path;p=Path(r'G:\0-rk3588\train\ARD-MAV\ARD-MAV-yolo\splits');print([(f.name,len(f.read_text().splitlines())) for f in p.glob('*.txt')])"
```

验收：`7605 / 1539 / 1617` 三行；`train.py --help` 能看到 `--mosaic/--patience/--pretrained`。

### 阶段 1：640 从零基线

```powershell
& $PY tools\ard_mav\train.py --variant baseline --imgsz 640 --batch 16 --workers 2 --epochs 100 --patience 20 `
  --name r2_base_fs --out G:\0-rk3588\train\ARD-MAV\runs
```

验收（val）：
- `8-16px recall@conf0.05 ≥ 0.35`；
- `all FP/frame@conf0.05 ≤ 0.60`；
- `adjacent_co_hit ≥ 0.50`；
- 若 best.pt 落在 epoch ≤ 25 就停 → 视为训练不足/学习率问题，先按 §4.3 处理，再进入阶段 2。

### 阶段 2：分辨率对照 960（~7.5 h），必要时 1280

```powershell
python ard_mav\train.py --variant baseline --imgsz 960 --batch 8 --workers 2 --epochs 100 --plots --patience 20 --name r2_base_960 --out G:\0-rk3588\train\ARD-MAV\runs 
python ard_mav\train.py --variant baseline --imgsz 960 --batch 8 --workers 2 --epochs 100 --plots --patience 20 --name r2_base_960 --out G:\0-rk3588\train\ARD-MAV\runs 

# 仅当 960 相对 640 的 8-16px recall 提升 ≥10% 时才跑：
python ard_mav\train.py --variant baseline --imgsz 1280 --batch 4 --workers 2 --epochs 100 --plots --patience 20 --name r2_base_1280 --out G:\0-rk3588\train\ARD-MAV\runs
```

判据：`8-16px recall` 提升 ≥10%（相对）且 `FP/frame` 上涨 ≤50%。如果只是 recall 涨、FP 暴涨，则属于阈值可调，先看 conf 曲线再决定。

### 阶段 2b（新增）：原图 + DoG 附加通道

依据 `tools/mrf_preprocess/report.md`（10 序列 × 250 帧真实 ARD-MAV）：MRF 残差类在建筑/地面把 GT 环带 SNR 抬到 1.5–1.95×，但同召回下每帧虚警是原图的 2–3.7 倍，**不进默认前处理**；门控变体对低对比度天空会吃掉 21–67% 的目标；唯一在高召回工作点上全面胜出的是 **DoG(σ1=1.0, σ2=3.0)**（FA@rec.5：天空 15 / 建筑 52 / 地面 46，原图 69/139/106），推荐作为**附加输入通道**而非替换原图。

阶段 2b 与阶段 3 相互独立：阶段 3 的 pretrained A/B 仍按原顺序执行；若 2b 证明有效，阶段 3 结束后可用 DoG 数据集复跑「最佳 pretrained 臂」。

#### 2b.0 接口核对

| 事实 | 值 |
| --- | --- |
| DoG 实现 | `tools/mrf_preprocess/filters.py::dog(x, s1=1.0, s2=3.0)`，单通道 float32(0–255) → 同尺寸 float32 残差（带符号） |
| `make_dataset.py` 的输入 | 只吃 `images/{split}` **目录**（`glob`），不支持 `.txt` 清单；输出 3 通道整图 + `data-mrf.yaml` |
| 因此是否需要新脚本 | 需要：我们的划分是 `splits/*.txt`，且要「原图 + DoG」双流对齐。新增 [make_dog_channel.py](file:///g:/0-xiandong/rk3576/ultralytice_self/ultralytics/tools/ard_mav/make_dog_channel.py)（`import filters` 只读复用，不修改 `tools/mrf_preprocess/`） |
| 对齐规则 | 输出文件名 = 原图 stem；帧数必须等于 7605 / 1539 / 1617；label 只做复制，**不重算、不改任何框坐标** |

#### 2b.1 数据准备命令 + 实测耗时与磁盘

本机实测（真跑样本，非估算）：

```
# dog 模式 120 帧：62.8 ms/帧，103.3 MB → 0.86 MB/帧
# compose 模式 40 帧：77.8 ms/帧，39.2 MB → 0.98 MB/帧
```

全量推算：10,761 帧 → `dog` 约 11–14 min / ≈9.3 GB；`compose` 约 14–18 min / ≈10.7 GB。G 盘余量 91 GB，两套并存约 20 GB 可接受（也可只留 compose，省 10 GB）。

```powershell
# (a) 4 通道素材：单通道 DoG PNG（供未来的 4 通道 loader 使用）
& $PY tools\ard_mav\make_dog_channel.py --mode dog --dst G:\0-rk3588\train\ARD-MAV\ARD-MAV-dog

# (b) 3 通道重构数据集：存储 BGR = [DoG, 原G, 原R]；Ultralytics 载入后做 BGR→RGB，因此 DoG 落在网络第 0 通道
& $PY tools\ard_mav\make_dog_channel.py --mode compose --dst G:\0-rk3588\train\ARD-MAV\ARD-MAV-dog-rgb

# 对照臂：把 DoG 换成常数 128（隔离"只是输入分布变了"）
& $PY tools\ard_mav\make_dog_channel.py --mode compose --const --dst G:\0-rk3588\train\ARD-MAV\ARD-MAV-const-rgb
```

产物：

```
ARD-MAV-dog-rgb\images\{train,val,test}\<stem>.jpg
ARD-MAV-dog-rgb\labels\{train,val,test}\<stem>.txt   # 从原 labels 复制
ARD-MAV-dog-rgb\data-dog.yaml                        # path/train/val/test + nc=1 Drone
ARD-MAV-dog-rgb\dog_manifest.json                    # 逐帧 source 映射，便于核对
```

验收：三个 split 的图片数 = 7605/1539/1617；label 数相同；随机抽 3 帧把 `data-dog.yaml` 喂给 `yolo` 的 `check_det_dataset`，确认 label 路径解析正常；DoG PNG 与 JPEG 的 stem 必须一一对应（`dog_manifest.json` 即证据）。

#### 2b.2 融合方案与取舍

| 方案 | 训练侧改动 | 预训练加载 | RKNN / 板端预处理 | 结论 |
| --- | --- | --- | --- | --- |
| (a) 4 通道输入 | 首层 `Conv(4, c1, 3)`；`BaseDataset` 增加 4 通道读取（当前 `channels` 仅支持 1/3，`IMREAD_COLOR` 会丢掉第 4 通道）；`cfg/default.yaml` 加 `channels: 4` | 首层权重由 `(16,3,3,3)` 变 `(16,4,3,3)`，`model.load('yolo26n.pt')` 会 shape mismatch；需自定义：复制 `w[:, :3]`、第 4 通道**零初始化**，等价于"初始时完全不信 DoG" | 输入张量 4 通道，RGA 常规 RGB888 转换只处理 3 通道 → 需额外一次 4 通道拼接（CPU memcpy 或 RGA 4 通道 blit）；校准集也必须 4 通道；RKNN 允许 4 通道输入但量化/校验都要跟着改 | 改动跨 `data/base.py`、`nn/tasks.py`、`cfg/default.yaml`，**本轮约束不允许改这些文件**，列为"待授权" |
| (b) 3 通道重构（B=DoG） | 无。`data-dog.yaml` 直接可训 | 无变化（首层仍是 3 通道） | 仍是 RGB888 3 通道输入，RKNN/量化不用改；但**板端仍要自己算 DoG**（见下） | **推荐**：本轮先用它判定"DoG 有没有用" |
| （仅记录，不实施）图内固定 DoG | 在网络最前面加两层固定高斯 depthwise conv（σ1/σ2），DoG 由 NPU 计算 | 首层前插模块，权重为常量、不训练 | 板端零预处理成本、零额外通道，RKNN 图里自带 DoG | 部署最优，但属结构改动，超出本轮范围 |

关于 (b) 的代价：丢失蓝通道（≈11% 亮度 + 部分色差），目标与天空的颜色线索变弱；但 DoG 是带通，对云/建筑边缘有抑制，属于"用一条颜色通道换一路带通特征"。

**板端能不能算 DoG（判断）**：不建议放在板端 CPU 上实时算。1080p 单通道两次高斯 + 相减 + 映射，ARM 上约 30–60 ms/帧（保守估算），而现有整条链路 `infer_ms=173.7 / lat_ms=228.1`，直接吃掉 15–25% 时延预算，且与本项目已暴露的后处理瓶颈叠加。可行的三条路：
1. 训练/离线用，部署改回原图（DoG 只作为"提高召回的训练增强"）；
2. 板上走 RGA/ISP 硬件模糊（若 RGA 的 Gaussian 路径可用，成本可降到个位数 ms，需实测）；
3. 把 DoG 做成图内固定 conv 交给 NPU（方案 3，板端成本≈0，但要改结构并重训）。

#### 2b.3 训练配置

```powershell
# 主实验：与阶段 1 唯一差异是 --data，其余全部沿用默认（mosaic 0.5 / close 20 / mixup 0 / scale 0.3 / multi_scale 0 / seed 0）
& $PY tools\ard_mav\train.py --variant baseline `
  --data G:\0-rk3588\train\ARD-MAV\ARD-MAV-dog-rgb\data-dog.yaml `
  --imgsz 640 --batch 16 --workers 2 --epochs 100 --patience 20 --name r2_dog640_fs `
  --out G:\0-rk3588\train\ARD-MAV\runs

# 对照臂：常数通道（同 split、同 imgsz、同增强、同从零起点）
& $PY tools\ard_mav\train.py --variant baseline `
  --data G:\0-rk3588\train\ARD-MAV\ARD-MAV-const-rgb\data-dog.yaml `
  --imgsz 640 --batch 16 --workers 2 --epochs 100 --patience 20 --name r2_const640_fs `
  --out G:\0-rk3588\train\ARD-MAV\runs

# 仅当 640 通过验收，再在 960 复验（对照 = 阶段 2 的 r2_base_960）
& $PY tools\ard_mav\train.py --variant baseline `
  --data G:\0-rk3588\train\ARD-MAV\ARD-MAV-dog-rgb\data-dog.yaml `
  --imgsz 960 --batch 8 --workers 2 --epochs 100 --patience 20 --name r2_dog960_fs `
  --out G:\0-rk3588\train\ARD-MAV\runs
```

控制变量说明：**与阶段 1 `r2_base_fs` 对照**（同 split/同 imgsz/同增强/同 seed/同从零起点），不与阶段 3 的 pretrained 臂对照；若要用 pretrained，必须同时补一条 `r2_base_pt` 的同条件复跑，不能拿旧数字比。

#### 2b.4 验收与判读

评测命令（与其它阶段完全一致）：

```powershell
& $PY tools\ard_mav\eval_size_buckets.py --weights G:\0-rk3588\train\ARD-MAV\runs\r2_dog640_fs\weights\best.pt `
  --data G:\0-rk3588\train\ARD-MAV\ARD-MAV-dog-rgb\data-dog.yaml --split val --imgsz 640 `
  --experiment r2_dog640_fs --csv G:\0-rk3588\train\ARD-MAV\eval_round2.csv
```

> 注意：`--data` 必须指向**同一套 DoG 数据集**，否则 `eval` 会用原图推理 DoG 模型，结果无意义。

- 有效提升：`8-16px Recall@0.05` 相对 +15% 且 `FP/frame@0.05` 增幅 ≤30%；或 R 持平且 `FP/frame` −30%。
- 辅助：`frame_hit` 相对 +10% 且 `breaks` 下降 ≥20%。
- **判定"DoG 真的帮到检测"的三个必要条件（同时满足）**：
  1. 同 split、同 imgsz、同增强、同 seed、同从零起点（不可跨阶段比较、不可跨 imgsz 比较）；
  2. 增益集中在 **8-16px 桶**（σ2=3 的带通正对 7–9px），而 4-8px 与 >32px 的变化 ≤5%；
  3. 常数通道对照臂 `r2_const640_fs` 的增益不足 DoG 臂的一半 —— 否则说明"只是输入分布变了"。
- **不认 mAP50**：mAP 被 16-32 / >32 桶稀释，本阶段只看分桶 R 与 FP/frame 的组合变化。

#### 2b.5 失败判据

| 条件 | 结论 |
| --- | --- |
| 640 下 `8-16px R@0.05` 相对基线 <+5%，且 `FP/frame@0.05` 未下降 | DoG 无效，放弃该通道 |
| 常数通道对照拿到 ≥2/3 的增益 | 判为分布漂移/初始化噪声，DoG 未被验证 |
| 4-8px 或 >32px 桶 R 相对下降 >5% | 算子破坏其它尺度，拒绝 |
| 960 复验增益 <+5%（640 有效但 960 消失） | 视为尺度不匹配，停止扩展 |
| 板端 DoG 增量 >25 ms/帧且无 RGA 方案 | 即使指标有效，也不作为默认部署路径（只保留训练增强） |

#### 2b.6 风险

- 这是**预处理域**改动：之前所有基于原图的数字（§5 的示例、阶段 1/2 的 best.pt）**都不能与 DoG 结果直接横比**，只能与"同阶段新跑的对照臂"比较。
- DoG 通道写进 JPEG（q95）有量化损失；若 4-8px 桶反而下降，先把 `--jpeg-quality` 提到 100 或改用 PNG 再判定算子本身无效。
- σ 是按原图像素的目标尺寸（7–9px）定的，**不随 `--imgsz` 变**：同一目标在 960/1280 输入下像素更小，带通仍对应原始 7–9px，这正是我们要保持的语义。

### 阶段 3：pretrained A/B（~3.4 h / 组）

见 §4.2 的 A/B 命令。验收：同划分上 `mAP50-95` 与 `8-16px recall` 至少一项目标超过从零基线 5%；否则说明 COCO 先验对 8px 目标没有帮助，后续实验一律从零或只用 ARD 内预训练。

### 阶段 4：增强消融

```powershell
# mosaic 全开对照
& $PY tools\ard_mav\train.py --variant baseline --imgsz 640 --batch 16 --mosaic 1.0 --name r2_mosaic10
# mixup 小比例对照
& $PY tools\ard_mav\train.py --variant baseline --imgsz 640 --batch 16 --mixup 0.1 --name r2_mixup01
# 更保守的 scale
& $PY tools\ard_mav\train.py --variant baseline --imgsz 640 --batch 16 --scale 0.15 --name r2_scale015
```

判据：与阶段 1 的基线对比，只有 `8-16px recall` 提升 ≥8% 或 `FP/frame` 下降 ≥25% 才算有效；否则保留默认值。

### 阶段 5：test 一次性终评

```powershell
& $PY tools\ard_mav\eval_size_buckets.py --weights ...\weights\best.pt --split test --imgsz <最优> `
  --experiment final_<name> --csv G:\0-rk3588\train\ARD-MAV\eval_round2.csv
```

**test 只在最后一次评测**，不参与调参。如果 test 与 val 的 8-16px recall 差 >0.10，说明 9 序列的 val 方差太大，需要扩充 val 序列（例如 12/12），而不是继续调参。

### 「有效提升」与「该停下」的统一判据

| 判定 | 条件 |
| --- | --- |
| 有效提升 | 8-16px recall 相对 +15%，同时 FP/frame 增幅 ≤30%；或 recall 持平但 FP/frame 降 ≥30% |
| 边际提升 | 8-16px recall +5%~15%，可接受但不值得再投入算力 |
| 无效 | 任一桶 recall 下降 >5%，或 FP/frame 上涨 >50% |
| 该停下重新考虑 | 连续 3 组实验在 8-16px 上无提升；或 640→960→1280 三档收益都 <5%；或 val/test 相差 >0.10 |

---

## 7. 关于 1280 与切片推理

- 1280 相比 640：像素 4×，卷积 MACs 约 4×（P2 层为 stride 4，特征图 320×320，占算力比重高），RK3576 上单帧推理时间预计 ~4×（若 NPU 带宽成为瓶颈，可能 4–5×）。
- 切片（SAHI 式）推理：把 1280 切成 2×2 的 640 tile，算力 ≈ 4×（与整图 1280 同级），但**后处理与 NMS 合并会额外增加 CPU 负载**，且 RK3576 的 CPU 核远弱于 3080；当前 `bench:yolo ... post_us=18328.9` 已显示后处理是瓶颈之一。
- 结论：**先看 960 是否够用**。若 1280 明显有利且必须上板，优先整图 1280（一次推理），切片只作为「远距离小目标召回补救」的离线方案；估算 RK3576 端 1280 单帧 ≥ 4× 现有时延，需要在板上实测后再决策。

---

## 8. 限制

1. 本文档**没有**执行 100/200 epoch 训练；只做了 1 epoch、2% 数据、`--no-val` 的参数连通性 smoke（`runs_smoke\r2_smoke`），因此所有时长、显存都是**基于实测 12 epoch = 0.405 h 的推算**。
2. 数据集盘点使用每 split 300 个标注的随机抽样（VisioDECT 289 个），不是全量统计；`3rd_anti_uav` 的「1,055,401 张 jpg」包含原始未转换目录，实际可用于训练的是 yolo_dataset 的 86,324 + 29,468。
3. 新 val 只有 9 个序列且 ≤16px 占比 76.9%，指标方差偏大；最终判断必须结合 test（44.8%）。
4. 旧 checkpoint 在新 val 上有 7/9 序列属于其训练集，**任何引用旧 best.pt 的评测数字都不可用于决策**。
5. 所有 RK3576 端耗时均为比例估算，没有板上实测；AMP 在本机历史上回退 FP32，实际速度可能比表中更快。
6. 合并外部数据集的建议没有落地脚本：一旦决定合并，需要先按 §3 的比例抽样并重算尺寸桶分布，再改 `splits/train.txt`（val/test 不变）。
