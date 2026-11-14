# ARD-MAV 极小目标训练步骤（YOLO26n-P2 / +DWR+NWD，100 epoch）

本文件只覆盖两个实验，供手工执行：

1. `baseline-p2`：`yolo26n-p2.yaml`（YOLO26n + P2 检测头，P2/4–P5/32 四尺度，NMS-free）
2. `dwr_nwd-p2`：同一 P2 结构，backbone 浅层插入 DWR 模块，并开启 NWD 回归损失

不含 VisDrone、不含 QAT/RKNN 细节。所有命令为 PowerShell，路径均为绝对路径，可直接复制执行。

---

## 1. 环境

### 1.1 使用的解释器（已实测可用）

```powershell
$PY = "G:\0-xiandong\rk3576\train\torch\torch\python.exe"
```

版本自检（已在本机执行通过）：

```powershell
& "G:\0-xiandong\rk3576\train\torch\torch\python.exe" -c "import sys, torch; print(sys.executable); print(sys.version); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

实测输出：

```text
G:\0-xiandong\rk3576\train\torch\torch\python.exe
3.9.25 (main, Nov  3 2025, 22:44:01) [MSC v.1929 64 bit (AMD64)]
torch 2.1.0+cu121 cuda 12.1 available True
NVIDIA GeForce RTX 3080
```

再确认跑的是本仓库代码而不是 site-packages 里的旧版（本机 `site-packages` 另装有 `ultralytics 8.4.15`，缺少 `nwd` 等新参数，必须避免被它遮蔽）：

```powershell
Set-Location G:\0-xiandong\rk3576\ultralytice_self\ultralytics
& "G:\0-xiandong\rk3576\train\torch\torch\python.exe" -c "import ultralytics; print(ultralytics.__version__, ultralytics.__file__)"
```

期望版本为 `8.4.174`，且路径指向 `G:\0-xiandong\rk3576\ultralytice_self\ultralytics\ultralytics\__init__.py`。

> 关键：每次开新终端后，**先 `Set-Location` 到仓库根目录再运行命令**。在仓库根目录下 `python`/`python -m` 会优先导入本地 `ultralytics/`，否则会导入 8.4.15 旧版并报 `unexpected keyword argument 'nwd'`。

### 1.2 CUDA 不可用时的替代方案

若上面的自检输出 `available False` 或 `NO CUDA`：

1. 先不要用该环境启动训练（CPU 版 torch 会让 100 epoch 变成不可接受的时间）。
2. 在本机其它位置查找可用 CUDA 环境，确认命令（逐个试，找到 `available True` 即可）：

```powershell
conda env list
Get-ChildItem G:\0-xiandong\rk3576\train, G:\rk3576\train, G:\0-xiandong\rk3576\yolo26_train -Recurse -Depth 3 -Filter python.exe -ErrorAction SilentlyContinue | Select-Object FullName
```

（不要对 `G:\` 根目录做全盘递归查找，会很慢。）

3. 都不可用时新建环境（建议 Python 3.10/3.11 + torch 2.x cu121，NVIDIA 驱动已满足 cu121）：

```powershell
python -m venv G:\0-xiandong\rk3576\envs\yolo26cu121
G:\0-xiandong\rk3576\envs\yolo26cu121\Scripts\python.exe -m pip install --upgrade pip
G:\0-xiandong\rk3576\envs\yolo26cu121\Scripts\python.exe -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
G:\0-xiandong\rk3576\envs\yolo26cu121\Scripts\python.exe -m pip install "numpy<2" opencv-python pyyaml pillow psutil pandas matplotlib tqdm requests scipy
```

安装后重复 1.1 的自检，两项都为 True 才继续。

4. 本机已知问题：该 torch 环境的 `_lzma.pyd` 曾因缺少 `liblzma.dll` 导致 `torchvision` 导入失败；如复现，把 `D:\code_tool\anaconda3\Library\bin\liblzma.dll` 复制为 `G:\0-xiandong\rk3576\train\torch\torch\DLLs\liblzma.dll` 即可（该修复已在现有环境中做过）。

---

## 2. 前置检查清单

### 2.1 数据集与 data.yaml

| 项 | 值 |
| --- | --- |
| 数据集根目录 | `G:\0-rk3588\train\ARD-MAV\ARD-MAV-yolo` |
| data.yaml | `G:\0-rk3588\train\ARD-MAV\ARD-MAV-yolo\data.yaml` |
| 类别 | `nc: 1`，`names: ['Drone']` |
| train | `images/train` = 8,478 张 JPEG，`labels/train` = 8,478 个 txt |
| val | `images/val` = 2,283 张 JPEG，`labels/val` = 2,283 个 txt |
| 划分方式 | 序列级划分（48 个序列 train / 12 个序列 val，无同序列泄漏） |
| 抽帧 | `stride=10`（原始标注 107,497 帧 → 10,761 张） |
| 原始标注规模 | 1920x1080，单类 Drone，106,456 个目标；尺寸分桶：<4px 12 / 4-8px 7,919 / 8-16px 49,460 / >16px 49,065 |

data.yaml 内容（当前实际）：

```yaml
path: g:/0-rk3588/train/ARD-MAV/ARD-MAV-yolo
train: images/train
val: images/val
nc: 1
names: ['Drone']
```

图片/标签一一对应检查（数量必须相等）：

```powershell
$D = "G:\0-rk3588\train\ARD-MAV\ARD-MAV-yolo"
Test-Path "$D\data.yaml"
(Get-ChildItem "$D\images\train" -Filter *.jpg).Count
(Get-ChildItem "$D\labels\train" -Filter *.txt).Count
(Get-ChildItem "$D\images\val" -Filter *.jpg).Count
(Get-ChildItem "$D\labels\val" -Filter *.txt).Count
```

逐文件配对检查（输出应为 `0`，即没有缺标签或缺图片的样本）：

```powershell
$D = "G:\0-rk3588\train\ARD-MAV\ARD-MAV-yolo"
foreach ($s in "train","val") {
  $img = Get-ChildItem "$D\images\$s" -Filter *.jpg | ForEach-Object { $_.BaseName }
  $lbl = Get-ChildItem "$D\labels\$s" -Filter *.txt | ForEach-Object { $_.BaseName }
  $missLbl = (Compare-Object $img $lbl | Where-Object SideIndicator -eq "<=").Count   # 有图无标签
  $missImg = (Compare-Object $img $lbl | Where-Object SideIndicator -eq "=>").Count   # 有标签无图
  "$s : missing_labels=$missLbl missing_images=$missImg"
}
```

若 `data.yaml` 被移动到别处，只需同步修改 `path` 一行；训练命令里的 `data=` 指 data.yaml 本身。

### 2.2 模型与权重

| 项 | 路径 |
| --- | --- |
| baseline 模型定义 | `G:\0-xiandong\rk3576\ultralytice_self\ultralytics\ultralytics\cfg\models\26\yolo26n-p2.yaml` |
| DWR 变体模型定义 | `G:\0-xiandong\rk3576\ultralytice_self\ultralytics\ultralytics\cfg\models\26\yolo26n-p2-dwr.yaml` |
| 预训练权重（存在） | `G:\0-xiandong\rk3576\ultralytice_self\ultralytics\weights\yolo26n.pt`（5.5 MB） |

两个实验都使用 `pretrained=False`（从零训练）：

- 官方 `yolo26n.pt` 是 P3–P5 三尺度头，与 P2 四尺度头的检测层形状不一致，加载只会部分匹配，反而让两个实验的初始化不可控；
- 两个实验共享同一初始化策略，对比才有效。

若确实想尝试迁移，可把 `pretrained=False` 改成 `True` 并把 `model` 换成 `G:\0-xiandong\rk3576\ultralytice_self\ultralytics\weights\yolo26n.pt`，但这会改变本文件的对比前提，不作为本文件方案。

### 2.3 磁盘与显存

```powershell
Get-PSDrive G | Select-Object Name, @{n='FreeGB'; e={[math]::Round($_.Free/1GB,1)}}
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
```

- 每个 run 的权重 + 日志约几十 MB，建议 G: 至少留 5 GB 空闲（100 epoch 的 `results.csv`、日志、`last.pt`/`best.pt` 都在这里）。
- 显存：RTX 3080 10 GB。`batch=16`、`imgsz=640`、P2 四尺度实测可跑；如果 `nvidia-smi` 显示已被别的进程占用超过 ~3 GB，先关掉再训练。

---

## 3. 训练命令

以下命令都假设终端当前目录为仓库根目录：

```powershell
Set-Location G:\0-xiandong\rk3576\ultralytice_self\ultralytics
```

### 3.1 baseline-p2（100 epoch）

脚本方式（`tools\ard_mav\train.py`）：

```powershell
Set-Location G:\0-xiandong\rk3576\ultralytice_self\ultralytics
& "G:\0-xiandong\rk3576\train\torch\torch\python.exe" tools\ard_mav\train.py --variant baseline --epochs 100 --imgsz 640 --batch 16 --workers 2 --device 0 --data G:\0-rk3588\train\ARD-MAV\ARD-MAV-yolo\data.yaml --out G:\0-rk3588\train\ARD-MAV\runs --name baseline_p2_e100 --exist-ok
```

等价的 CLI 命令（参数全部显式写出，便于逐项确认）：

```powershell
Set-Location G:\0-xiandong\rk3576\ultralytice_self\ultralytics
& "G:\0-xiandong\rk3576\train\torch\torch\python.exe" -c "from ultralytics.cfg import entrypoint; entrypoint(r'yolo train model=g:/0-xiandong/rk3576/ultralytice_self/ultralytics/ultralytics/cfg/models/26/yolo26n-p2.yaml data=g:/0-rk3588/train/ARD-MAV/ARD-MAV-yolo/data.yaml imgsz=640 epochs=100 batch=16 workers=2 device=0 pretrained=False patience=100 cache=False amp=True max_det=300 close_mosaic=10 project=g:/0-rk3588/train/ARD-MAV/runs name=baseline_p2_e100 exist_ok=True plots=False seed=0')"
```

> 为什么不是 `yolo train ...`：该 torch 环境没有可用的 `yolo` 入口（PATH 中没有 `yolo.exe`），而 `site-packages` 里的 `ultralytics 8.4.15` 既不识别 `nwd` 参数、也没有 DWR 模块。上面的 `entrypoint(r'yolo train ...')` 与真正的 `yolo train ...` 是同一个入口函数，只是把命令串直接传给 `entrypoint(debug=...)`，因此参数语义完全一致；命令里用正斜杠路径，避免 Python 字符串把 `\0`、`\t`、`\u` 当作转义。

### 3.2 dwr_nwd-p2（100 epoch）

脚本方式：

```powershell
Set-Location G:\0-xiandong\rk3576\ultralytice_self\ultralytics
& "G:\0-xiandong\rk3576\train\torch\torch\python.exe" tools\ard_mav\train.py --variant dwr_nwd --epochs 100 --imgsz 640 --batch 16 --workers 2 --device 0 --nwd-threshold 16.0 --nwd-weight 1.0 --nwd-c 12.0 --data G:\0-rk3588\train\ARD-MAV\ARD-MAV-yolo\data.yaml --out G:\0-rk3588\train\ARD-MAV\runs --name dwr_nwd_p2_e100 --exist-ok
```

等价的 CLI 命令（显式写出 NWD 四个参数）：

python ard_mav\train.py --variant baseline --imgsz 640 --batch 16 --workers 2 --epochs 100 --patience 20 --name r2_base_fs --out G:\0-rk3588\train\ARD-MAV\runs

```powershell
Set-Location G:\0-xiandong\rk3576\ultralytice_self\ultralytics
& "G:\0-xiandong\rk3576\train\torch\torch\python.exe" -c "from ultralytics.cfg import entrypoint; entrypoint(r'yolo train model=g:/0-xiandong/rk3576/ultralytice_self/ultralytics/ultralytics/cfg/models/26/yolo26n-p2-dwr.yaml data=g:/0-rk3588/train/ARD-MAV/ARD-MAV-yolo/data.yaml imgsz=640 epochs=100 batch=16 workers=2 device=0 pretrained=False patience=100 cache=False amp=True max_det=300 close_mosaic=10 project=g:/0-rk3588/train/ARD-MAV/runs name=dwr_nwd_p2_e100 exist_ok=True plots=False seed=0 nwd=True nwd_threshold=16.0 nwd_weight=1.0 nwd_c=12.0')"
```

NWD 参数语义（与仓库已实现一致）：

| 参数 | 取值 | 含义 |
| --- | --- | --- |
| `nwd` | `True` | 开启 NWD 回归项；默认 `False` 时行为与原生完全一致 |
| `nwd_threshold` | `16.0` | 以**原图像素**为单位的 GT 最长边阈值，只有 `max(w,h) <= 16px` 的 GT 才叠加 NWD |
| `nwd_c` | `12.0` | NWD 归一化常数，约等于本数据集小目标的平均绝对尺寸（本数据 8–16px 占比最高） |
| `nwd_weight` | `1.0` | NWD 项权重；NWD 是**叠加**在 CIoU/L1 之上，不是替换 |

> NWD 只作用于小框，正常尺寸目标仍走原回归路径；YOLO26 是 `reg_max=1`，其回归项为 L1 而非 DFL。

### 3.3 参数取值理由

| 参数 | 取值 | 理由 |
| --- | --- | --- |
| `model` | `...\yolo26n-p2.yaml` / `...\yolo26n-p2-dwr.yaml` | P2/4 起步，640 输入时 P2 特征图为 160×160，小目标可用 stride 4 采样 |
| `data` | `G:\0-rk3588\train\ARD-MAV\ARD-MAV-yolo\data.yaml` | 序列级划分，train/val 无泄漏 |
| `imgsz` | `640` | 与后续 RK3576 部署输入一致；1080p 原始帧下 8–16px 目标在 640 下约 2.4–4.7px，P2 是唯一能覆盖的分支 |
| `epochs` | `100` | 用户指定；配合 `patience=100` 保证两个实验都完整跑满 100 epoch，利于公平对比 |
| `batch` | `16` | 10 GB 显存实测稳定值；内存紧张时可降到 8（见第 5 节） |
| `workers` | `2` | Windows 上过大 workers 会把可用内存压到 1–2 GB 并导致 worker 崩溃；2 是本机实测稳定值 |
| `device` | `0` | 单卡 RTX 3080 |
| `pretrained` | `False` | P2 四尺度头与官方权重不匹配，见 2.2 |
| `patience` | `100` | 等于 epoch 数，即不早停；两个变体都跑满预算再比 mAP |
| `cache` | `False` | 8,478 张 1080p JPEG 缓存到内存会挤占本已紧张的物理内存 |
| `amp` | `True` | 省显存、提速；若 AMP check 失败会自动回退 FP32（见第 7 节），不需要手动关 |
| `max_det` | `300` | 每图最多 2 个 GT，300 足够；保持默认，便于与后续导出/部署一致 |
| `close_mosaic` | `10` | 最后 10 epoch 关闭 mosaic，让指标收敛到真实分布（第 91 epoch 起会明显跳升） |
| `project` / `name` | `G:\0-rk3588\train\ARD-MAV\runs` / `baseline_p2_e100`、`dwr_nwd_p2_e100` | 两实验独立目录，互不覆盖 |
| `exist_ok` | `True` | 重跑同名实验时直接复用目录，不报错（要保留旧结果就改 `name`） |
| `seed` | `0` | 两个实验同一随机种子，减少数据增强/采样带来的差异 |
| `plots` | `False` | 省 IO 与内存；指标看 `results.csv` 即可 |

---

## 4. 参数对照表

相同项合并说明：`data`、`imgsz=640`、`epochs=100`、`batch=16`、`workers=2`、`device=0`、`pretrained=False`、`patience=100`、`cache=False`、`amp=True`、`max_det=300`、`close_mosaic=10`、`seed=0`、`plots=False`、`project=G:\0-rk3588\train\ARD-MAV\runs`、`exist_ok=True`。

| 项 | baseline-p2 | dwr_nwd-p2 |
| --- | --- | --- |
| 模型 yaml | `G:\0-xiandong\rk3576\ultralytice_self\ultralytics\ultralytics\cfg\models\26\yolo26n-p2.yaml` | `G:\0-xiandong\rk3576\ultralytice_self\ultralytics\ultralytics\cfg\models\26\yolo26n-p2-dwr.yaml` |
| 检测头尺度 | P2/4、P3/8、P4/16、P5/32（4 尺度，`Detect`） | 同上，输出层完全一致 |
| 模型差异 | backbone 第 3 层为 C3k2 输出 | backbone 第 3 层后插入 `DWR`：通道分组 + 膨胀率 1/2/3/4 的 depthwise 3×3 + 1×1 pointwise + 残差 |
| 参数量（实测） | 2,515,976 | 2,520,904（+4,928） |
| NWD 开关 | `nwd=False` | `nwd=True` |
| NWD 参数 | — | `nwd_threshold=16.0`、`nwd_weight=1.0`、`nwd_c=12.0` |
| 损失 | 原生分类/回归（L1 + CIoU） | 同上 + 对小目标叠加 NWD 项 |
| 权重初始化 / 预训练权重 | 从零（`pretrained=False`）；`G:\0-xiandong\rk3576\ultralytice_self\ultralytics\weights\yolo26n.pt` 存在但不加载，两实验一致 | 同左（从零，同一 seed=0，保证可比） |
| 脚本变体名 | `--variant baseline` | `--variant dwr_nwd` |
| 输出目录 | `G:\0-rk3588\train\ARD-MAV\runs\baseline_p2_e100` | `G:\0-rk3588\train\ARD-MAV\runs\dwr_nwd_p2_e100` |

---

## 5. 时长预算与稳定性

### 5.1 实测吞吐推算

本机实测 baseline（12 epoch，batch=16、workers=2、imgsz=640）：

```text
epoch 1  : 132.8 s
epoch 12 : 1457.1 s（累计）→ 平均约 121 s/epoch
```

换算成 100 epoch：

- 单实验约 **3.4 小时**，两个实验合计约 **7 小时**（正常内存条件下）。
- 加 DWR 的参数增量可忽略；NWD 只对小框额外计算，预计每 epoch 增幅 < 5%。
- 波动范围：如果系统可用内存被其它进程挤到 1–2 GB，实测 it/s 会从 ~4.8 掉到 1.3 甚至 0.16–0.5，单 epoch 变成 400–3300 s，100 epoch 可能拉到 **8–12 小时以上**。GPU 利用率高但功耗只有 ~150 W（而非 ~320 W）是这类"假忙"的典型特征。

### 5.2 规避内存耗尽

- 训练前关闭浏览器、其它 Python/训练进程、Docker 等大内存应用。
- `cache=False`（本文件默认），不要改成 `cache=ram`。
- `workers=2` 起步；如果出现 `DataLoader worker (pid ...) exited unexpectedly`，先降到 `workers=0`（单进程加载，最稳、略慢）。
- 显存不足时 `batch=16 → 8`，不要动 `imgsz`（会破坏与部署输入的一致性）。
- 用下面两条命令监控，`Available MBytes` 长期低于 2000 MB 就先腾内存再训练：

```powershell
nvidia-smi
Get-Counter '\Memory\Available MBytes'
```

- 崩溃后不要从头跑，用断点续训（保留 epoch 计数器和优化器状态）：

```powershell
Set-Location G:\0-xiandong\rk3576\ultralytice_self\ultralytics
& "G:\0-xiandong\rk3576\train\torch\torch\python.exe" tools\ard_mav\train.py --resume G:\0-rk3588\train\ARD-MAV\runs\dwr_nwd_p2_e100\weights\last.pt
```

CLI 等价写法：

```powershell
Set-Location G:\0-xiandong\rk3576\ultralytice_self\ultralytics
& "G:\0-xiandong\rk3576\train\torch\torch\python.exe" -c "from ultralytics.cfg import entrypoint; entrypoint(r'yolo train model=g:/0-rk3588/train/ARD-MAV/runs/dwr_nwd_p2_e100/weights/last.pt resume=True')"
```

---

## 6. 训练后

### 6.1 产物位置

| 内容 | 路径 |
| --- | --- |
| 最佳权重 | `G:\0-rk3588\train\ARD-MAV\runs\baseline_p2_e100\weights\best.pt` |
| 最后权重 | `G:\0-rk3588\train\ARD-MAV\runs\baseline_p2_e100\weights\last.pt` |
| 指标曲线数据 | `G:\0-rk3588\train\ARD-MAV\runs\baseline_p2_e100\results.csv` |
| 本次参数快照 | `G:\0-rk3588\train\ARD-MAV\runs\baseline_p2_e100\args.yaml` |
| DWR+NWD 对应文件 | `G:\0-rk3588\train\ARD-MAV\runs\dwr_nwd_p2_e100\weights\best.pt` 等 |

### 6.2 验证命令

```powershell
Set-Location G:\0-xiandong\rk3576\ultralytice_self\ultralytics
& "G:\0-xiandong\rk3576\train\torch\torch\python.exe" -c "from ultralytics.cfg import entrypoint; entrypoint(r'yolo val model=g:/0-rk3588/train/ARD-MAV/runs/baseline_p2_e100/weights/best.pt data=g:/0-rk3588/train/ARD-MAV/ARD-MAV-yolo/data.yaml imgsz=640 batch=8 device=0 conf=0.001 iou=0.5 plots=False workers=0')"
```

把 `model` 换成 `dwr_nwd_p2_e100\weights\best.pt` 即验证第二个实验。

### 6.3 按尺寸分桶召回

mAP 不直接体现极小目标，这里用贪心 IoU 匹配（IoU 0.5）统计各尺寸桶 recall：

```powershell
Set-Location G:\0-xiandong\rk3576\ultralytice_self\ultralytics
& "G:\0-xiandong\rk3576\train\torch\torch\python.exe" tools\ard_mav\eval_size_buckets.py --weights G:\0-rk3588\train\ARD-MAV\runs\baseline_p2_e100\weights\best.pt --data G:\0-rk3588\train\ARD-MAV\ARD-MAV-yolo\data.yaml --imgsz 640 --conf 0.001 --iou 0.5 --batch 8
```

输出形如：

```text
  <4px: recall=0.xxxx  (hits/total)
  4-8px: ...
 8-16px: ...
  >16px: ...
    all: ...
```

解读：

- 桶按 **GT 在原图（1920×1080）中的最长边**划分：`<4px`、`4-8px`、`8-16px`、`>16px`。
- 用 `conf=0.001` 是为了看"召回上限"，不是实际部署阈值；部署阈值另做。
- 12 epoch 的 baseline 实测参考（用于判断 100 epoch 是否异常）：`all=0.5355`、`8-16px=0.4011`、`>16px=0.8160`、`4-8px=0.0123`、`<4px=0/2`。100 epoch 应明显高于这组值，尤其 4–8px 桶。
- 若 `dwr_nwd` 的 `4-8px/8-16px` 桶高于 baseline 而 `>16px` 基本持平，说明 DWR+NWD 生效且没有损害正常尺寸目标；若 `>16px` 明显掉，说明 NWD 阈值/权重过大。

### 6.4 从 results.csv 判断收敛

```powershell
Import-Csv G:\0-rk3588\train\ARD-MAV\runs\baseline_p2_e100\results.csv | Select-Object -Last 15 epoch, @{n='mAP50';e={$_.'metrics/mAP50(B)'}}, @{n='mAP50-95';e={$_.'metrics/mAP50-95(B)'}}, @{n='time';e={$_.time}}
```

- 看 `metrics/mAP50(B)`、`metrics/mAP50-95(B)`。
- 最后 10–20 epoch 的 mAP50-95 增幅 < 0.002 可视为收敛；仍在稳定上升则值得延长训练。
- 注意 `close_mosaic=10`：第 91 epoch 起关闭 mosaic，指标通常会跳变一次，这是正常现象，不要当成异常。
- 学习率按 cosine 退火（`lrf=0.01`），末段 lr 很小，曲线一般比较平。

### 6.5 可选：ONNX 导出

需要中间验证时：

```powershell
Set-Location G:\0-xiandong\rk3576\ultralytice_self\ultralytics
& "G:\0-xiandong\rk3576\train\torch\torch\python.exe" -c "from ultralytics.cfg import entrypoint; entrypoint(r'yolo export model=g:/0-rk3588/train/ARD-MAV/runs/baseline_p2_e100/weights/best.pt format=onnx imgsz=640 opset=17 simplify=True')"
```

QAT / RKNN 不在本文范围，先确认 FP32 ONNX 的输入输出正常再谈量化。

---

## 7. 常见报错处理

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `torch.cuda.OutOfMemoryError` / `CUDA out of memory` | batch 过大或显存被占 | 先 `nvidia-smi` 看占用进程；`batch=16 → 8`；仍不够则关掉其它 GPU 程序。不要改 `imgsz` |
| `RuntimeError: DataLoader worker (pid(s) ...) exited unexpectedly` | Windows 下多进程 + 物理内存不足（本机上一轮即此原因，非代码 bug） | 降 `workers=2 → 0`；关闭占内存应用；从 `weights\last.pt` 续训，不要重跑 |
| 日志出现 `AMP checks failed, disabling AMP` / `AMP check failed` | 本机 AMP 自检不通过（实测发生过），自动回退 FP32 | 属预期行为，精度不受影响，仅略慢；想跳过自检可加 `amp=False` |
| `unexpected keyword argument 'nwd'`，或参数被忽略、跑的是三尺度模型 | 当前目录不在仓库根，导入了 site-packages 的 `ultralytics 8.4.15` | `Set-Location G:\0-xiandong\rk3576\ultralytice_self\ultralytics` 后重跑；用 1.1 的版本检查确认是 `8.4.174` |
| `ImportError: DLL load failed while importing _lzma`（torchvision 导入失败） | torch 环境缺 `liblzma.dll` | 复制 `D:\code_tool\anaconda3\Library\bin\liblzma.dll` 到 `G:\0-xiandong\rk3576\train\torch\torch\DLLs\liblzma.dll` |
| 路径含空格/中文时报找不到文件 | PowerShell 未加引号，或 data.yaml 用了反斜杠转义 | 所有含空格路径用双引号；命令中一律用 `\`（PowerShell 里不需要再转义）；data.yaml 内部用正斜杠 |
| `Dataset images not found` / 验证集为空 | `data.yaml` 的 `path` 写错，或 train/val 相对路径与实际目录不符 | 用 2.1 的命令确认四个目录存在且计数正确；只改 `path` 一行即可 |
| val 阶段很慢或 worker 报错 | `batch=8` 的 val 也走多进程 | 验证时加 `workers=0` |

---

## 8. 附录：改用全帧数据集（stride=1）

当前训练用的是 `stride=10` 子集（10,761 张，train 8,478 / val 2,283）。若要用全部 107,497 个标注帧，重新生成到**独立目录**，不覆盖现有数据：

```powershell
Set-Location G:\0-xiandong\rk3576\ultralytice_self\ultralytics
& "G:\0-xiandong\rk3576\train\torch\torch\python.exe" tools\ard_mav\prepare_dataset.py --root G:\0-rk3588\train\ARD-MAV --out G:\0-rk3588\train\ARD-MAV\ARD-MAV-yolo-full --stride 1 --val-every 5 --jpeg-quality 95
```

该脚本逐视频顺序解码、只写被标注的帧，并按序列划分（`--val-every 5` 表示每 5 个序列取 1 个进 val，规则与当前数据一致，因此可对比）。

影响：

- 图片量约 10 倍，训练 epoch 时间也约 10 倍（约 20 分钟/epoch），100 epoch 约 **1.5–2 天/实验**；两个实验不建议串行跑满。
- 磁盘按 1080p JPEG（q95）每张约 200–300 KB 估算，新增约 **25–35 GB**，先确认 G: 剩余空间再执行。
- 抽帧是逐帧顺序读取视频，10 万帧会跑较久（CPU/磁盘密集），不要和训练同时进行。
- 训练时把 `data=` 改成 `G:\0-rk3588\train\ARD-MAV\ARD-MAV-yolo-full\data.yaml`，其它参数不变。
