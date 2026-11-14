
import os
import multiprocessing
from pathlib import Path
from collections import Counter

_LZMA_DLL_HANDLE = os.add_dll_directory(
    r"C:\Users\Ttigone\.conda\pkgs\xz-5.8.2-h53af0af_0\Library\bin"
)

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv


# YOLO26n 预训练权重
WEIGHTS = Path(
    r"G:\0-xiandong\rk3576\ultralytice_self\ultralytics\weights\yolo26n.pt"
)

# P2/P3/P4/P5 四尺度（小目标更敏感，板端后处理已支持 8 输出 p2..p5）
MODEL_YAML = Path(
    r"G:\0-xiandong\rk3576\ultralytice_self\ultralytics\ultralytics\cfg\models\26\yolo26n-p2.yaml"
)
RUN_NAME = "yolo26n_p2_uav_anti_uav_640"

# UAV-Anti-UAV（空对空，单类别 uav）
DATA = Path(
    r"G:\0-rk3588\train\uav-anti-uav\UAV-Anti-UAV\yolo_dataset\data.yaml"
)

# 单独保存 UAV-Anti-UAV 的训练结果
PROJECT = r"G:\0-rk3588\train\uav-anti-uav\UAV-Anti-UAV\runs"

# 首次训练必须为 False；恢复本次训练时才改成 True
RESUME = True
RESUME_FROM = Path(PROJECT) / RUN_NAME / "weights" / "last.pt"

# RTX 3080 10G：640 分辨率 + P2 头，batch=8 一般可跑；OOM 时改 4
BATCH = 8
EPOCHS = 100
IMGSZ = 640


if __name__ == "__main__":
    multiprocessing.freeze_support()

    assert MODEL_YAML.is_file(), f"找不到模型 YAML：{MODEL_YAML}"
    assert WEIGHTS.is_file(), f"找不到预训练权重：{WEIGHTS}"
    assert DATA.is_file(), f"找不到数据配置：{DATA}"

    # RKNN 友好：SiLU -> LeakyReLU（与 DUT 训练脚本保持一致）
    Conv.default_act = nn.LeakyReLU(
        negative_slope=0.1,
        inplace=True,
    )

    device = 0 if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("YOLO26n UAV-Anti-UAV Training")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Training device: {device}")
    print(f"Model YAML: {MODEL_YAML}")
    print(f"Pretrained weights: {WEIGHTS}")
    print(f"Dataset: {DATA}")
    print(f"Output: {Path(PROJECT) / RUN_NAME}")
    print(f"Resume: {RESUME}")

    if RESUME:
        assert RESUME_FROM.is_file(), f"找不到断点文件：{RESUME_FROM}"
        print(f"Resuming from: {RESUME_FROM}")
        model = YOLO(str(RESUME_FROM))
    else:
        model = YOLO(str(MODEL_YAML), task="detect")
        print(
            "Activation modules after construction:",
            Counter(
                type(m).__name__
                for m in model.model.modules()
                if isinstance(m, (nn.SiLU, nn.LeakyReLU, nn.ReLU))
            ),
        )
        model.load(str(WEIGHTS))
        head = model.model.model[-1]
        print("Head:", type(head).__name__)
        print("Classes before dataset setup:", getattr(head, "nc", None))
        print("End-to-end:", getattr(head, "end2end", None))

    model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=2,
        cache=False,
        device=device,

        patience=30,
        max_det=1000,
        amp=True,

        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,
        resume=RESUME,
    )
