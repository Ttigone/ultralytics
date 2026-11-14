
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


# YOLO26n 预训练权重（P3-P5 部分会按形状匹配加载，P2 分支随机初始化）
WEIGHTS = Path(
    r"G:\0-xiandong\rk3576\ultralytice_self\ultralytics\weights\yolo26n.pt"
)

# P2/4 ~ P5/32 四尺度 + P2/P3 CBAM 双注意力
MODEL_YAML = Path(
    r"G:\0-xiandong\rk3576\ultralytice_self\ultralytics\ultralytics\cfg\models\26\yolo26n-p2-cbam.yaml"
)
RUN_NAME = "yolo26n_p2_cbam_uav_anti_uav_640"

# UAV-Anti-UAV（空对空，单类别 uav）
DATA = Path(r"G:\0-rk3588\train\uav-anti-uav\UAV-Anti-UAV\yolo_dataset\data.yaml")
PROJECT = r"G:\0-rk3588\train\uav-anti-uav\UAV-Anti-UAV\runs"

# 新模型必须 False；中断后续训时改成 True
RESUME = False
RESUME_FROM = Path(PROJECT) / RUN_NAME / "weights" / "last.pt"

# RTX 3080 10G：640 + P2 头 batch=8；OOM 改 4。
# 想提升小目标可试 IMGSZ=768（导出/上板必须一致），延迟约 1.4 倍。
BATCH = 8
EPOCHS = 100
IMGSZ = 640


if __name__ == "__main__":
    multiprocessing.freeze_support()

    assert MODEL_YAML.is_file(), f"找不到模型 YAML：{MODEL_YAML}"
    assert WEIGHTS.is_file(), f"找不到预训练权重：{WEIGHTS}"
    assert DATA.is_file(), f"找不到数据配置：{DATA}"

    # RKNN 友好：SiLU -> LeakyReLU
    Conv.default_act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    device = 0 if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("YOLO26n-P2 + CBAM (P2/P3) Training")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Training device: {device}")
    print(f"Model YAML: {MODEL_YAML}")
    print(f"Dataset: {DATA}")
    print(f"Output: {Path(PROJECT) / RUN_NAME}")
    print(f"Resume: {RESUME}")

    if RESUME:
        assert RESUME_FROM.is_file(), f"找不到断点文件：{RESUME_FROM}"
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
        print("End-to-end:", getattr(head, "end2end", None))
        print("Detect scales:", getattr(head, "nl", None))

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

        # 小目标增广：保持 mosaic/scale，IR 数据需把 HSV 调低
        mosaic=1.0,
        close_mosaic=10,
        scale=0.5,

        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,
        resume=RESUME,
    )
