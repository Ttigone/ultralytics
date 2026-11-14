
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


# # YOLO26n 预训练权重
WEIGHTS = Path(
    r"G:\0-xiandong\rk3576\ultralytice_self\ultralytics\weights\yolo26n.pt"
)

# # P2/P3/P4/P5 模型结构
MODEL_YAML = Path(
    r"G:\0-xiandong\rk3576\ultralytice_self\ultralytics\ultralytics\cfg\models\26\yolo26n-p2.yaml"
)
RUN_NAME = "yolo26n_p2_p5_dut_640"


# WEIGHTS = Path(
#     r"G:\0-xiandong\rk3576\ultralytice_self\ultralytics\yolo26s.pt"
# )
# # YOLO26s 的 P2-P5 模型配置
# MODEL_YAML = Path(
#     r"G:\0-xiandong\rk3576\ultralytice_self\ultralytics\ultralytics\cfg\models\26\yolo26s-p2.yaml"
# )
# RUN_NAME = "yolo26s_p2_p5_dut_640"


# DUT-Anti-UAV 数据集
DATA = Path(
    r"G:\0-rk3588\train\DUT-Anti-UAV\yolo_dataset\data.yaml"
)

# 单独保存 DUT 的训练结果，不覆盖 VisDrone
PROJECT = r"G:\0-rk3588\train\DUT-Anti-UAV\runs"


# 首次训练必须为 False。
# 后续只有想恢复本次 DUT 训练时，才改成 True。
RESUME = False

RESUME_FROM = (
    Path(PROJECT) / RUN_NAME / "weights" / "last.pt"
)

if __name__ == "__main__":
    multiprocessing.freeze_support()

    assert MODEL_YAML.is_file(), (
        f"找不到模型 YAML：{MODEL_YAML}"
    )
    assert WEIGHTS.is_file(), (
        f"找不到预训练权重：{WEIGHTS}"
    )
    assert DATA.is_file(), (
        f"找不到数据配置：{DATA}"
    )

    # 不再全局替换 Conv.default_act: LeakyReLU 会让网络结构与 COCO 预训练权重(yolo26n.pt)不匹配,
    # 只能转移部分层; 统一用默认 SiLU 后结构与预训练严格一致, 导出链路也走标准路径.
    device = 0 if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("YOLO26s DUT-Anti-UAV Training")
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
        assert RESUME_FROM.is_file(), (
            f"找不到 DUT 断点文件：{RESUME_FROM}"
        )

        print(f"Resuming from: {RESUME_FROM}")

        # 恢复本次 DUT 的训练状态
        model = YOLO(str(RESUME_FROM))

    else:
        # 从 YAML 构建 P2/P3/P4/P5 模型
        model = YOLO(str(MODEL_YAML), task="detect")

        # 统计模型构建后的激活函数
        print(
            "Activation modules after construction:",
            Counter(
                type(m).__name__
                for m in model.model.modules()
                if isinstance(
                    m,
                    (nn.SiLU, nn.LeakyReLU, nn.ReLU)
                )
            ),
        )

        # 加载 YOLO26 预训练权重
        model.load(str(WEIGHTS))

        head = model.model.model[-1]

        print("Head:", type(head).__name__)
        print("Classes before dataset setup:",
              getattr(head, "nc", None))
        print("End-to-end:", getattr(head, "end2end", None))

        for name in ("one2one_cv2", "one2one_cv3"):
            print(name, hasattr(head, name))

        print(
            "Activation summary:",
            Counter(
                type(m).__name__
                for m in model.model.modules()
                if isinstance(
                    m,
                    (nn.SiLU, nn.LeakyReLU, nn.ReLU)
                )
            ),
        )

    model.train(
        data=str(DATA),

        # 基本训练参数
        epochs=300,
        imgsz=640,
        batch=4,
        workers=2,
        cache=False,
        device=device,

        # 验证与目标数量
        patience=30,
        max_det=1000,

        # 混合精度
        amp=True,

        # 单独保存 DUT 实验
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,

        # 只有本次 DUT 已经训练过且需要恢复时才开启
        resume=RESUME,
    )
