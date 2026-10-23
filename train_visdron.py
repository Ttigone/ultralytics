import os
import multiprocessing
from pathlib import Path

_LZMA_DLL_HANDLE = os.add_dll_directory(
    r"C:\Users\Ttigone\.conda\pkgs\xz-5.8.2-h53af0af_0\Library\bin"
)

import torch
from ultralytics import YOLO
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics import YOLO

# 增加 p2, 移除 p5
# WEIGHTS = Path(r"G:\0-xiandong\rk3576\ultralytice_self\ultralytics\weights\yolo26n.pt")
# MODEL_YAML = Path(r"G:\0-xiandong\rk3576\ultralytice_self\ultralytics\ultralytics\cfg\models\26\yolo26n-p2_p4.yaml")
# DATA = Path(r"G:\0-xiandong\rk3576\need_train\VisDrone.yaml")
# PROJECT = r"G:\0-xiandong\rk3576\need_train\runs"
# RUN_NAME = "yolo26n_p2_p4_visdrone_640_md1000"

# P2_P5
# WEIGHTS = Path(r"G:\0-xiandong\rk3576\ultralytice_self\ultralytics\weights\yolo26n.pt")
# MODEL_YAML = Path(r"G:\0-xiandong\rk3576\ultralytice_self\ultralytics\ultralytics\cfg\models\26\yolo26n-p2.yaml")
WEIGHTS = Path(r"G:\0-xiandong\rk3576\ultralytice_self\ultralytics\yolo26s.pt")
MODEL_YAML = Path(r"G:\0-xiandong\rk3576\ultralytice_self\ultralytics\ultralytics\cfg\models\26\yolo26s-p2.yaml")
DATA = Path(r"G:\0-xiandong\rk3576\need_train\VisDrone.yaml")
PROJECT = r"G:\0-xiandong\rk3576\need_train\runs"
RUN_NAME = "yolo26s_p2_p5_visdrone_640_md1000"

RESUME = False
RESUME_FROM = Path(PROJECT) / RUN_NAME / "weights" / "last.pt"

if __name__ == '__main__':
    multiprocessing.freeze_support()

    assert MODEL_YAML.is_file(), MODEL_YAML
    assert WEIGHTS.is_file(), WEIGHTS
    assert DATA.is_file(), DATA

    # 只替换常规 SiLU 激活，不要替换 Sigmoid 等输出运算
    Conv.default_act = nn.LeakyReLU(
        negative_slope=0.1, inplace=True
    )

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")
    print(f"Model YAML: {MODEL_YAML}")

    if RESUME:
        assert RESUME_FROM.is_file(), RESUME_FROM
        print(f"Resuming from checkpoint: {RESUME_FROM}")
        model = YOLO(str(RESUME_FROM))
    else:
        model = YOLO(str(MODEL_YAML), task="detect")

        # 先统计构建后模型中的激活函数
        before = {}
        for module in model.model.modules():
            name = type(module).__name__
            if isinstance(module, (nn.SiLU, nn.LeakyReLU)):
                before[name] = before.get(name, 0) + 1
        print("Activation modules after construction:", before)

        # 加载原始 yolo26n 预训练权重
        model.load(str(WEIGHTS))

        head = model.model.model[-1]
        print("Head:", type(head).__name__)
        print("Classes:", getattr(head, "nc", None))
        print("End-to-end:", getattr(head, "end2end", None))

        for name in ("one2one_cv2", "one2one_cv3"):
            print(name, hasattr(head, name))

        # 输出所有常见激活的统计
        from collections import Counter
        print("Activation summary:", Counter(
            type(m).__name__ for m in model.model.modules()
            if isinstance(m, (nn.SiLU, nn.LeakyReLU, nn.ReLU))
        ))

    model.train(
        data=str(DATA),
        epochs=100,
        imgsz=640,
        batch=4,
        workers=0,          
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