import os

_LZMA_DLL_HANDLE = os.add_dll_directory(
    r"C:\Users\Ttigone\.conda\pkgs\xz-5.8.2-h53af0af_0\Library\bin"
)


from pathlib import Path

import torch
from ultralytics import YOLO

# WEIGHTS = Path(
#     r"G:\0-xiandong\rk3576\ultralytice_self\ultralytics\yolo26s.pt"
# )
WEIGHTS = Path(
    r"G:\0-xiandong\rk3576\ultralytice_self\ultralytics\weights\yolo26n.pt"
)

DATA = Path(
    r"G:\0-xiandong\rk3576\need_train\VisDrone.yaml"
)
PROJECT = r"G:\0-xiandong\rk3576\need_train\runs"

if not WEIGHTS.is_file():
    raise FileNotFoundError(f"找不到预训练权重：{WEIGHTS}")

if not DATA.is_file():
    raise FileNotFoundError(f"找不到数据集配置：{DATA}")

device = 0 if torch.cuda.is_available() else "cpu"
print(f"Training device: {device}")

model = YOLO(str(WEIGHTS))

# 修改 Detect 检测头的端到端输出上限
detect_head = model.model.model[-1]
print("Head:", type(detect_head).__name__)
print("Classes:", detect_head.nc)
print("Old max_det:", detect_head.max_det)

detect_head.max_det = 1000

print("New max_det:", detect_head.max_det)

model.train(
    data=str(DATA),
    epochs=100,
    imgsz=640,
    batch=8,
    workers=0,
    device=device,
    pretrained=True,
    patience=30,
    max_det=1000,
    amp=False,
    project=PROJECT,
    # name="yolo26s_visdrone_640_md1000",
    name="yolo26n_visdrone_640_md1000",
    exist_ok=True,
)