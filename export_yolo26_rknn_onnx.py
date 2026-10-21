
from pathlib import Path
from collections import Counter
import sys

import onnx
import torch
from ultralytics import YOLO


def main():
    # 第一个参数：.pt 权重
    # 第二个参数：输出 .onnx 路径
    weights = Path(sys.argv[1] if len(sys.argv) > 1 else "yolo26s.pt")
    output = Path(
        sys.argv[2] if len(sys.argv) > 2
        else f"{weights.stem}_raw_rknn.onnx"
    )

    if not weights.is_file():
        raise FileNotFoundError(f"权重文件不存在：{weights}")

    device = torch.device("cpu")
    yolo = YOLO(str(weights))
    model = yolo.model.to(device).float().eval()

    # YOLO26 检测头
    head = model.model[-1]
    required = ("cv2", "cv3", "nl", "nc", "reg_max")
    missing = [name for name in required if not hasattr(head, name)]
    if missing:
        raise RuntimeError(f"最后一层不是预期的 Detect 检测头，缺少属性：{missing}")

    # 只影响本次原始输出导出，不改变训练代码。
    head.export = True
    head.format = "rknn"
    if hasattr(head, "dynamic"):
        head.dynamic = False
    if hasattr(head, "shape"):
        head.shape = None

    imgsz = 640
    dummy = torch.zeros((1, 3, imgsz, imgsz), device=device)

    print(f"Weights : {weights.resolve()}")
    print(f"Output  : {output.resolve()}")
    print(f"Classes : {head.nc}")
    print(f"Scales  : {head.nl}")
    print(f"reg_max : {head.reg_max}")

    # 先在 PyTorch 中检查输出，避免直接导出错误图。
    with torch.inference_mode():
        raw = model(dummy)

    if isinstance(raw, torch.Tensor):
        raw = (raw,)
    elif isinstance(raw, (tuple, list)):
        raw = tuple(raw)
    else:
        raise RuntimeError(f"意外的模型输出类型：{type(raw)}")

    expected_count = head.nl * 2
    if len(raw) != expected_count:
        raise RuntimeError(
            f"期望 {expected_count} 个原始输出，实际得到 {len(raw)} 个。"
            "请检查 Detect.forward() 的 RKNN 分支是否生效。"
        )

    output_names = []
    for i in range(head.nl):
        output_names.extend((f"output{i}_reg", f"output{i}_cls"))

    print("\nPyTorch raw output shapes:")
    for name, tensor in zip(output_names, raw):
        print(f"  {name}: {tuple(tensor.shape)}")

    output.parent.mkdir(parents=True, exist_ok=True)

    # 直接导出已设置为 rknn 模式的模型，
    # 不经过 Ultralytics 默认 end-to-end 输出路径。
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["images"],
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
    )

    exported = onnx.load(str(output))
    onnx.checker.check_model(exported)

    ops = Counter(node.op_type for node in exported.graph.node)
    graph_outputs = [
        (
            value.name,
            [
                dim.dim_value
                for dim in value.type.tensor_type.shape.dim
            ],
        )
        for value in exported.graph.output
    ]

    print("\nONNX graph outputs:")
    for name, shape in graph_outputs:
        print(f"  {name}: {shape}")

    print("\nImportant operator counts:")
    for name in ("TopK", "GatherElements", "Sigmoid", "Softmax"):
        print(f"  {name}: {ops.get(name, 0)}")

    print(f"\nONNX check passed: {output.resolve()}")
    print("注意：Sigmoid 节点仍可能存在于 SiLU 等网络激活中，不能仅凭数量判断分类头是否带 Sigmoid。")


if __name__ == "__main__":
    main()