"""ARD-MAV sequence reader: video frames + VOC XML ground truth."""

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterator, List, Optional

import cv2
import numpy as np

DEFAULT_ROOT = r"G:\0-rk3588\train\ARD-MAV"

# Background classes picked by visual inspection of mid-sequence frames
# (see docs/mrf_preprocess_report.md).
SEQ_CLASS = {
    "sky": ["phantom43", "phantom77", "phantom84"],
    "urban": ["phantom22", "phantom56", "phantom64"],
    "ground": ["phantom38", "phantom63", "phantom73", "phantom86"],
}


@dataclass
class Sequence:
    root: str
    name: str
    cls: str
    video: str
    ann_dir: str
    frame_count: int

    @property
    def xml_max(self):
        files = os.listdir(self.ann_dir) if os.path.isdir(self.ann_dir) else []
        return len(files)


@dataclass
class Sample:
    seq: str
    cls: str
    frame: int                    # 0-based cv2 frame index
    gray: np.ndarray              # float32 HxW, luma 0..255
    boxes: List[List[float]]      # absolute x1,y1,x2,y2 (inclusive)

    @property
    def sizes(self):
        return [(b[2] - b[0] + 1, b[3] - b[1] + 1) for b in self.boxes]


def make_sequence(root, name, cls="unknown") -> Sequence:
    video = os.path.join(root, "videos", name + ".mp4")
    ann = os.path.join(root, "Annotations", name)
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return Sequence(root, name, cls, video, ann, n)


def read_boxes(ann_dir, name, frame1) -> Optional[List[List[float]]]:
    """frame1 is the 1-based index used in the XML filenames."""
    f = os.path.join(ann_dir, "%s_%04d.xml" % (name, frame1))
    if not os.path.exists(f):
        return None
    try:
        root = ET.parse(f).getroot()
    except ET.ParseError:
        return None
    boxes = []
    for o in root.findall("object"):
        b = o.find("bndbox")
        if b is None:
            continue
        x1 = float(b.find("xmin").text)
        y1 = float(b.find("ymin").text)
        x2 = float(b.find("xmax").text)
        y2 = float(b.find("ymax").text)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        boxes.append([x1, y1, x2, y2])
    return boxes or None


def iter_samples(seq: Sequence, n_frames=200, max_dim=None, min_dim=2,
                 start_frac=0.0, end_frac=1.0, gray_mode=True) -> Iterator[Sample]:
    """Uniformly sample ``n_frames`` frames and yield those with valid GT."""
    lo = int(seq.frame_count * start_frac)
    hi = max(lo + 1, int(seq.frame_count * end_frac) - 1)
    hi = min(hi, seq.xml_max)                      # last usable annotated frame
    if hi <= lo:
        return
    idxs = np.linspace(lo, hi, num=min(int(n_frames), hi - lo + 1))
    idxs = sorted(set(int(round(i)) for i in idxs))

    cap = cv2.VideoCapture(seq.video)
    if not cap.isOpened():
        raise RuntimeError("cannot open %s" % seq.video)
    cur = -1
    for idx in idxs:
        while cur < idx - 1:
            if not cap.grab():
                break
            cur += 1
        ok, frame = cap.read()
        cur += 1
        if not ok:
            break
        boxes = read_boxes(seq.ann_dir, seq.name, idx + 1)
        if not boxes:
            continue
        if max_dim is not None:
            if max(max(b[2] - b[0], b[3] - b[1]) + 1 for b in boxes) > max_dim:
                continue
        if min(max(b[2] - b[0], b[3] - b[1]) + 1 for b in boxes) < min_dim:
            continue
        if gray_mode:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            g = frame
        yield Sample(seq.name, seq.cls, idx, g, boxes)
    cap.release()


def scan(root=DEFAULT_ROOT):
    """All sequences with their (wide) target-size statistics."""
    out = []
    for f in sorted(os.listdir(os.path.join(root, "videos"))):
        if not f.endswith(".mp4"):
            continue
        name = f[:-4]
        seq = make_sequence(root, name)
        sizes = []
        for i in range(1, seq.xml_max + 1, 10):   # every 10th frame is enough
            b = read_boxes(seq.ann_dir, name, i)
            if b:
                sizes += [(x[2] - x[0] + 1, x[3] - x[1] + 1) for x in b]
        if not sizes:
            continue
        mw = np.median([s[0] for s in sizes])
        mh = np.median([s[1] for s in sizes])
        out.append(dict(name=name, frames=seq.frame_count, boxes=len(sizes),
                        med_w=float(mw), med_h=float(mh),
                        tiny_frac=float(np.mean([max(s) <= 12 for s in sizes])))
        )
    return out
