# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build a sequence-level train/val/test split for the ARD-MAV dataset.

Splitting by sequence (not by frame) keeps adjacent frames of the same video out of different splits, which is the
leakage the first training round needed to avoid. Images are referenced through Ultralytics `.txt` image lists, so
nothing is copied or moved on disk.

Usage:
    python tools/ard_mav/make_splits.py --root g:/0-rk3588/train/ARD-MAV/ARD-MAV-yolo --seed 2026
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml

BUCKETS = ((0, 4, "<4px"), (4, 8, "4-8px"), (8, 16, "8-16px"), (16, 32, "16-32px"), (32, 1e9, ">32px"))


def bucket_of(size: float) -> str:
    """Return the size-bucket label for a box's longest side in original-image pixels."""
    return next(name for lo, hi, name in BUCKETS if lo <= size < hi)


def natural_key(stem: str) -> tuple:
    """Sort ``phantom12_0031`` by (sequence, numeric frame index)."""
    seq, _, idx = stem.rpartition("_")
    return (seq, int(idx) if idx.isdigit() else 0)


def collect(root: Path) -> dict[str, dict]:
    """Aggregate frame and box statistics per sequence from the existing YOLO label files."""
    stats = json.loads((root / "stats.json").read_text(encoding="utf-8"))
    resolutions = stats.get("resolutions", {})
    if len(resolutions) != 1:
        raise SystemExit(f"Expected one image resolution, got {sorted(resolutions)}; per-image sizes are required")
    width, height = (int(v) for v in next(iter(resolutions)).split("x"))

    sequences: dict[str, dict] = {}
    for split in ("train", "val"):
        for image in sorted((root / "images" / split).glob("*.jpg"), key=lambda p: natural_key(p.stem)):
            seq = image.stem.rpartition("_")[0]
            rec = sequences.setdefault(seq, {"frames": 0, "boxes": 0, "buckets": defaultdict(int), "images": []})
            rec["frames"] += 1
            rec["images"].append(image)
            label = root / "labels" / split / f"{image.stem}.txt"
            if not label.is_file():
                rec["missing_labels"] = rec.get("missing_labels", 0) + 1
                continue
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                _, _, _, bw, bh = (float(v) for v in parts)
                rec["boxes"] += 1
                rec["buckets"][bucket_of(max(bw * width, bh * height))] += 1
    return sequences


def assign(sequences: dict[str, dict], shares: dict[str, float], seed: int) -> dict[str, list[str]]:
    """Greedy split assignment: tiny-target-heavy sequences are spread across splits by frame-share deficit.

    Ace: with 60 sequences this yields 42/9/9 and keeps the <=16px box ratio similar in all three splits, without
    swapping scene diversity (each sequence is a different flight recording).
    """
    total_frames = sum(s["frames"] for s in sequences.values())
    caps = {name: max(1, round(len(sequences) * share)) for name, share in shares.items()}
    # rounding may leave one sequence unassigned (e.g. 60 * 0.15 = 9.0 is exact; 43 * 0.15 = 6.45 -> 6+6)
    while sum(caps.values()) < len(sequences):
        name = max(shares, key=lambda n: len(sequences) * shares[n] - caps[n])
        caps[name] += 1
    while sum(caps.values()) > len(sequences):
        name = max(caps, key=lambda n: caps[n])
        caps[name] -= 1

    order = sorted(sequences, key=lambda s: (-sequences[s]["buckets"].get("4-8px", 0) / max(sequences[s]["boxes"], 1), s))
    rng = random.Random(seed)
    # shuffle inside equal-key runs so the seed still matters without disturbing tiny-target prioritisation
    for i in range(len(order)):
        j = i + 1
        while j < len(order) and sequences[order[j]]["buckets"].get("4-8px", 0) == sequences[order[i]]["buckets"].get("4-8px", 0):
            j += 1
        if j - i > 1:
            rng.shuffle(order[i:j])
        i = j

    frames = dict.fromkeys(shares, 0)
    counts = dict.fromkeys(shares, 0)
    out = {name: [] for name in shares}
    target_frames = {name: share * total_frames for name, share in shares.items()}
    for seq in order:
        candidates = [name for name in shares if counts[name] < caps[name]]
        pick = max(candidates, key=lambda name: (target_frames[name] - frames[name]) / max(caps[name] - counts[name], 1))
        out[pick].append(seq)
        counts[pick] += 1
        frames[pick] += sequences[seq]["frames"]
    return out


def rebalance(
    split_names: dict[str, list[str]], sequences: dict[str, dict], shares: dict[str, float], max_passes: int = 500
) -> dict[str, list[str]]:
    """Swap single sequences between splits until each split's size-bucket mix tracks the global mix.

    The first greedy pass balances frame counts only, which can leave val/test dominated by one bucket (a tiny-only
    val split would make recall measurements meaningless). Swapping one val/test sequence with one train sequence
    keeps the 42/9/9 counts and keeps every frame share inside +/-2.5 percentage points of the requested share.

    ``max_passes`` must be large: with 60 heterogeneous sequences and only 9 val / 9 test slots, a single improving
    swap per pass converges slowly, and the default of 6 left val at 88.6% tiny targets against a 56.7% global mix.
    """
    bucket_names = [b[2] for b in BUCKETS]
    total_frames = sum(s["frames"] for s in sequences.values())
    total_boxes = sum(s["boxes"] for s in sequences.values())
    global_dist = {
        b: sum(s["buckets"].get(b, 0) for s in sequences.values()) / max(total_boxes, 1) for b in bucket_names
    }
    target_frames = {name: share * total_frames for name, share in shares.items()}
    tolerance = 0.025 * total_frames

    def measure(state: dict[str, list[str]]) -> tuple[float, dict[str, int]]:
        frames = {name: sum(sequences[s]["frames"] for s in names) for name, names in state.items()}
        boxes = {name: sum(sequences[s]["boxes"] for s in names) for name, names in state.items()}
        dist = {
            name: {b: sum(sequences[s]["buckets"].get(b, 0) for s in names) / max(boxes[name], 1) for b in bucket_names}
            for name, names in state.items()
        }
        return sum(abs(dist[name][b] - global_dist[b]) for name in state for b in bucket_names), frames

    best_score, _ = measure(split_names)
    for _ in range(max_passes):
        improved = False
        for small, big in (("val", "train"), ("test", "train")):
            for i, small_seq in enumerate(split_names[small]):
                for j, big_seq in enumerate(split_names[big]):
                    candidate = {name: list(names) for name, names in split_names.items()}
                    candidate[small][i] = big_seq
                    candidate[big][j] = small_seq
                    score, frames = measure(candidate)
                    if all(abs(frames[name] - target_frames[name]) <= tolerance for name in shares) and (
                        score < best_score - 1e-6
                    ):
                        split_names, best_score = candidate, score
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return split_names


def summarize(sequences: dict[str, dict], names: list[str]) -> dict:
    """Aggregate frames, boxes and buckets for one split."""
    rec = {"sequences": names, "frames": 0, "boxes": 0, "buckets": dict.fromkeys([b[2] for b in BUCKETS], 0)}
    for name in names:
        rec["frames"] += sequences[name]["frames"]
        rec["boxes"] += sequences[name]["boxes"]
        for bucket, count in sequences[name]["buckets"].items():
            rec["buckets"][bucket] += count
    rec["tiny_share"] = (rec["buckets"]["<4px"] + rec["buckets"]["4-8px"] + rec["buckets"]["8-16px"]) / max(rec["boxes"], 1)
    return rec


def main() -> None:
    """Write image lists, a data YAML and a manifest for the new split."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("g:/0-rk3588/train/ARD-MAV/ARD-MAV-yolo"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    args = parser.parse_args()

    root = args.root.resolve()
    out = (args.out or root / "splits").resolve()
    shares = {"train": args.train, "val": args.val, "test": args.test}
    sequences = collect(root)
    if any(v.get("missing_labels") for v in sequences.values()):
        raise SystemExit("Some images have no label file; fix the dataset before splitting")

    split_names = rebalance(assign(sequences, shares, args.seed), sequences, shares)
    all_names = [n for names in split_names.values() for n in names]
    if len(all_names) != len(set(all_names)) or len(all_names) != len(sequences):
        raise SystemExit("Split assignment lost or duplicated sequences")

    out.mkdir(parents=True, exist_ok=True)
    for split, names in split_names.items():
        lines = []
        for seq in sorted(names):
            for image in sequences[seq]["images"]:
                lines.append(image.as_posix())
        (out / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {split: summarize(sequences, names) for split, names in split_names.items()}
    total_frames = sum(sequences[s]["frames"] for s in sequences)
    (out / "data.yaml").write_text(
        "\n".join(
            [
                "# ARD-MAV sequence-level split (70/15/15), generated by tools/ard_mav/make_splits.py",
                f"path: {root.as_posix()}",
                "train: splits/train.txt",
                "val: splits/val.txt",
                "test: splits/test.txt",
                "nc: 1",
                "names: ['Drone']",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = {
        "root": root.as_posix(),
        "seed": args.seed,
        "shares_requested": shares,
        "images": sum(s["frames"] for s in sequences.values()),
        "sequences_total": len(sequences),
        "splits": {
            split: {
                **summary[split],
                "sequence_count": len(split_names[split]),
                "frame_share": round(summary[split]["frames"] / total_frames, 4),
            }
            for split in split_names
        },
        "per_sequence": {
            seq: {
                "split": next(s for s, names in split_names.items() if seq in names),
                "frames": sequences[seq]["frames"],
                "boxes": sequences[seq]["boxes"],
                "buckets": dict(sequences[seq]["buckets"]),
            }
            for seq in sorted(sequences)
        },
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"root={root} out={out} images={manifest['images']} sequences={len(sequences)}")
    for split, rec in manifest["splits"].items():
        print(
            f"  {split:>5}: sequences={rec['sequence_count']:2d} frames={rec['frames']:5d} "
            f"({rec['frame_share']:.1%}) boxes={rec['boxes']:5d} tiny<=16px={rec['tiny_share']:.1%} "
            f"buckets={rec['buckets']}"
        )


if __name__ == "__main__":
    main()
