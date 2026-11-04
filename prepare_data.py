"""Convert the raw ShapeNet-Part "PartAnnotation" folders into one cached .npz file.

Expected layout (from the README you uploaded):
    <root>/<synset>/points/<model_id>.pts
    <root>/<synset>/expert_verified/points_label/<model_id>.seg   (folder name may be point_labels)

Usage:
    python prepare_data.py --root /path/to/PartAnnotation --num_points 2048 --out cache_2048.npz
    python prepare_data.py --root /path/to/PartAnnotation --max_per_cat 200 --out cache_small.npz   # quick subset

Only shapes that have expert-verified, fully labelled points are used.
Anything that looks wrong (point/label count mismatch, label out of range) is skipped and reported.
"""
import argparse
import glob
import os

import numpy as np

# (category, synset id, number of parts). Global part ids follow this order: 0..49.
CATEGORIES = [
    ("Airplane", "02691156", 4), ("Bag", "02773838", 2), ("Cap", "02954340", 2),
    ("Car", "02958343", 4), ("Chair", "03001627", 4), ("Earphone", "03261776", 3),
    ("Guitar", "03467517", 3), ("Knife", "03624134", 2), ("Lamp", "03636649", 4),
    ("Laptop", "03642806", 2), ("Motorbike", "03790512", 6), ("Mug", "03797390", 2),
    ("Pistol", "03948459", 3), ("Rocket", "04099429", 3), ("Skateboard", "04225987", 3),
    ("Table", "04379243", 3),
]


def find_label_dir(synset_dir):
    """Return the folder with per-point labels for all parts, or None."""
    for d in sorted(glob.glob(os.path.join(synset_dir, "expert_verified", "*"))):
        if os.path.isdir(d) and "label" in os.path.basename(d).lower():
            return d
    return None


def normalize(pts):
    pts = pts - pts.mean(0, keepdims=True)
    scale = np.max(np.linalg.norm(pts, axis=1))
    return pts / (scale if scale > 0 else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="PartAnnotation folder (contains the synset folders)")
    ap.add_argument("--num_points", type=int, default=2048)
    ap.add_argument("--max_per_cat", type=int, default=0, help="0 = use all shapes")
    ap.add_argument("--out", default="cache_2048.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    P, Y, C = [], [], []
    offset = 0
    print(f"{'category':<11}{'used':>6}{'skipped':>9}")
    for ci, (name, synset, k) in enumerate(CATEGORIES):
        sdir = os.path.join(args.root, synset)
        label_dir = find_label_dir(sdir)
        used = skipped = 0
        if label_dir is None or not os.path.isdir(os.path.join(sdir, "points")):
            print(f"{name:<11}{'-':>6}{'-':>9}   (folder or labels not found under {sdir})")
            offset += k
            continue
        label_files = sorted(glob.glob(os.path.join(label_dir, "*")))
        if args.max_per_cat:
            rng.shuffle(label_files)
        for lf in label_files:
            if args.max_per_cat and used >= args.max_per_cat:
                break
            base = os.path.splitext(os.path.basename(lf))[0]
            pf = os.path.join(sdir, "points", base + ".pts")
            if not os.path.exists(pf):
                skipped += 1
                continue
            pts = np.loadtxt(pf, dtype=np.float32).reshape(-1, 3)
            lab = np.loadtxt(lf, dtype=np.int64).reshape(-1) - 1  # files are 1-based per category
            if len(pts) != len(lab) or lab.min() < 0 or lab.max() >= k:
                skipped += 1
                continue
            n = len(pts)
            sel = rng.choice(n, args.num_points, replace=n < args.num_points)
            P.append(normalize(pts[sel]))
            Y.append((lab[sel] + offset).astype(np.int8))
            C.append(ci)
            used += 1
        print(f"{name:<11}{used:>6}{skipped:>9}")
        offset += k

    if not P:
        raise SystemExit("No shapes found. Check --root and the folder layout (see top of this file).")
    np.savez(args.out, points=np.stack(P).astype(np.float32), labels=np.stack(Y), cats=np.array(C, dtype=np.int64))
    print(f"Saved {len(P)} shapes x {args.num_points} points to {args.out}")


if __name__ == "__main__":
    main()
