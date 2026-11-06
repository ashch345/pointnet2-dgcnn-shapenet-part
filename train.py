"""Train and evaluate PointNet++ or DGCNN on the cached ShapeNet-Part data.

Examples:
    python train.py --model pointnet2 --cache cache_small.npz --epochs 30 --batch_size 16 --out runs/pn2
    python train.py --model dgcnn     --cache cache_small.npz --epochs 30 --batch_size 8  --out runs/dgcnn

Use the SAME cache, split (--seed) and epochs for both models so the comparison is fair.
Note: the split is a random 80/10/10 per category made here, NOT the official benchmark split,
so numbers are not directly comparable to published results.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from metrics import shape_iou, summarize
from models import DGCNNSeg, PointNet2Seg
from prepare_data import CATEGORIES

CAT_NAMES = [c[0] for c in CATEGORIES]
NPARTS = [c[2] for c in CATEGORIES]
OFFSETS = np.concatenate([[0], np.cumsum(NPARTS)[:-1]]).astype(int)
NUM_PARTS, NUM_CATS = int(sum(NPARTS)), len(CATEGORIES)


def split_indices(cats, seed, frac=(0.8, 0.1)):
    rng = np.random.RandomState(seed)
    tr, va, te = [], [], []
    for c in np.unique(cats):
        idx = np.where(cats == c)[0]
        rng.shuffle(idx)
        a, b = int(len(idx) * frac[0]), int(len(idx) * (frac[0] + frac[1]))
        tr += list(idx[:a]); va += list(idx[a:b]); te += list(idx[b:])
    return np.array(tr), np.array(va), np.array(te)


def augment(x):
    """Random anisotropic scale and shift, per shape."""
    B = x.shape[0]
    scale = torch.empty(B, 1, 3, device=x.device).uniform_(0.8, 1.25)
    shift = torch.empty(B, 1, 3, device=x.device).uniform_(-0.1, 0.1)
    return x * scale + shift


@torch.no_grad()
def evaluate(model, P, Y, C, idx, device, bs):
    model.eval()
    ious, cats = [], []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        x = torch.from_numpy(P[b]).to(device)
        c = torch.from_numpy(C[b]).long().to(device)
        logits = model(x, F.one_hot(c, NUM_CATS).float()).cpu().numpy()
        for j, s in enumerate(b):
            lo, k = OFFSETS[C[s]], NPARTS[C[s]]
            pred = logits[j][:, lo:lo + k].argmax(-1) + lo  # only choose among the category's own parts
            ious.append(shape_iou(pred, Y[s].astype(np.int64), lo, k)[0])
            cats.append(C[s])
    return summarize(ious, cats, CAT_NAMES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_2048.npz")
    ap.add_argument("--model", choices=["pointnet2", "dgcnn"], required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_points", type=int, default=2048)
    ap.add_argument("--k", type=int, default=20, help="DGCNN neighbours")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt if it exists")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    d = np.load(args.cache)
    P, Y, C = d["points"][:, :args.num_points], d["labels"][:, :args.num_points], d["cats"]
    tr, va, te = split_indices(C, args.seed)
    print(f"device={device}  shapes: train {len(tr)}  val {len(va)}  test {len(te)}  points/shape {P.shape[1]}")

    model = (PointNet2Seg(NUM_PARTS, NUM_CATS) if args.model == "pointnet2"
             else DGCNNSeg(NUM_PARTS, NUM_CATS, k=args.k)).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    bs = args.batch_size
    ckpt = os.path.join(args.out, "best.pt")
    last = os.path.join(args.out, "last.pt")

    def save_atomic(obj, path):  # write to a temp file first so a power cut cannot corrupt the checkpoint
        torch.save(obj, path + ".tmp")
        os.replace(path + ".tmp", path)

    start_epoch, best, resuming = 1, -1.0, False
    if args.resume and os.path.exists(last):
        st = torch.load(last, map_location=device)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start_epoch, best, resuming = st["epoch"] + 1, st["best"], True
        print(f"resuming after epoch {st['epoch']} (best val instance mIoU so far {best:.4f})")

    with open(os.path.join(args.out, "log.csv"), "a" if resuming else "w", newline="") as f:
        log = csv.writer(f)
        if not resuming:
            log.writerow(["epoch", "train_loss", "val_instance_mIoU", "val_class_mIoU", "seconds"])
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            t0, tot, n = time.time(), 0.0, 0
            perm = np.random.permutation(tr)
            for i in range(0, len(perm) - bs + 1, bs):  # drop the last partial batch (BatchNorm)
                b = perm[i:i + bs]
                x = augment(torch.from_numpy(P[b]).to(device))
                y = torch.from_numpy(Y[b]).long().to(device)
                c = torch.from_numpy(C[b]).long().to(device)
                loss = F.cross_entropy(model(x, F.one_hot(c, NUM_CATS).float()).reshape(-1, NUM_PARTS), y.reshape(-1))
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); n += 1
            sched.step()
            val = evaluate(model, P, Y, C, va, device, bs)
            log.writerow([epoch, tot / max(n, 1), val["instance_mIoU"], val["class_mIoU"], time.time() - t0]); f.flush()
            print(f"epoch {epoch:3d}  loss {tot / max(n, 1):.4f}  val inst-mIoU {val['instance_mIoU']:.4f}  "
                  f"class-mIoU {val['class_mIoU']:.4f}  ({time.time() - t0:.0f}s)")
            if val["instance_mIoU"] > best:
                best = val["instance_mIoU"]
                save_atomic(model.state_dict(), ckpt)
            save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                         "epoch": epoch, "best": best}, last)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    test = evaluate(model, P, Y, C, te, device, bs)
    test.update(model=args.model, epochs=args.epochs, num_points=int(P.shape[1]), seed=args.seed,
                n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(te)))
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(test, f, indent=2)
    print(f"\nTEST  instance mIoU {test['instance_mIoU']:.4f}   class mIoU {test['class_mIoU']:.4f}")
    for name, v in test["per_category_mIoU"].items():
        print(f"  {name:<11}{v:.4f}")


if __name__ == "__main__":
    main()
