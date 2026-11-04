"""Per-point gradient saliency for a trained PointNet++ or DGCNN part-segmentation model.

For a test shape, we take the sum of the predicted-part logits over all points and compute its gradient
with respect to the input coordinates. The length of that gradient at each point shows how sensitive the
model's prediction is to moving that point. It is a sensitivity map, not proof that the model "uses" a
region, so treat it as a diagnostic.

Example (Colab, after training):
    python explain.py --model pointnet2 --cache cache_small.npz --ckpt runs/pn2_s0/best.pt --out figs/pn2
    python explain.py --model dgcnn     --cache cache_small.npz --ckpt runs/dgcn_s0/best.pt --out figs/dgcnn
Use the same --seed as in training so the same test shapes are used.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from models import DGCNNSeg, PointNet2Seg
from train import CAT_NAMES, NPARTS, NUM_CATS, NUM_PARTS, OFFSETS, split_indices


def scatter(ax, pts, c, title, cmap, vmin, vmax):
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=c, s=4, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=15, azim=-60)
    ax.set_axis_off()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["pointnet2", "dgcnn"], required=True)
    ap.add_argument("--cache", default="cache_small.npz")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num_points", type=int, default=2048)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0, help="same seed as training (decides the test split)")
    ap.add_argument("--n_shapes", type=int, default=8)
    ap.add_argument("--noise_samples", type=int, default=10, help="SmoothGrad copies (1 = plain gradient)")
    ap.add_argument("--sigma", type=float, default=0.01, help="SmoothGrad noise std (shapes are scaled to unit radius)")
    ap.add_argument("--smooth_k", type=int, default=16, help="average saliency over k nearest points (1 = off)")
    ap.add_argument("--boundary_k", type=int, default=16,
                    help="a point is a boundary point if one of its k nearest points has a different true part")
    ap.add_argument("--n_figs", type=int, default=8, help="save figures only for the first n shapes")
    ap.add_argument("--out", default="figs")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    d = np.load(args.cache)
    P, Y, C = d["points"][:, :args.num_points], d["labels"][:, :args.num_points], d["cats"]
    test_idx = split_indices(C, args.seed)[2]
    chosen = np.random.RandomState(0).choice(test_idx, min(args.n_shapes, len(test_idx)), replace=False)

    model = (PointNet2Seg(NUM_PARTS, NUM_CATS) if args.model == "pointnet2"
             else DGCNNSeg(NUM_PARTS, NUM_CATS, k=args.k)).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    sal_correct, sal_wrong = [], []
    b_rel, i_rel, b_frac = [], [], []  # per shape: boundary saliency, interior saliency, boundary fraction
    for si, s in enumerate(chosen):
        lo, k = OFFSETS[C[s]], NPARTS[C[s]]
        x0 = torch.from_numpy(P[s:s + 1]).to(device)
        onehot = F.one_hot(torch.tensor([int(C[s])], device=device), NUM_CATS).float()
        with torch.no_grad():  # prediction on the clean input
            pred_t = model(x0, onehot)[0][:, lo:lo + k].argmax(-1)
        # SmoothGrad: average the gradient over several slightly noisy copies of the input
        sal = np.zeros(P.shape[1])
        for _ in range(args.noise_samples):
            x = x0 + args.sigma * torch.randn_like(x0) if args.noise_samples > 1 else x0.clone()
            x = x.detach().requires_grad_(True)
            logits = model(x, onehot)[0][:, lo:lo + k]  # only this category's parts
            score = logits.gather(1, pred_t.unsqueeze(1)).sum()
            model.zero_grad()
            score.backward()
            sal += x.grad[0].norm(dim=-1).cpu().numpy() / args.noise_samples
        p = P[s]
        dist = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)
        order = np.argsort(dist, axis=1)[:, :max(args.smooth_k, args.boundary_k + 1)]  # nearest points (self first)
        if args.smooth_k > 1:  # average each point's saliency with its nearest neighbours
            sal = sal[order[:, :args.smooth_k]].mean(1)
        pred = pred_t.cpu().numpy() + lo
        gt = Y[s].astype(np.int64)

        ok = pred == gt
        rel = sal / (sal.mean() + 1e-12)  # relative to this shape's average
        sal_correct += list(rel[ok]); sal_wrong += list(rel[~ok])

        # boundary test: a point is a boundary point if any of its nearest points has a different true part
        is_b = (gt[order[:, :args.boundary_k + 1]] != gt[:, None]).any(1)
        if is_b.sum() >= 20 and (~is_b).sum() >= 20:
            b_rel.append(rel[is_b].mean()); i_rel.append(rel[~is_b].mean()); b_frac.append(is_b.mean())

        if si < args.n_figs:
            sal_plot = np.clip(sal / (np.percentile(sal, 99) + 1e-12), 0, 1)
            fig = plt.figure(figsize=(12, 4))
            scatter(fig.add_subplot(1, 3, 1, projection="3d"), p, gt - lo, "ground truth parts", "tab10", 0, 9)
            scatter(fig.add_subplot(1, 3, 2, projection="3d"), p, pred - lo,
                    f"prediction ({100 * ok.mean():.0f}% points correct)", "tab10", 0, 9)
            scatter(fig.add_subplot(1, 3, 3, projection="3d"), p, sal_plot, "gradient saliency", "inferno", 0, 1)
            fig.suptitle(f"{CAT_NAMES[C[s]]} (test shape {s})", fontsize=12)
            fig.savefig(os.path.join(args.out, f"{CAT_NAMES[C[s]]}_{s}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved {min(args.n_figs, len(chosen))} figures to {args.out}/")
    if b_rel:
        ratios = np.array(b_rel) / np.array(i_rel)
        print(f"boundary test over {len(ratios)} of {len(chosen)} shapes (shapes with a clear boundary): "
              f"{100 * np.mean(b_frac):.0f}% of points are boundary points; "
              f"mean relative saliency boundary {np.mean(b_rel):.2f} vs interior {np.mean(i_rel):.2f}; "
              f"boundary/interior ratio per shape: mean {ratios.mean():.2f}, median {np.median(ratios):.2f}, "
              f"above 1 in {int((ratios > 1).sum())} of {len(ratios)} shapes")
    if sal_wrong:
        print(f"mean relative saliency: correctly predicted points {np.mean(sal_correct):.2f}, "
              f"wrongly predicted points {np.mean(sal_wrong):.2f}  (over {len(chosen)} shapes; 1.0 = shape average)")
    else:
        print("No wrongly predicted points in the chosen shapes.")


if __name__ == "__main__":
    main()
