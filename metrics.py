"""Standard ShapeNet-Part metrics (numpy only).

For each shape, IoU is averaged over the parts of that shape's category
(a part absent from both prediction and ground truth counts as IoU = 1).
  - instance mIoU: mean over all shapes
  - class mIoU:    mean over categories of the mean IoU of their shapes
"""
import numpy as np


def shape_iou(pred, gt, lo, k):
    """pred, gt: 1-D arrays of global part ids; the category owns ids lo .. lo+k-1."""
    ious = []
    for p in range(lo, lo + k):
        inter = np.sum((pred == p) & (gt == p))
        union = np.sum((pred == p) | (gt == p))
        ious.append(1.0 if union == 0 else inter / union)
    return float(np.mean(ious)), ious


def summarize(shape_ious, shape_cats, cat_names):
    shape_ious = np.asarray(shape_ious)
    shape_cats = np.asarray(shape_cats)
    per_cat = {}
    for ci, name in enumerate(cat_names):
        m = shape_cats == ci
        if m.any():
            per_cat[name] = float(shape_ious[m].mean())
    return {
        "instance_mIoU": float(shape_ious.mean()),
        "class_mIoU": float(np.mean(list(per_cat.values()))),
        "per_category_mIoU": per_cat,
    }
