"""Collect runs/*/results.json into one table (mean and standard deviation over seeds).

Usage:
    python aggregate.py --runs runs --epochs 30 --out results_table.md

Only runs with the given number of epochs are used, so short test runs are ignored.
Run both models with the same seeds. Seed also decides the train/val/test split.
"""
import argparse
import glob
import json
import os

import numpy as np


def fmt(values):
    values = np.asarray(values, dtype=float)
    if len(values) > 1:
        return f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"
    return f"{values.mean():.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="results_table.md")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(args.runs, "*", "results.json"))):
        with open(f) as fh:
            r = json.load(fh)
        if r.get("epochs") == args.epochs:
            rows.append(dict(r, run=os.path.basename(os.path.dirname(f))))
    if not rows:
        raise SystemExit(f"No results.json with epochs={args.epochs} found under {args.runs}/")

    models = sorted({r["model"] for r in rows})
    out = ["## Per run", "", "| run | model | seed | instance mIoU | class mIoU |", "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["model"], r["seed"])):
        out.append(f"| {r['run']} | {r['model']} | {r['seed']} | {r['instance_mIoU']:.4f} | {r['class_mIoU']:.4f} |")

    out += ["", "## Mean ± standard deviation over seeds", "",
            "| model | seeds | instance mIoU | class mIoU |", "|---|---|---|---|"]
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        seeds = ",".join(str(r["seed"]) for r in sorted(rs, key=lambda r: r["seed"]))
        out.append(f"| {m} | {seeds} | {fmt([r['instance_mIoU'] for r in rs])} | {fmt([r['class_mIoU'] for r in rs])} |")

    cats = list(rows[0]["per_category_mIoU"].keys())
    out += ["", "## Per category (mean ± std over seeds; small categories are very noisy)", "",
            "| category | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for c in cats:
        cells = []
        for m in models:
            vals = [r["per_category_mIoU"][c] for r in rows if r["model"] == m and c in r["per_category_mIoU"]]
            cells.append(fmt(vals) if vals else "-")
        out.append(f"| {c} | " + " | ".join(cells) + " |")

    text = "\n".join(out) + "\n"
    with open(args.out, "w") as f:
        f.write(text)
    print(text)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
