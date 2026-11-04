# PointNet++ vs DGCNN for part segmentation on ShapeNet-Part

A small learning project. I work on medical image segmentation in 2D, and I wanted a first hands-on project with 3D data. I implemented two standard point cloud models in PyTorch, trained both on the same data with three random seeds, and compared them. I also tried a simple explanation method (gradient saliency).

This is not a research result. The numbers cannot be compared with published results (see Limitations).

## The task
Each object is a cloud of 2,048 points. The model must give every point a part label, for example seat, back or leg for a chair. There are 16 object categories and 50 part labels in total. The category of the object is given to the model.

## Data
I used the expert-verified part annotations of ShapeNet-Part (Yi et al., 2016), which are built on ShapeNet (Chang et al., 2015). The data is not included in this repository. Please get it from the ShapeNet website and check its licence.

- 2,602 shapes: up to 200 shapes per category (fewer in small categories, for example 55 caps and 66 rockets).
- 2,048 points per shape, centred and scaled to unit size.
- A random 80/10/10 train/validation/test split inside each category (2,079 / 260 / 263 shapes). The seed also decides the split.

## Models
Both are written in plain PyTorch, without custom CUDA code. They are not exact copies of the original implementations.

- **PointNet++** (Qi et al., 2017): single-scale grouping, 3 set-abstraction levels, feature propagation back to every point. 1.39M parameters.
- **DGCNN** (Wang et al., 2019): three EdgeConv layers on a dynamic 20-nearest-neighbour graph, a global feature and the category embedding. 0.65M parameters.

Training: 30 epochs, batch size 16, Adam (learning rate 1e-3, weight decay 1e-4), cosine learning-rate schedule, cross-entropy loss, random scaling (0.8 to 1.25) and shifting (±0.1) as augmentation. The epoch with the best validation instance mIoU is tested once. At test time, predictions are limited to the parts of the given category. No hyperparameter tuning was done.

## Results
Test set, three seeds (0, 1, 2) per model, mean ± standard deviation. Run in one Colab notebook (Tesla T4 GPU, PyTorch 2.11).

| Model | Instance mIoU | Class mIoU | Time per epoch |
|---|---|---|---|
| PointNet++ | 0.8099 ± 0.0043 | 0.7981 ± 0.0030 | about 27 s |
| DGCNN | 0.8113 ± 0.0041 | 0.7975 ± 0.0045 | about 42 s |

The two models perform the same within noise. The instance mIoU differs by 0.0014 in favour of DGCNN, the class mIoU by 0.0006 in favour of PointNet++, and the spread between seeds is about 0.004. DGCNN has less than half the parameters but was about 1.6 times slower per epoch in this implementation.

All runs and per-category scores are in [results/results_table.md](results/results_table.md). Per-category numbers are very noisy, because each seed gives a different small test set (Rocket, for example, has only a handful of test shapes).

GPU training is not exactly repeatable. An earlier run of the same seeds gave slightly different scores (differences up to about 0.003). The numbers above are from the final notebook run.

## Explaining the predictions
`explain.py` computes gradient saliency: how much the model's prediction for a shape changes when each point is moved a little. Plain gradients are very noisy (single bright points), so I also used a smoothed version: the gradient averaged over 10 slightly noisy copies of the shape (SmoothGrad) and over each point's 16 nearest neighbours.

Brighter means more sensitive. Saliency shows what the prediction is sensitive to. It does not prove what the model "uses".

[FIGURES TO BE ADDED: side-by-side examples of PointNet++ and DGCNN on the same test shapes, with my observations.]

### Is saliency higher near part boundaries?
I defined a boundary point as a point with a different true part among its 16 nearest points (about 16% of points). For 200 test shapes per model and seed, I compared the average saliency of boundary points with interior points, shape by shape. Full numbers are in [results/saliency_summary.md](results/saliency_summary.md).

| Model (smoothed saliency) | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| DGCNN: mean ratio (shapes above 1) | 1.18 (129 of 195) | 1.14 (122 of 191) | 1.09 (108 of 197) |
| PointNet++: mean ratio (shapes above 1) | 1.03 (78 of 195) | 1.01 (74 of 191) | 1.00 (70 of 197) |

A ratio of 1 means no difference. For DGCNN, saliency was somewhat higher near part boundaries in all three seeds, but the effect is small (the median ratio was 1.12, 1.06 and 1.02), and in seed 2 only 55% of shapes were above 1, which is close to chance. For PointNet++ I found no effect (median ratio below 1 in every seed). I did not test why the two models differ. Boundary points are also often on edges or thin parts, so geometry alone could contribute.

I also compared correctly and wrongly predicted points. Wrong points were slightly more sensitive for DGCNN in seeds 0 and 1 (1.07 and 1.04 against 1.00) and slightly less sensitive for PointNet++ (0.94 to 0.95), with no consistent pattern across models and seeds.

## Limitations
- Small subset (2,602 shapes), my own random split (not the official one), no surface normals, 30 epochs, no tuning. Scores are lower than published ones and cannot be compared with them.
- Three seeds per model. The model difference is within noise.
- My implementations are simplified and not checked against the originals.
- Saliency effects are small and differ between the two models. PointNet++ chooses its centre points with a random start, so its saliency maps can change a little between runs.
- Only ShapeNet-Part (everyday objects). I have not tried this on medical or dental 3D data.

## How to run
Needs Python, numpy, matplotlib and PyTorch. A GPU is strongly recommended: the plain-PyTorch models are very slow on a CPU.

```
pip install -r requirements.txt

# 1. cache the data (use --max_per_cat 200 for the subset used here)
python prepare_data.py --root /path/to/PartAnnotation --max_per_cat 200 --num_points 2048 --out cache_small.npz

# 2. train (same seed for both models)
python train.py --model pointnet2 --cache cache_small.npz --epochs 30 --batch_size 16 --seed 0 --out runs/pn2_s0
python train.py --model dgcnn     --cache cache_small.npz --epochs 30 --batch_size 16 --seed 0 --out runs/dgcn_s0
# add --resume to continue an interrupted run

# 3. results table over seeds
python aggregate.py --runs runs --epochs 30 --out results_table.md

# 4. saliency figures and statistics
python explain.py --model pointnet2 --cache cache_small.npz --ckpt runs/pn2_s0/best.pt --seed 0 --n_shapes 200 --out figs/pn2
```

The notebook `notebooks/run_all_experiments.ipynb` runs all of the above on Google Colab: checks, 3 seeds x 2 models, results table, saliency figures and statistics. It is the notebook I ran for the results in this README.

## Files
| File | What it does |
|---|---|
| `prepare_data.py` | reads the raw files, samples points, saves one cache file |
| `models.py` | PointNet++ and DGCNN |
| `train.py` | training and testing |
| `metrics.py` | instance and class mIoU |
| `aggregate.py` | mean ± std over seeds |
| `explain.py` | gradient saliency figures and statistics |
| `notebooks/` | the Colab notebook with its outputs |
| `results/` | results table and saliency statistics |

## Note on AI assistance
I wrote this code with help from an AI assistant (Claude, by Anthropic). I ran all the experiments myself and the results above are from my own runs.

## References
- Yi et al., "A scalable active framework for region annotation in 3D shape collections", ACM Transactions on Graphics, 2016.
- Chang et al., "ShapeNet: An Information-Rich 3D Model Repository", arXiv:1512.03012, 2015.
- Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space", NeurIPS 2017.
- Wang et al., "Dynamic Graph CNN for Learning on Point Clouds", ACM Transactions on Graphics, 2019.
