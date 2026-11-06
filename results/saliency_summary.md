# Saliency statistics (200 test shapes per row)

"Boundary point" = a point with a different true part among its 16 nearest points.
Ratio = average saliency of boundary points / average saliency of interior points, computed per shape.
"Correct / wrong" = average relative saliency of correctly / wrongly predicted points (1.0 = shape average).
Shapes without a clear boundary are skipped (5 to 9 of 200).

| model | seed | saliency | boundary points | ratio, mean | ratio, median | shapes with ratio above 1 | correct | wrong |
|---|---|---|---|---|---|---|---|---|
| DGCNN | 0 | smoothed | 16% | 1.18 | 1.12 | 129 of 195 | 1.00 | 1.07 |
| DGCNN | 1 | smoothed | 17% | 1.14 | 1.06 | 122 of 191 | 1.00 | 1.04 |
| DGCNN | 2 | smoothed | 16% | 1.09 | 1.02 | 108 of 197 | 1.00 | 0.97 |
| DGCNN | 0 | raw | 16% | 1.26 | 1.15 | 135 of 195 | 0.99 | 1.10 |
| PointNet++ | 0 | smoothed | 16% | 1.03 | 0.94 | 78 of 195 | 1.00 | 0.95 |
| PointNet++ | 1 | smoothed | 17% | 1.01 | 0.92 | 74 of 191 | 1.00 | 0.95 |
| PointNet++ | 2 | smoothed | 16% | 1.00 | 0.90 | 70 of 197 | 1.01 | 0.94 |
| PointNet++ | 0 | raw | 16% | 1.11 | 0.98 | 89 of 195 | 1.00 | 0.98 |

Smoothed = SmoothGrad (10 noisy copies, sigma 0.01) plus averaging over the 16 nearest points. Raw = plain gradient.
