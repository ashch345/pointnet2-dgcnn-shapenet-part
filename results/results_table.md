## Per run

| run | model | seed | instance mIoU | class mIoU |
|---|---|---|---|---|
| dgcn_s0 | dgcnn | 0 | 0.8140 | 0.7958 |
| dgcn_s1 | dgcnn | 1 | 0.8066 | 0.7942 |
| dgcn_s2 | dgcnn | 2 | 0.8132 | 0.8027 |
| pn2_s0 | pointnet2 | 0 | 0.8143 | 0.7984 |
| pn2_s1 | pointnet2 | 1 | 0.8058 | 0.7950 |
| pn2_s2 | pointnet2 | 2 | 0.8097 | 0.8010 |

## Mean ± standard deviation over seeds

| model | seeds | instance mIoU | class mIoU |
|---|---|---|---|
| dgcnn | 0,1,2 | 0.8113 ± 0.0041 | 0.7975 ± 0.0045 |
| pointnet2 | 0,1,2 | 0.8099 ± 0.0043 | 0.7981 ± 0.0030 |

## Per category (mean ± std over seeds; small categories are very noisy)

| category | dgcnn | pointnet2 |
|---|---|---|
| Airplane | 0.7981 ± 0.0185 | 0.7825 ± 0.0308 |
| Bag | 0.8180 ± 0.0417 | 0.8498 ± 0.0427 |
| Cap | 0.8195 ± 0.0730 | 0.7830 ± 0.0789 |
| Car | 0.7147 ± 0.0205 | 0.7112 ± 0.0283 |
| Chair | 0.8625 ± 0.0136 | 0.8650 ± 0.0112 |
| Earphone | 0.7337 ± 0.0591 | 0.7423 ± 0.0705 |
| Guitar | 0.9128 ± 0.0143 | 0.9003 ± 0.0118 |
| Knife | 0.8360 ± 0.0379 | 0.8469 ± 0.0236 |
| Lamp | 0.8297 ± 0.0184 | 0.8334 ± 0.0159 |
| Laptop | 0.9504 ± 0.0196 | 0.9543 ± 0.0121 |
| Motorbike | 0.6221 ± 0.0280 | 0.6267 ± 0.0165 |
| Mug | 0.9263 ± 0.0092 | 0.9254 ± 0.0226 |
| Pistol | 0.8239 ± 0.0272 | 0.8204 ± 0.0196 |
| Rocket | 0.5448 ± 0.0564 | 0.5896 ± 0.0370 |
| Skateboard | 0.7549 ± 0.0487 | 0.7473 ± 0.0509 |
| Table | 0.8133 ± 0.0333 | 0.7922 ± 0.0210 |
