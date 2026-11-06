"""PointNet++ (single-scale grouping) and DGCNN for ShapeNet-Part segmentation.

Pure PyTorch, no custom CUDA ops, so they run anywhere (slower than the CUDA versions).
Both take xyz (B, N, 3) and a one-hot category vector (B, 16) and return logits (B, N, num_parts).
"""
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------- PointNet++
def square_distance(a, b):
    """a: (B,N,3), b: (B,M,3) -> (B,N,M) squared distances."""
    return torch.cdist(a, b) ** 2


def index_points(points, idx):
    """points: (B,N,C); idx: (B,S) or (B,S,K) -> (B,S,C) or (B,S,K,C)."""
    B = points.shape[0]
    view = [B] + [1] * (idx.dim() - 1)
    batch = torch.arange(B, device=points.device).view(view).expand_as(idx)
    return points[batch, idx, :]


def farthest_point_sample(xyz, npoint):
    B, N, _ = xyz.shape
    dev = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=dev)
    dist = torch.full((B, N), 1e10, device=dev)
    farthest = torch.randint(0, N, (B,), device=dev)
    batch = torch.arange(B, device=dev)
    for i in range(npoint):
        centroids[:, i] = farthest
        c = xyz[batch, farthest, :].view(B, 1, 3)
        dist = torch.minimum(dist, ((xyz - c) ** 2).sum(-1))
        farthest = dist.argmax(-1)
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    idx = torch.arange(N, device=xyz.device).view(1, 1, N).repeat(B, S, 1)
    idx[square_distance(new_xyz, xyz) > radius ** 2] = N
    idx = idx.sort(dim=-1)[0][:, :, :nsample]
    first = idx[:, :, 0:1].expand(-1, -1, nsample)
    mask = idx == N
    idx[mask] = first[mask]
    return idx


class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint, self.radius, self.nsample, self.group_all = npoint, radius, nsample, group_all
        layers, last = [], in_channel
        for out in mlp:
            layers += [nn.Conv2d(last, out, 1), nn.BatchNorm2d(out), nn.ReLU(inplace=True)]
            last = out
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points):
        B, N, _ = xyz.shape
        if self.group_all:
            new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
            grouped = xyz.unsqueeze(1)
            if points is not None:
                grouped = torch.cat([grouped, points.unsqueeze(1)], -1)
        else:
            new_xyz = index_points(xyz, farthest_point_sample(xyz, self.npoint))
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped = index_points(xyz, idx) - new_xyz.unsqueeze(2)
            if points is not None:
                grouped = torch.cat([grouped, index_points(points, idx)], -1)
        feats = self.mlp(grouped.permute(0, 3, 2, 1))  # (B, C, nsample, S)
        return new_xyz, feats.max(2)[0].permute(0, 2, 1)  # (B, S, C)


class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        layers, last = [], in_channel
        for out in mlp:
            layers += [nn.Conv1d(last, out, 1), nn.BatchNorm1d(out), nn.ReLU(inplace=True)]
            last = out
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz1, xyz2, points1, points2):
        """Interpolate points2 (at xyz2) onto xyz1 using 3 nearest neighbours."""
        B, N, _ = xyz1.shape
        S = xyz2.shape[1]
        if S == 1:
            interp = points2.expand(-1, N, -1)
        else:
            d, idx = torch.topk(square_distance(xyz1, xyz2), 3, dim=-1, largest=False)
            w = 1.0 / (d + 1e-8)
            w = w / w.sum(-1, keepdim=True)
            interp = (index_points(points2, idx) * w.unsqueeze(-1)).sum(2)
        new = interp if points1 is None else torch.cat([points1, interp], -1)
        return self.mlp(new.permute(0, 2, 1)).permute(0, 2, 1)


class PointNet2Seg(nn.Module):
    def __init__(self, num_parts=50, num_cats=16):
        super().__init__()
        self.sa1 = SetAbstraction(512, 0.2, 32, 3, [64, 64, 128])
        self.sa2 = SetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256])
        self.sa3 = SetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], group_all=True)
        self.fp3 = FeaturePropagation(1024 + 256, [256, 256])
        self.fp2 = FeaturePropagation(256 + 128, [256, 128])
        self.fp1 = FeaturePropagation(128 + num_cats + 3, [128, 128])
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Conv1d(128, num_parts, 1))

    def forward(self, xyz, onehot):
        B, N, _ = xyz.shape
        l1_xyz, l1 = self.sa1(xyz, None)
        l2_xyz, l2 = self.sa2(l1_xyz, l1)
        l3_xyz, l3 = self.sa3(l2_xyz, l2)
        l2 = self.fp3(l2_xyz, l3_xyz, l2, l3)
        l1 = self.fp2(l1_xyz, l2_xyz, l1, l2)
        l0_in = torch.cat([onehot.view(B, 1, -1).expand(-1, N, -1), xyz], -1)
        l0 = self.fp1(xyz, l1_xyz, l0_in, l1)
        return self.head(l0.permute(0, 2, 1)).permute(0, 2, 1)


# ----------------------------------------------------------------------------- DGCNN
def knn(x, k):
    """x: (B,C,N) -> indices of the k nearest neighbours (B,N,k)."""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = (x ** 2).sum(1, keepdim=True)
    return (-xx - inner - xx.transpose(2, 1)).topk(k=k, dim=-1)[1]


def graph_feature(x, k):
    """x: (B,C,N) -> edge features (B,2C,N,k) = [neighbour - centre, centre]."""
    B, C, N = x.shape
    idx = knn(x, k)
    idx = (idx + torch.arange(B, device=x.device).view(-1, 1, 1) * N).view(-1)
    xt = x.transpose(2, 1).contiguous()
    feat = xt.view(B * N, C)[idx].view(B, N, k, C)
    xt = xt.view(B, N, 1, C).expand(-1, -1, k, -1)
    return torch.cat([feat - xt, xt], dim=3).permute(0, 3, 1, 2).contiguous()


class DGCNNSeg(nn.Module):
    def __init__(self, num_parts=50, num_cats=16, k=20, emb=1024):
        super().__init__()
        self.k = k

        def edge(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2))

        self.e1, self.e2, self.e3 = edge(6, 64), edge(128, 64), edge(128, 64)
        self.glob = nn.Sequential(nn.Conv1d(192, emb, 1, bias=False), nn.BatchNorm1d(emb), nn.LeakyReLU(0.2))
        self.cat_emb = nn.Sequential(nn.Conv1d(num_cats, 64, 1, bias=False), nn.BatchNorm1d(64), nn.LeakyReLU(0.2))
        self.mlp = nn.Sequential(
            nn.Conv1d(192 + emb + 64, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(0.5),
            nn.Conv1d(256, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(0.5),
            nn.Conv1d(256, 128, 1, bias=False), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Conv1d(128, num_parts, 1))

    def forward(self, xyz, onehot):
        B, N, _ = xyz.shape
        x = xyz.transpose(1, 2)
        x1 = self.e1(graph_feature(x, self.k)).max(-1)[0]
        x2 = self.e2(graph_feature(x1, self.k)).max(-1)[0]
        x3 = self.e3(graph_feature(x2, self.k)).max(-1)[0]
        local = torch.cat([x1, x2, x3], 1)  # (B,192,N)
        g = self.glob(local).max(-1, keepdim=True)[0]  # (B,emb,1)
        c = self.cat_emb(onehot.view(B, -1, 1))  # (B,64,1)
        feat = torch.cat([g, c], 1).expand(-1, -1, N)
        return self.mlp(torch.cat([feat, local], 1)).transpose(1, 2)
