"""Point-set metrics. All clouds are tensors (B, N, 2).

Chamfer (CD)   squared distance to the nearest point of the other cloud, both directions added
EMD            mean distance under the best one-to-one point matching
MMD-CD         for each real cloud, CD to its closest generated cloud, averaged (lower is better)
COV-CD         share of real clouds that are the nearest real cloud of some generated cloud
1-NNA          nearest-neighbour accuracy at telling generated from real clouds (50% is ideal)
Class accuracy share of generated clouds whose nearest real cloud has the requested class

MMD-CD, COV-CD and 1-NNA are computed per class and then averaged.
"""
import torch

from utils import chamfer


@torch.no_grad()
def pairwise_chamfer(a, b, chunk=512):
    """(A, N, 2) x (R, M, 2) -> (A, R) matrix of Chamfer distances."""
    rows = []
    for cloud in a:
        parts = [chamfer(cloud.expand(len(b[i:i + chunk]), -1, -1), b[i:i + chunk]) for i in range(0, len(b), chunk)]
        rows.append(torch.cat(parts))
    return torch.stack(rows)


def emd(a, b):
    from scipy.optimize import linear_sum_assignment

    values = []
    for p, q in zip(a.detach().cpu(), b.detach().cpu()):
        d = torch.cdist(p, q).numpy()
        rows, cols = linear_sum_assignment(d)
        values.append(float(d[rows, cols].mean()))
    return torch.tensor(values)


@torch.no_grad()
def nearest_same_class(gen, gen_y, ref, ref_y):
    """Chamfer distance and index (into ref) of each cloud's closest same-class reference."""
    dist = torch.empty(len(gen), device=gen.device)
    index = torch.empty(len(gen), dtype=torch.long, device=gen.device)
    for c in gen_y.unique():
        g, r = (gen_y == c).nonzero(as_tuple=True)[0], (ref_y == c).nonzero(as_tuple=True)[0]
        best = pairwise_chamfer(gen[g], ref[r]).min(dim=1)
        dist[g], index[g] = best.values, r[best.indices]
    return dist, index


@torch.no_grad()
def distribution_metrics(gen, gen_y, ref, ref_y, n_classes):
    per_class = {"mmd_cd": [], "cov_cd": [], "nna": []}
    for c in range(n_classes):
        G, R = gen[gen_y == c], ref[ref_y == c]
        gr, gg, rr = pairwise_chamfer(G, R), pairwise_chamfer(G, G), pairwise_chamfer(R, R)
        per_class["mmd_cd"].append(gr.min(dim=0).values.mean().item())
        per_class["cov_cd"].append(gr.argmin(dim=1).unique().numel() / len(R))
        full = torch.cat([torch.cat([gg, gr], dim=1), torch.cat([gr.T, rr], dim=1)], dim=0)
        full.fill_diagonal_(float("inf"))
        is_generated = torch.cat([torch.ones(len(G)), torch.zeros(len(R))]).bool().to(full.device)
        per_class["nna"].append((is_generated == is_generated[full.argmin(dim=1)]).float().mean().item())
    return {k: sum(v) / len(v) for k, v in per_class.items()}, per_class


@torch.no_grad()
def class_accuracy(gen, gen_y, ref, ref_y):
    return (ref_y[pairwise_chamfer(gen, ref).argmin(dim=1)] == gen_y).float().mean().item()
