import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def chamfer(a, b):
    """Symmetric Chamfer distance (squared) between clouds a (A, N, 2) and b (A, M, 2).
    For each point: squared distance to its nearest point in the other cloud.
    Independent of point order. Returns one value per pair, shape (A,)."""
    d = torch.cdist(a, b) ** 2
    return d.min(dim=2).values.mean(dim=1) + d.min(dim=1).values.mean(dim=1)


def nearest_train_chamfer(samples, train):
    """For each sample, the Chamfer distance to its closest training cloud and that
    cloud's index. When overfitting, a good model gets this close to 0."""
    dists = []
    for sample in samples:
        dists.append(chamfer(sample.expand(len(train), -1, -1), train))
    return torch.stack(dists).min(dim=1)


def save_clouds(clouds, path, ncols=8):
    clouds = clouds.detach().cpu()
    nrows = math.ceil(len(clouds) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.6 * ncols, 1.6 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, cloud in zip(axes.flat, clouds):
        ax.set_axis_on()
        ax.scatter(cloud[:, 0], cloud[:, 1], s=2, linewidths=0)
        ax.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), xticks=[], yticks=[], aspect="equal")
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=120)
    plt.close(fig)
