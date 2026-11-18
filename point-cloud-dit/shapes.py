"""Toy dataset: 2D point clouds sampled from the outlines of basic shapes.

Every cloud is a tensor (n_points, 2). Points are drawn independently and
uniformly along the outline, so their order carries no meaning.

Each base shape has radius 1 around the origin. A random rotation, scale and
translation are applied such that the whole cloud stays inside [-1, 1]^2.
"""
import math

import torch

SHAPES = ("circle", "square", "triangle", "star")


def _regular_polygon(k):
    angles = 2 * math.pi * torch.arange(k) / k
    return torch.stack([angles.cos(), angles.sin()], dim=1)


def _star(tips=5, inner_radius=0.45):
    angles = 2 * math.pi * torch.arange(2 * tips) / (2 * tips)
    radii = torch.ones(2 * tips)
    radii[1::2] = inner_radius
    return torch.stack([radii * angles.cos(), radii * angles.sin()], dim=1)


def _polygon_outline(vertices, n, g):
    """n points uniformly distributed (by length) along a closed polygon."""
    edges = vertices.roll(-1, dims=0) - vertices
    lengths = edges.norm(dim=1)
    ends = lengths.cumsum(0)
    u = torch.rand(n, generator=g) * ends[-1]
    edge = torch.searchsorted(ends, u, right=True).clamp(max=len(vertices) - 1)
    frac = (u - (ends[edge] - lengths[edge])) / lengths[edge]
    return vertices[edge] + frac[:, None] * edges[edge]


def base_shape(name, n, g):
    if name == "circle":
        angles = 2 * math.pi * torch.rand(n, generator=g)
        return torch.stack([angles.cos(), angles.sin()], dim=1)
    if name == "square":
        return _polygon_outline(_regular_polygon(4), n, g)
    if name == "triangle":
        return _polygon_outline(_regular_polygon(3), n, g)
    if name == "star":
        return _polygon_outline(_star(), n, g)
    raise ValueError(name)


def sample_cloud(name, n_points, g, min_scale=0.4, max_scale=0.8):
    points = base_shape(name, n_points, g)

    angle = 2 * math.pi * torch.rand(1, generator=g).item()
    c, s = math.cos(angle), math.sin(angle)
    rotation = torch.tensor([[c, -s], [s, c]])

    scale = min_scale + (max_scale - min_scale) * torch.rand(1, generator=g).item()
    # radius after scaling is `scale`, so a shift up to (1 - scale) stays in bounds
    shift = (torch.rand(2, generator=g) * 2 - 1) * (1 - scale)
    return scale * points @ rotation.T + shift


def make_dataset(n_clouds, n_points=256, seed=0, shapes=SHAPES):
    """Returns clouds (n_clouds, n_points, 2) and shape labels (n_clouds,).
    Shapes are cycled, so the classes are as balanced as possible."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.arange(n_clouds) % len(shapes)
    clouds = torch.stack([sample_cloud(shapes[i], n_points, g) for i in labels.tolist()])
    return clouds, labels
