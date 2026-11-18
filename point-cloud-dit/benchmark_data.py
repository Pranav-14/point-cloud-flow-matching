"""Shape benchmark shared with the team's experiments.

Eight outline shapes, 256 points each. Every shape is centred on its bounding box and
scaled so its largest radius is 1. Training clouds get a random rotation, scale (0.6-1.0),
translation (up to 0.3 per axis) and per-point jitter (std 0.01).
"""
import math

import torch

DEFAULT_SHAPES = ("triangle", "square", "rectangle", "pentagon", "hexagon", "circle", "star", "cross")


def _ngon(k, phase=math.pi / 2):
    a = phase + 2 * math.pi * torch.arange(k, dtype=torch.float64) / k
    return torch.stack([a.cos(), a.sin()], dim=1)


def _star(tips=5, inner=0.4):
    a = math.pi / 2 + math.pi * torch.arange(2 * tips, dtype=torch.float64) / tips
    r = torch.ones(2 * tips, dtype=torch.float64)
    r[1::2] = inner
    return torch.stack([r * a.cos(), r * a.sin()], dim=1)


def _cross(w=0.33):
    return torch.tensor(
        [[w, 1], [-w, 1], [-w, w], [-1, w], [-1, -w], [-w, -w], [-w, -1], [w, -1], [w, -w], [1, -w], [1, w], [w, w]],
        dtype=torch.float64,
    )


VERTICES = {
    "triangle": lambda: _ngon(3),
    "square": lambda: torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=torch.float64),
    "rectangle": lambda: torch.tensor([[-1, -0.5], [1, -0.5], [1, 0.5], [-1, 0.5]], dtype=torch.float64),
    "pentagon": lambda: _ngon(5),
    "hexagon": lambda: _ngon(6),
    "circle": lambda: _ngon(128, phase=0.0),
    "star": _star,
    "cross": _cross,
}


def _normalize(v):
    v = v - (v.min(dim=0).values + v.max(dim=0).values) / 2
    return (v / v.norm(dim=1).max()).float()


class ShapeBank:
    def __init__(self, names=DEFAULT_SHAPES, n_points=256, device="cpu"):
        self.names, self.n_points, self.device = list(names), n_points, device
        self.polygons = []
        for name in self.names:
            start = _normalize(VERTICES[name]()).to(device)
            edge = start.roll(-1, dims=0) - start
            length = edge.norm(dim=1)
            self.polygons.append((start, edge, length, length.cumsum(0)))

    @property
    def n_classes(self):
        return len(self.names)

    def canonical(self, labels, g):
        """Clean clouds: points uniformly distributed along each labelled outline."""
        clouds = torch.empty(len(labels), self.n_points, 2, device=self.device)
        for c, (start, edge, length, ends) in enumerate(self.polygons):
            idx = (labels == c).nonzero(as_tuple=True)[0]
            if len(idx):
                u = torch.rand(len(idx), self.n_points, generator=g, device=self.device) * ends[-1]
                seg = torch.searchsorted(ends, u, right=True).clamp(max=len(ends) - 1)
                frac = (u - (ends[seg] - length[seg])) / length[seg]
                clouds[idx] = start[seg] + frac.unsqueeze(-1) * edge[seg]
        return clouds

    def augment(self, x, g, rotate=True, scale=(0.6, 1.0), translate=0.3, jitter=0.01):
        B, device = x.shape[0], x.device
        if rotate:
            angle = torch.rand(B, generator=g, device=device) * 2 * math.pi
            c, s = angle.cos(), angle.sin()
            x = x @ torch.stack([torch.stack([c, s], -1), torch.stack([-s, c], -1)], -2)
        x = x * (scale[0] + (scale[1] - scale[0]) * torch.rand(B, 1, 1, generator=g, device=device))
        x = x + (torch.rand(B, 1, 2, generator=g, device=device) * 2 - 1) * translate
        if jitter > 0:
            x = x + torch.randn(x.shape, generator=g, device=device) * jitter
        return x

    def batch(self, size, g, **aug):
        labels = torch.randint(self.n_classes, (size,), generator=g, device=self.device)
        return self.augment(self.canonical(labels, g), g, **aug), labels

    def fixed(self, per_class, seed, augment=False, **aug):
        """The same seed always gives the same clouds."""
        g = torch.Generator(device=self.device).manual_seed(seed)
        labels = torch.arange(self.n_classes, device=self.device).repeat_interleave(per_class)
        x = self.canonical(labels, g)
        return (self.augment(x, g, **aug) if augment else x), labels


def augmentation(args):
    return dict(rotate=not args.no_rot, scale=(args.scale_min, args.scale_max), translate=args.trans, jitter=args.jitter)
