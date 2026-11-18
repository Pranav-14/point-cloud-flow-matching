"""DiT and PTv3 for images, built from the point-cloud models.

An image is cut into patches and every patch becomes one token. Noise is in the
colours only: the patch positions are a fixed grid. Both models therefore receive
a fixed 2D sin-cos position embedding, and PTv3 sorts and finds neighbours on the grid.

Each training image has its own label, so a sample can be compared with the image
it was asked to reproduce.
"""
import torch
import torch.nn as nn

from dit import PointDiT
from ptv3 import PointPTv3


def sincos_pos_embed_2d(dim, grid):
    """Fixed 2D sin-cos position embedding, shape (grid ** 2, dim)."""
    assert dim % 4 == 0
    ys, xs = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")

    def embed_1d(d, pos):
        omega = 1.0 / 10000 ** (torch.arange(d // 2, dtype=torch.float64) / (d / 2))
        out = pos.flatten().double()[:, None] * omega[None]
        return torch.cat([out.sin(), out.cos()], dim=1)

    return torch.cat([embed_1d(dim // 2, ys), embed_1d(dim // 2, xs)], dim=1).float()


def grid_coords(grid):
    """Patch centres in [-1, 1]^2, row by row, shape (grid ** 2, 2)."""
    centres = (torch.arange(grid, dtype=torch.float32) + 0.5) / grid * 2 - 1
    ys, xs = torch.meshgrid(centres, centres, indexing="ij")
    return torch.stack([xs.flatten(), ys.flatten()], dim=1)


def patchify(x, p):
    """(B, C, H, W) -> (B, N, C * p * p), row by row."""
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, (H // p) * (W // p), C * p * p)


def unpatchify(tokens, p, grid, channels=3):
    """(B, N, C * p * p) -> (B, C, H, W)."""
    B = tokens.shape[0]
    x = tokens.reshape(B, grid, grid, channels, p, p).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, channels, grid * p, grid * p)


class ImageModel(nn.Module):
    def __init__(self, image_size, patch_size, n_labels, cond_dim, token_dim):
        super().__init__()
        assert image_size % patch_size == 0
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        self.label_embed = nn.Embedding(n_labels, cond_dim)
        nn.init.normal_(self.label_embed.weight, std=0.02)
        self.register_buffer("pos_embed", sincos_pos_embed_2d(token_dim, self.grid)[None], persistent=False)
        self.register_buffer("coords", grid_coords(self.grid)[None], persistent=False)

    def output(self, tokens):
        return unpatchify(tokens, self.patch_size, self.grid)


class ImageDiT(ImageModel):
    def __init__(self, image_size=32, patch_size=2, n_labels=10, hidden=256, depth=6, heads=4):
        super().__init__(image_size, patch_size, n_labels, hidden, hidden)
        self.net = PointDiT(point_dim=3 * patch_size**2, hidden=hidden, depth=depth, heads=heads)

    def forward(self, z, t, y):
        h = self.net.point_embed(patchify(z, self.patch_size)) + self.pos_embed
        c = self.net.t_embed(t) + self.label_embed(y)
        for block in self.net.blocks:
            h = block(h, c)
        return self.output(self.net.final(h, c))


class ImagePTv3(ImageModel):
    def __init__(self, image_size=32, patch_size=2, n_labels=10, attn_patch=64, knn=8):
        net = PointPTv3(point_dim=3 * patch_size**2, patch=attn_patch, k=knn)
        super().__init__(image_size, patch_size, n_labels, net.t_embed.mlp[-1].out_features, net.point_embed.out_features)
        self.net = net

    def forward(self, z, t, y):
        h = self.net.point_embed(patchify(z, self.patch_size)) + self.pos_embed
        c = self.net.t_embed(t) + self.label_embed(y)
        coord = self.coords.expand(z.shape[0], -1, -1)
        return self.output(self.net.final(self.net.backbone(h, coord, c), c))
