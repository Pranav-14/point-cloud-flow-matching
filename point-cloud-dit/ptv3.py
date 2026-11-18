"""A 2D Point Transformer V3 (Wu et al. 2024) for flow matching on point clouds.

Pipeline:  points (B, N, 2) -> embedding -> U-Net of PTv3 blocks -> linear -> (B, N, 2)

PTv3 ideas kept:
  - serialization: points are sorted along space-filling curves
  - patch attention: attention only inside groups of consecutive sorted points
  - shift / shuffle order: blocks use different curves, shuffled while training
  - a local position encoding (xCPE) before attention
  - a U-Net with pooling, unpooling and skip connections

Changes for this project:
  - 2D instead of 3D. The 3D sparse convolution of xCPE is replaced by a small
    convolution over each point's k nearest neighbours, whose per-channel weights
    are computed from the relative neighbour positions.
  - Pooling merges consecutive pairs along the Z-order curve, so every stage has a
    fixed number of points. N must be divisible by 2 ** (stages - 1).
  - Time conditioning: adaLN-Zero in every block and in the output layer, as in the DiT.
  - The curve grid covers [-extent, extent], because noisy points leave [-1, 1].
"""
import torch
import torch.nn as nn

from dit import FourierEmbed, TimestepEmbedder, modulate

ORDERS = ("z", "z-trans", "hilbert", "hilbert-trans")


def quantize(coord, bits=16, extent=4.0):
    n = 1 << bits
    return ((coord + extent) / (2 * extent) * n).floor().long().clamp(0, n - 1)


def z_code(x, y, bits):
    """Interleave the bits of x and y."""
    code = torch.zeros_like(x)
    for i in range(bits):
        code |= ((x >> i) & 1) << (2 * i + 1)
        code |= ((y >> i) & 1) << (2 * i)
    return code


def hilbert_code(x, y, bits):
    """Position of the integer cell (x, y) along a Hilbert curve."""
    n = 1 << bits
    code = torch.zeros_like(x)
    for i in reversed(range(bits)):
        s = 1 << i
        rx = ((x & s) > 0).long()
        ry = ((y & s) > 0).long()
        code += s * s * ((3 * rx) ^ ry)
        # rotate / flip the remaining lower bits into this quadrant's orientation
        flip = (ry == 0) & (rx == 1)
        x = torch.where(flip, n - 1 - x, x)
        y = torch.where(flip, n - 1 - y, y)
        swap = ry == 0
        x, y = torch.where(swap, y, x), torch.where(swap, x, y)
    return code


def serialize(coord, order, bits=16, extent=4.0):
    """coord (B, N, 2) -> perm, inv (B, N). coord[perm] follows the curve, inv undoes the sort."""
    grid = quantize(coord, bits, extent)
    x, y = grid[..., 0], grid[..., 1]
    if order.endswith("trans"):
        x, y = y, x
    code = z_code(x, y, bits) if order.startswith("z") else hilbert_code(x, y, bits)
    perm = code.argsort(dim=1, stable=True)
    return perm, perm.argsort(dim=1)


def take(values, index):
    """values (B, N, ...) and index (B, ...) -> values at those per-cloud indices."""
    batch = torch.arange(values.shape[0], device=values.device)
    batch = batch.view(-1, *[1] * (index.dim() - 1))
    return values[batch, index]


class LocalEncoding(nn.Module):
    """xCPE for points: a depthwise convolution over the k nearest neighbours.
    Each channel's neighbour weight is a learned function of the relative position."""

    def __init__(self, dim):
        super().__init__()
        self.kernel = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, coord, knn):
        relative = take(coord, knn) - coord.unsqueeze(2)  # (B, N, k, 2)
        # divide by the cloud's mean neighbour distance, so the kernel works at every noise level
        scale = relative.norm(dim=-1).mean(dim=(1, 2)).clamp(min=1e-6).view(-1, 1, 1, 1)
        weights = self.kernel(relative / scale)  # (B, N, k, D)
        return self.norm(self.proj((weights * take(h, knn)).mean(dim=2)))


class PatchAttention(nn.Module):
    """Sort along a curve, split into patches, attend inside each patch, undo the sort."""

    def __init__(self, dim, heads, patch):
        super().__init__()
        self.patch = patch
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, h, perm, inv):
        B, N, D = h.shape
        P = min(self.patch, N)
        full = N // P * P
        rest = N - full
        tokens = take(h, perm)
        # leftover points form one last patch that overlaps the patch before it
        x = torch.cat([tokens[:, :full], tokens[:, -P:]], dim=1) if rest else tokens
        x = x.reshape(-1, P, D)
        x = self.attn(x, x, x, need_weights=False)[0].reshape(B, -1, D)
        if rest:
            x = torch.cat([x[:, :full], x[:, -rest:]], dim=1)
        return take(x, inv)


class PTv3Block(nn.Module):
    """Local encoding -> attention -> MLP, each one a residual branch gated by adaLN-Zero."""

    def __init__(self, dim, heads, patch, cond_dim, mlp_ratio=4.0):
        super().__init__()
        self.local = LocalEncoding(dim)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = PatchAttention(dim, heads, patch)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 7 * dim))

    def forward(self, h, stage, order, c):
        gate0, shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c).chunk(7, dim=-1)
        h = h + gate0.unsqueeze(1) * self.local(h, stage["coord"], stage["knn"])
        x = modulate(self.norm1(h), shift1, scale1)
        h = h + gate1.unsqueeze(1) * self.attn(x, stage["perm"][order], stage["inv"][order])
        x = modulate(self.norm2(h), shift2, scale2)
        return h + gate2.unsqueeze(1) * self.mlp(x)


class Pool(nn.Module):
    """Merge consecutive point pairs along the first curve (max of features, mean of positions)."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_in, dim_out), nn.LayerNorm(dim_out), nn.GELU())

    def forward(self, h, stage):
        B, N, _ = h.shape
        perm, inv = stage["perm"][0], stage["inv"][0]
        h = take(self.net(h), perm).reshape(B, N // 2, 2, -1).max(dim=2).values
        coord = take(stage["coord"], perm).reshape(B, N // 2, 2, 2).mean(dim=2)
        return h, coord, inv // 2  # parent[i] = coarse token that fine point i was merged into


class Unpool(nn.Module):
    """Copy each coarse feature to its two fine points and add the encoder's skip connection.
    The skip has no LayerNorm: normalising each token would remove how far a noisy point
    is from the origin, which eps- and v-prediction need. The coarse branch starts at zero,
    so an untrained U-Net passes every point straight through, like the DiT."""

    def __init__(self, dim_coarse, dim_skip, dim_out):
        super().__init__()
        self.coarse = nn.Sequential(nn.Linear(dim_coarse, dim_out), nn.GELU(), nn.Linear(dim_out, dim_out))
        self.skip = nn.Linear(dim_skip, dim_out)

    def forward(self, h, skip, parent):
        return take(self.coarse(h), parent) + self.skip(skip)


class FinalLayer(nn.Module):
    def __init__(self, dim, cond_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_dim)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * dim))

    def forward(self, h, c):
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(h), shift, scale))


class PointPTv3(nn.Module):
    def __init__(
        self,
        point_dim=2,
        fourier=0,
        cond_dim=256,
        enc_dims=(64, 128, 192, 256),
        enc_depths=(2, 2, 2, 2),
        enc_heads=(2, 4, 8, 8),
        dec_dims=(64, 128, 192),
        dec_depths=(1, 1, 1),
        dec_heads=(2, 4, 8),
        patch=64,
        k=8,
        orders=ORDERS,
        shuffle_orders=True,
        bits=16,
        extent=4.0,
        n_classes=0,
    ):
        super().__init__()
        # one extra label: the "no class" token used by label dropout
        self.label_embed = nn.Embedding(n_classes + 1, cond_dim) if n_classes else None
        self.stages = len(enc_dims)
        self.k, self.orders, self.shuffle_orders = k, tuple(orders), shuffle_orders
        self.bits, self.extent = bits, extent

        embed_dim = enc_dims[0]
        self.point_embed = FourierEmbed(point_dim, fourier, embed_dim) if fourier else nn.Linear(point_dim, embed_dim)
        self.t_embed = TimestepEmbedder(cond_dim)

        def blocks(dim, heads, depth):
            return nn.ModuleList([PTv3Block(dim, heads, patch, cond_dim) for _ in range(depth)])

        self.encoder = nn.ModuleList([blocks(*spec) for spec in zip(enc_dims, enc_heads, enc_depths)])
        self.pools = nn.ModuleList([Pool(a, b) for a, b in zip(enc_dims[:-1], enc_dims[1:])])

        # decoder stage s turns stage s + 1 back into stage s
        coarse_dims = list(dec_dims[1:]) + [enc_dims[-1]]
        self.unpools = nn.ModuleList([Unpool(c, e, d) for c, e, d in zip(coarse_dims, enc_dims[:-1], dec_dims)])
        self.decoder = nn.ModuleList([blocks(*spec) for spec in zip(dec_dims, dec_heads, dec_depths)])

        self.final = FinalLayer(dec_dims[0], cond_dim, point_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for layer in (self.t_embed.mlp[0], self.t_embed.mlp[2]):
            nn.init.normal_(layer.weight, std=0.02)
        if self.label_embed is not None:
            nn.init.normal_(self.label_embed.weight, std=0.02)
        # adaLN-Zero: every block starts as the identity and the output starts at 0
        for stage in list(self.encoder) + list(self.decoder):
            for block in stage:
                nn.init.zeros_(block.adaLN[-1].weight)
                nn.init.zeros_(block.adaLN[-1].bias)
        for unpool in self.unpools:
            nn.init.zeros_(unpool.coarse[-1].weight)
            nn.init.zeros_(unpool.coarse[-1].bias)
        nn.init.zeros_(self.final.adaLN[-1].weight)
        nn.init.zeros_(self.final.adaLN[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    @torch.no_grad()
    def _prepare_stage(self, coord):
        sorted_orders = [serialize(coord, order, self.bits, self.extent) for order in self.orders]
        k = min(self.k, coord.shape[1])
        knn = torch.cdist(coord, coord).topk(k, dim=-1, largest=False).indices
        return {
            "coord": coord,
            "knn": knn,
            "perm": [perm for perm, _ in sorted_orders],
            "inv": [inv for _, inv in sorted_orders],
        }

    def _block_orders(self, n_blocks):
        """Shift order: block b uses curve b mod K. Shuffle order: permute the curves while training."""
        K = len(self.orders)
        base = torch.randperm(K).tolist() if self.training and self.shuffle_orders else list(range(K))
        return [base[b % K] for b in range(n_blocks)]

    def _run(self, blocks, h, stage, c):
        for block, order in zip(blocks, self._block_orders(len(blocks))):
            h = block(h, stage, order, c)
        return h

    def forward(self, z, t, y=None):
        """z: (B, N, 2) noisy points, t: (B,) time in [0, 1], y: (B,) optional labels. Returns (B, N, 2)."""
        c = self.t_embed(t)
        if y is not None:
            c = c + self.label_embed(y)
        return self.final(self.backbone(self.point_embed(z), z, c), c)

    def backbone(self, h, coord, c):
        """Token features h (B, N, D) at positions coord (B, N, 2) -> features of the first stage."""
        if h.shape[1] % 2 ** (self.stages - 1):
            raise ValueError(f"number of points must be divisible by {2 ** (self.stages - 1)}")
        prepared, skips, parents = [], [], []

        for s in range(self.stages):
            stage = self._prepare_stage(coord)
            prepared.append(stage)
            h = self._run(self.encoder[s], h, stage, c)
            if s < self.stages - 1:
                skips.append(h)
                h, coord, parent = self.pools[s](h, stage)
                parents.append(parent)

        for s in reversed(range(self.stages - 1)):
            h = self.unpools[s](h, skips[s], parents[s])
            h = self._run(self.decoder[s], h, prepared[s], c)
        return h
