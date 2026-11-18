"""A small Diffusion Transformer (DiT) for 2D point clouds.

Pipeline:  points (B, N, 2) -> one token per point (linear layer)
           -> depth x DiTBlock (conditioned on time t via adaLN-Zero)
           -> per-token linear -> output (B, N, 2)

Differences to the image DiT: no patches, and no position embedding. Without
a position embedding, reordering the input points reorders the output in the
same way (permutation equivariance), which suits unordered point sets.
What the output *means* (x, eps or v) is decided in flow.py.
"""
import math

import torch
import torch.nn as nn


def modulate(x, shift, scale):
    # x: (B, N, D); shift/scale: (B, D) -> broadcast over the N tokens
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FourierEmbed(nn.Module):
    """Coordinates -> [x, sin(2^k pi x), cos(2^k pi x)] for k < n_freqs -> linear layer.
    The high frequencies make small position differences easier to see."""

    def __init__(self, point_dim, n_freqs, hidden):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freqs) * math.pi, persistent=False)
        self.proj = nn.Linear(point_dim * (1 + 2 * n_freqs), hidden)

    def forward(self, x):
        angles = x.unsqueeze(-1) * self.freqs  # (B, N, 2, n_freqs)
        return self.proj(torch.cat([x, angles.sin().flatten(-2), angles.cos().flatten(-2)], dim=-1))


class TimestepEmbedder(nn.Module):
    """t (float in [0, 1]) -> sinusoidal features -> MLP -> vector of size hidden."""

    def __init__(self, hidden, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = (t.float() * 1000)[:, None] * freqs[None]  # scale t to [0, 1000] like DiT's integer steps
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero: the time vector c predicts
    shift/scale for each LayerNorm and a gate for each residual branch."""

    def __init__(self, hidden, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden), nn.GELU(approximate="tanh"), nn.Linear(mlp_hidden, hidden)
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))

    def forward(self, x, c):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift1, scale1)
        x = x + gate1.unsqueeze(1) * self.attn(h, h, h, need_weights=False)[0]
        h = modulate(self.norm2(x), shift2, scale2)
        x = x + gate2.unsqueeze(1) * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden, out_dim)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class PointDiT(nn.Module):
    def __init__(self, point_dim=2, hidden=256, depth=6, heads=4, mlp_ratio=4.0, fourier=0, n_classes=0):
        super().__init__()
        self.point_embed = FourierEmbed(point_dim, fourier, hidden) if fourier else nn.Linear(point_dim, hidden)
        self.t_embed = TimestepEmbedder(hidden)
        # one extra label: the "no class" token used by label dropout
        self.label_embed = nn.Embedding(n_classes + 1, hidden) if n_classes else None
        self.blocks = nn.ModuleList([DiTBlock(hidden, heads, mlp_ratio) for _ in range(depth)])
        self.final = FinalLayer(hidden, point_dim)
        self._init_weights()

    def _init_weights(self):
        # same scheme as the official DiT code
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for layer in (self.t_embed.mlp[0], self.t_embed.mlp[2]):
            nn.init.normal_(layer.weight, std=0.02)
        if self.label_embed is not None:
            nn.init.normal_(self.label_embed.weight, std=0.02)
        # "Zero" part of adaLN-Zero: every block starts as the identity,
        # and the final layer starts by outputting 0
        for block in self.blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)
        nn.init.zeros_(self.final.adaLN[-1].weight)
        nn.init.zeros_(self.final.adaLN[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    def forward(self, z, t, y=None):
        """z: (B, N, 2) noisy points, t: (B,) time in [0, 1], y: (B,) optional labels. Returns (B, N, 2)."""
        x = self.point_embed(z)
        c = self.t_embed(t)
        if y is not None:
            c = c + self.label_embed(y)
        for block in self.blocks:
            x = block(x, c)
        return self.final(x, c)
