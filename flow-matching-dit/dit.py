"""A small Diffusion Transformer (DiT, Peebles & Xie 2023) for images.

Pipeline:  image -> patches -> tokens (+ fixed 2D sin-cos positions)
           -> N x DiTBlock (conditioned on time t via adaLN-Zero)
           -> per-token linear -> patches -> image (same shape as input)

The network is unconditional (no class labels). It only gets the noisy image z_t
and the time t. What the output *means* (x, eps or v) is decided in flow.py.
"""
import math

import torch
import torch.nn as nn


def modulate(x, shift, scale):
    # x: (B, N, D); shift/scale: (B, D) -> broadcast over the N tokens
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def sincos_pos_embed_2d(dim, grid_size):
    """Fixed 2D sin-cos position embedding, shape (grid_size**2, dim)."""
    assert dim % 4 == 0
    ys, xs = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing="ij")

    def embed_1d(d, pos):
        omega = 1.0 / 10000 ** (torch.arange(d // 2, dtype=torch.float64) / (d / 2))
        out = pos.flatten().double()[:, None] * omega[None]
        return torch.cat([out.sin(), out.cos()], dim=1)

    # half the channels encode the row, half the column
    return torch.cat([embed_1d(dim // 2, ys), embed_1d(dim // 2, xs)], dim=1).float()


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
    def __init__(self, hidden, patch_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden, patch_size * patch_size * out_channels)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class DiT(nn.Module):
    def __init__(self, image_size=32, patch_size=2, in_channels=3, hidden=256, depth=6, heads=4, mlp_ratio=4.0):
        super().__init__()
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = in_channels
        self.grid = image_size // patch_size

        self.patch_embed = nn.Conv2d(in_channels, hidden, kernel_size=patch_size, stride=patch_size)
        self.register_buffer("pos_embed", sincos_pos_embed_2d(hidden, self.grid)[None])  # (1, N, D), not trained
        self.t_embed = TimestepEmbedder(hidden)
        self.blocks = nn.ModuleList([DiTBlock(hidden, heads, mlp_ratio) for _ in range(depth)])
        self.final = FinalLayer(hidden, patch_size, in_channels)
        self._init_weights()

    def _init_weights(self):
        # same scheme as the official DiT code
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        w = self.patch_embed.weight
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.bias)
        for layer in (self.t_embed.mlp[0], self.t_embed.mlp[2]):
            nn.init.normal_(layer.weight, std=0.02)
        # "Zero" part of adaLN-Zero: every block starts as the identity,
        # and the final layer starts by outputting 0
        for block in self.blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)
        nn.init.zeros_(self.final.adaLN[-1].weight)
        nn.init.zeros_(self.final.adaLN[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    def unpatchify(self, x):
        # (B, N, p*p*C) -> (B, C, H, W)
        B, p, c, g = x.shape[0], self.patch_size, self.channels, self.grid
        x = x.reshape(B, g, g, p, p, c)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(B, c, g * p, g * p)

    def forward(self, z, t):
        """z: (B, C, H, W) noisy image, t: (B,) time in [0, 1]. Returns (B, C, H, W)."""
        x = self.patch_embed(z).flatten(2).transpose(1, 2)  # (B, N, D), row-major patch order
        x = x + self.pos_embed
        c = self.t_embed(t)
        for block in self.blocks:
            x = block(x, c)
        return self.unpatchify(self.final(x, c))
