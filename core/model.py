"""
model.py
--------
DeepSets-based vector field for flow matching over modal parameter sets.

Architecture:
  Encoder : 1D CNN on log-magnitude FFT -> conditioning vector z
  VectorField : DeepSets
      - pointwise MLP per mode token  (equivariant by construction)
      - mean pool -> global context   (invariant aggregation)
      - inject global context + z + t via Ada-LN
      - pointwise MLP out             (equivariant)

The vector field predicts the drift u_theta(x_t, t | z) for the rectified
flow path x_t = (1-t)*x0 + t*x1, where x0 ~ N(0,I) and x1 = modes.

Token dimension: 4  (f0_norm, sigma_norm, gain_norm, exists)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─── Encoder ──────────────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    """1D residual block with BatchNorm."""
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class LogMagFFTEncoder(nn.Module):
    """
    1D CNN encoder that maps a log-magnitude FFT spectrum to a
    conditioning vector.

    Input  : (B, N_fft) — one-sided log-magnitude spectrum
    Output : (B, d_cond)

    Architecture mirrors Hayes's k-osc CNN:
      4 residual blocks, each followed by strided conv that halves
      the sequence and doubles channels. Final flatten + linear.
    """

    def __init__(self, n_fft: int = 8192, d_cond: int = 128, base_channels: int = 24):
        super().__init__()
        n_freq = n_fft // 2 + 1  # one-sided

        # Initial projection
        self.input_proj = nn.Conv1d(1, base_channels, kernel_size=7, padding=3)

        layers = []
        ch = base_channels
        for _ in range(4):
            layers.append(ResBlock1D(ch))
            layers.append(nn.Conv1d(ch, ch * 2, kernel_size=4, stride=3, padding=1))
            ch *= 2
        self.blocks = nn.Sequential(*layers)

        # Compute flattened size after strided convs
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_freq)
            dummy = self.input_proj(dummy)
            dummy = self.blocks(dummy)
            flat_size = dummy.shape[1] * dummy.shape[2]

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, d_cond * 2),
            nn.GELU(),
            nn.Linear(d_cond * 2, d_cond),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N_freq)
        x = x.unsqueeze(1)          # (B, 1, N_freq)
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.head(x)
        return x                     # (B, d_cond)


# ─── Ada-LN ───────────────────────────────────────────────────────────────────

class AdaLN(nn.Module):
    """Adaptive layer norm: shift + scale conditioned on z."""
    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.norm  = nn.LayerNorm(d_model)
        self.shift = nn.Linear(d_cond, d_model)
        self.scale = nn.Linear(d_cond, d_model)
        nn.init.zeros_(self.shift.weight); nn.init.zeros_(self.shift.bias)
        nn.init.ones_(self.scale.weight);  nn.init.zeros_(self.scale.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # x: (B, M, d_model)   z: (B, d_cond)
        x = self.norm(x)
        shift = self.shift(z).unsqueeze(1)   # (B, 1, d_model)
        scale = self.scale(z).unsqueeze(1)
        return x * scale + shift


# ─── Sinusoidal time encoding ─────────────────────────────────────────────────

class SinusoidalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        half = d_model // 2
        k = torch.arange(half)
        basis = 1.0 / torch.pow(10000.0, k / half)
        self.register_buffer('basis', basis)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B, 1)
        t = t * self.basis.unsqueeze(0)       # (B, half)
        return torch.cat([t.sin(), t.cos()], dim=-1)  # (B, d_model)


# ─── DeepSets block ───────────────────────────────────────────────────────────

class DeepSetsBlock(nn.Module):
    """
    One DeepSets layer:
      1. Ada-LN normalise using conditioning z
      2. Pointwise MLP (equivariant)
      3. Mean pool over M -> global context g
      4. Broadcast g back, concat with token, project
      5. Residual

    This is strictly permutation-equivariant.
    """

    def __init__(self, d_model: int, d_cond: int, d_ff: int):
        super().__init__()
        self.ada_ln = AdaLN(d_model, d_cond)

        # Pointwise MLP
        self.pointwise = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        nn.init.zeros_(self.pointwise[-1].weight)
        nn.init.zeros_(self.pointwise[-1].bias)

        # After pooling: project (d_model + d_model) -> d_model
        self.pool_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        nn.init.zeros_(self.pool_proj[-1].weight)
        nn.init.zeros_(self.pool_proj[-1].bias)

    def forward(
        self,
        x: torch.Tensor,        # (B, M, d_model)
        z: torch.Tensor,        # (B, d_cond)
        mask: Optional[torch.Tensor] = None,  # (B, M) bool, True = real mode
    ) -> torch.Tensor:
        res = x
        x = self.ada_ln(x, z)

        # Pointwise path
        pw = self.pointwise(x)     # (B, M, d_model)

        # Global mean pool (masked to ignore padding)
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()        # (B, M, 1)
            g = (x * mask_f).sum(1) / mask_f.sum(1)   # (B, d_model)
        else:
            g = x.mean(dim=1)                          # (B, d_model)

        g = g.unsqueeze(1).expand_as(x)               # (B, M, d_model)
        combined = torch.cat([pw, g], dim=-1)          # (B, M, 2*d_model)
        out = self.pool_proj(combined)                 # (B, M, d_model)

        return res + out


# ─── Full DeepSets vector field ───────────────────────────────────────────────

class DeepSetsVectorField(nn.Module):
    """
    Flow matching vector field u_theta(x_t, t | z).

    Input:
      x_t : (B, M, 4)   — noisy mode parameters at time t
      t   : (B, 1)      — flow time in [0, 1]
      z   : (B, d_cond) — IR conditioning (None for CFG dropout)

    Output:
      drift : (B, M, 4) — predicted vector field
    """

    def __init__(
        self,
        d_model:    int = 128,
        d_cond:     int = 128,
        d_ff:       int = 256,
        n_layers:   int = 6,
        token_dim:  int = 4,       # (f0, sigma, gain, exists)
        d_time:     int = 128,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.d_model   = d_model

        # Token in/out projection (simple linear, no PARAM2TOK complexity for now)
        self.token_in  = nn.Linear(token_dim, d_model)
        self.token_out = nn.Linear(d_model, token_dim)
        nn.init.zeros_(self.token_out.weight)
        nn.init.zeros_(self.token_out.bias)

        # Time encoding: sinusoidal -> linear -> d_cond
        self.time_enc  = SinusoidalEncoding(d_time)
        self.time_proj = nn.Linear(d_time, d_cond)

        # CFG dropout token
        self.cfg_dropout_token = nn.Parameter(torch.randn(1, d_cond) * 0.02)

        # Conditioning fusion: z + t -> d_cond
        self.cond_proj = nn.Sequential(
            nn.Linear(d_cond * 2, d_cond),
            nn.GELU(),
            nn.Linear(d_cond, d_cond),
        )

        # DeepSets layers
        self.layers = nn.ModuleList([
            DeepSetsBlock(d_model, d_cond, d_ff)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.out_norm = nn.LayerNorm(d_model)

    def apply_dropout(self, z: torch.Tensor, rate: float = 0.1) -> torch.Tensor:
        """CFG conditioning dropout."""
        if rate == 0.0 or not self.training:
            return z
        mask = (torch.rand(z.shape[0], 1, device=z.device) > rate)
        dropout_z = self.cfg_dropout_token.expand(z.shape[0], -1)
        return torch.where(mask, z, dropout_z)

    def forward(
        self,
        x_t: torch.Tensor,                    # (B, M, 4)
        t:   torch.Tensor,                    # (B, 1)
        z:   Optional[torch.Tensor] = None,   # (B, d_cond) or None
        mask: Optional[torch.Tensor] = None,  # (B, M) bool
    ) -> torch.Tensor:
        B, M, _ = x_t.shape

        if z is None:
            z = self.cfg_dropout_token.expand(B, -1)

        # Encode time and fuse with conditioning
        t_enc = self.time_proj(self.time_enc(t))   # (B, d_cond)
        cond  = self.cond_proj(torch.cat([z, t_enc], dim=-1))  # (B, d_cond)

        # Project tokens up
        h = self.token_in(x_t)                    # (B, M, d_model)

        # DeepSets layers
        for layer in self.layers:
            h = layer(h, cond, mask=mask)

        h = self.out_norm(h)
        drift = self.token_out(h)                 # (B, M, 4)
        return drift