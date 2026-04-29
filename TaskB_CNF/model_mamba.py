"""
model_mamba.py
--------------
Bidirectional Mamba vector field for flow matching.
Uses the highly optimized, official mamba_ssm CUDA kernels.
"""

import torch
import torch.nn as nn
from typing import Optional
from model import LogMagFFTEncoder, SinusoidalEncoding, AdaLN

# --- NEW: Import the official hardware-fused Mamba! ---
from mamba_ssm import Mamba


class BidirectionalMambaBlock(nn.Module):
    """
    Bidirectional Mamba using the official CUDA kernels.
    """
    def __init__(
        self,
        d_model: int,
        d_cond:  int,
        d_state: int = 16,
        d_conv:  int = 4,
        expand:  int = 2,
    ):
        super().__init__()
        self.ada_ln  = AdaLN(d_model, d_cond)
        
        # Replace minimal blocks with the official fused blocks
        self.mamba_fwd = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.mamba_bwd = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )

        self.out_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.ada_ln(x, z)
        fwd = self.mamba_fwd(x)
        bwd = self.mamba_bwd(x.flip(1)).flip(1)
        return res + self.out_proj(fwd + bwd)


class BidirectionalMambaVectorField(nn.Module):
    def __init__(
        self,
        d_model:  int = 256,
        d_cond:   int = 256,
        d_ff:     int = 512,   # unused, kept for API compat
        n_layers: int = 8,
        token_dim: int = 4,
        d_time:   int = 256,
        d_state:  int = 16,
        d_conv:   int = 4,
        expand:   int = 2,
    ):
        super().__init__()
        self.token_in  = nn.Linear(token_dim, d_model)
        self.token_out = nn.Linear(d_model, token_dim)
        nn.init.zeros_(self.token_out.weight)
        nn.init.zeros_(self.token_out.bias)

        self.time_enc  = SinusoidalEncoding(d_time)
        self.time_proj = nn.Linear(d_time, d_cond)

        self.cfg_dropout_token = nn.Parameter(torch.randn(1, d_cond) * 0.02)

        self.cond_proj = nn.Sequential(
            nn.Linear(d_cond * 2, d_cond),
            nn.GELU(),
            nn.Linear(d_cond, d_cond),
        )

        self.layers = nn.ModuleList([
            BidirectionalMambaBlock(
                d_model=d_model,
                d_cond=d_cond,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_layers)
        ])

        self.out_norm = nn.LayerNorm(d_model)

    def apply_dropout(self, z: torch.Tensor, rate: float = 0.1) -> torch.Tensor:
        if rate == 0.0 or not self.training:
            return z
        mask = (torch.rand(z.shape[0], 1, device=z.device) > rate)
        return torch.where(mask, z, self.cfg_dropout_token.expand(z.shape[0], -1))

    def forward(self, x_t, t, z=None, mask=None):
        B = x_t.shape[0]
        if z is None:
            z = self.cfg_dropout_token.expand(B, -1)

        # 1. Silence padded inputs
        if mask is not None:
            mask_float = mask.unsqueeze(-1).float()
            x_t = x_t * mask_float

        t_enc = self.time_proj(self.time_enc(t))
        cond  = self.cond_proj(torch.cat([z, t_enc], dim=-1))

        h = self.token_in(x_t)
        for layer in self.layers:
            h = layer(h, cond)
            # 2. Stop echo-chamber bleeding
            if mask is not None:
                h = h * mask_float

        h = self.out_norm(h)
        out = self.token_out(h)
        
        # 3. Force final drift to strictly zero for padding
        if mask is not None:
            out = out * mask_float
            
        return out