"""
mamba_minimal.py
----------------
Pure PyTorch implementation of a Mamba (SSM) block.
No mamba_ssm, causal_conv1d, or CUDA extensions required.
Runs on any device: CPU, MPS, CUDA.

Based on:
  Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
  https://arxiv.org/abs/2312.00752

This implements the core selective SSM recurrence in pure PyTorch.
It is slower than the Triton/CUDA kernels in mamba_ssm, but correct
and dependency-free. For sequences of length ~30000 on an A100, the
recurrent scan will be the bottleneck — consider using a chunked
parallel scan if speed is critical (easy to add later).

Usage (drop-in for BidirectionalMambaBlock in model_mamba.py):
    from mamba_minimal import MambaBlock, BidirectionalMambaBlock
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MambaBlock(nn.Module):
    """
    Single-direction Mamba block in pure PyTorch.

    Args:
        d_model : token dimension
        d_state  : SSM state size (N in the paper)
        d_conv   : local convolution width
        expand   : inner dimension expansion factor (d_inner = expand * d_model)
    """

    def __init__(
        self,
        d_model:       int,
        d_state:       int   = 16,
        d_conv:        int   = 4,
        expand:        int   = 2,
        dt_min:        float = 0.001,
        dt_max:        float = 0.1,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv  = d_conv
        self.d_inner = expand * d_model
        # Match official rank: ceil(d_model / 16)
        self.dt_rank = math.ceil(d_model / 16)

        # Input projection: d_model -> 2 * d_inner  (z and x branches)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Causal depthwise conv on the x branch
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        # SSM input-dependent projections
        # x_proj: d_inner -> dt_rank + 2*d_state  (matches official)
        self.x_proj  = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        # dt_proj: dt_rank -> d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialise dt_proj.weight (uniform, matches official "random" init)
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Initialise dt_proj.bias so softplus(bias) is in [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))   # inverse softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A: (d_inner, d_state) — kept in float32, negative by construction
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection (zero-init for stable residual training)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective state-space model forward pass (Parallel Scan).
        x: (B, L, d_inner)
        returns: (B, L, d_inner)
        """
        B, L, D = x.shape
        N = self.d_state

        # Compute input-dependent dt, B, C
        xBCdt  = self.x_proj(x)                          # (B, L, dt_rank + 2N)
        dt     = xBCdt[..., :self.dt_rank]               # (B, L, dt_rank)
        B_ssm  = xBCdt[..., self.dt_rank:self.dt_rank+N] # (B, L, N)
        C_ssm  = xBCdt[..., self.dt_rank+N:]             # (B, L, N)

        # dt_proj: dt_rank -> d_inner, then softplus
        dt = F.softplus(self.dt_proj(dt))    # (B, L, D)

        # Discretise A: A_bar = exp(dt * A)
        A = -torch.exp(self.A_log.float())   # (D, N)
        
        # dA: (B, L, D, N)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # dB: (B, L, D, N)
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)

        # Calculate initial step values: dB * x_t
        # x is (B, L, D) -> unsqueeze to (B, L, D, 1) for broadcasting over N
        h = dB * x.unsqueeze(-1)  # (B, L, D, N)

        # Parallel prefix scan (Hillis-Steele) in O(log L) steps
        steps = math.ceil(math.log2(L))
        for i in range(steps):
            shift = 2 ** i
            
            # Efficiently shift tensors using F.pad. 
            # Padding dims are evaluated from back to front: (N_left, N_right, D_left, D_right, L_left, L_right)
            dA_shifted = F.pad(dA[:, :-shift], (0, 0, 0, 0, shift, 0), value=1.0)
            h_shifted  = F.pad(h[:, :-shift],  (0, 0, 0, 0, shift, 0), value=0.0)

            # Update associative state
            h = h + dA * h_shifted
            dA = dA * dA_shifted

        # h now contains the hidden state at each time step t
        # Vectorized computation of y
        # C_ssm is (B, L, N) -> unsqueeze to (B, L, 1, N)
        y = (h * C_ssm.unsqueeze(2)).sum(dim=-1)  # (B, L, D)

        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_model)
        returns: (B, L, d_model)
        """
        B, L, _ = x.shape

        xz = self.in_proj(x)                 # (B, L, 2*d_inner)
        x_, z = xz.chunk(2, dim=-1)          # each (B, L, d_inner)

        # Causal conv (transpose for Conv1d: B, C, L)
        x_ = x_.transpose(1, 2)             # (B, d_inner, L)
        x_ = self.conv1d(x_)[..., :L]       # crop padding
        x_ = x_.transpose(1, 2)             # (B, L, d_inner)
        x_ = F.silu(x_)

        y = self.ssm(x_)
        y = y * F.silu(z)
        return self.out_proj(y)              # (B, L, d_model)


class BidirectionalMambaBlock(nn.Module):
    """
    Bidirectional Mamba: forward scan + backward scan, summed.
    Ada-LN conditioned on z.

    Drop-in replacement for BidirectionalMambaBlock in model_mamba.py.
    No mamba_ssm dependency.
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
        from model import AdaLN   # reuse AdaLN from model.py

        self.ada_ln  = AdaLN(d_model, d_cond)
        self.mamba_fwd = MambaBlock(d_model, d_state, d_conv, expand)
        self.mamba_bwd = MambaBlock(d_model, d_state, d_conv, expand)

        self.out_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.ada_ln(x, z)
        fwd = self.mamba_fwd(x)
        bwd = self.mamba_bwd(x.flip(1)).flip(1)
        return res + self.out_proj(fwd + bwd)