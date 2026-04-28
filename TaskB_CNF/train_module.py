"""
train_module.py
---------------
Lightning module for Task B: modal parameter estimation via flow matching.

Adapts the SurgeFlowMatchingModule from Hayes et al. for our setting:
  - Encoder    : LogMagFFTEncoder (1D CNN on log-magnitude FFT)
  - VectorField: DeepSetsVectorField (permutation-equivariant)
  - Flow       : Rectified flow (linear interpolation between noise and modes)
  - CFG        : Classifier-free guidance dropout on the IR conditioning

Batch dict keys (from dataset.py):
  'log_mag_fft' : (B, N_freq)
  'modes'       : (B, M, 4)   — normalised (f0, sigma, gain, exists)
  'n_modes'     : (B,)        — number of real modes per sample
  'noise'       : (B, M, 4)   — pre-sampled Gaussian noise
"""

from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from model import LogMagFFTEncoder, DeepSetsVectorField


# ─── CFG helpers ──────────────────────────────────────────────────────────────

def call_with_cfg(
    f,
    x_t:          torch.Tensor,
    t:            torch.Tensor,
    conditioning: Optional[torch.Tensor],
    cfg_strength: float,
    mask:         Optional[torch.Tensor] = None,
) -> torch.Tensor:
    y_c = f(x_t, t, conditioning, mask)
    y_u = f(x_t, t, None,         mask)
    return (1 - cfg_strength) * y_u + cfg_strength * y_c


def rk4_step(
    f,
    x:            torch.Tensor,
    t:            torch.Tensor,
    dt:           float,
    conditioning: Optional[torch.Tensor],
    cfg_strength: float,
    mask:         Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _f = partial(call_with_cfg, f,
                 conditioning=conditioning,
                 cfg_strength=cfg_strength,
                 mask=mask)
    k1 = _f(x,              t)
    k2 = _f(x + dt*k1/2,   t + dt/2)
    k3 = _f(x + dt*k2/2,   t + dt/2)
    k4 = _f(x + dt*k3,     t + dt)
    return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


# ─── Lightning module ─────────────────────────────────────────────────────────

class ModalFlowMatchingModule(LightningModule):
    def __init__(
        self,
        # model hparams
        n_fft:                  int   = 8192,
        d_cond:                 int   = 128,
        encoder_base_channels:  int   = 24,
        d_model:                int   = 128,
        d_ff:                   int   = 256,
        n_layers:               int   = 6,
        # training hparams
        lr:                     float = 1e-4,
        warmup_steps:           int   = 2000,
        cfg_dropout_rate:       float = 0.1,
        # inference hparams
        val_steps:              int   = 50,
        val_cfg:                float = 2.0,
        test_steps:             int   = 100,
        test_cfg:               float = 2.0,
        # misc
        compile:                bool  = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = LogMagFFTEncoder(
            n_fft=n_fft,
            d_cond=d_cond,
            base_channels=encoder_base_channels,
        )

        self.vector_field = DeepSetsVectorField(
            d_model=d_model,
            d_cond=d_cond,
            d_ff=d_ff,
            n_layers=n_layers,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_mask(self, n_modes: torch.Tensor, max_m: int) -> torch.Tensor:
        """
        Build a boolean mask (B, M) where True = real mode.
        n_modes: (B,) int tensor
        """
        B = n_modes.shape[0]
        idx = torch.arange(max_m, device=n_modes.device).unsqueeze(0)  # (1, M)
        return idx < n_modes.unsqueeze(1)                               # (B, M)

    def _encode(self, batch: dict) -> torch.Tensor:
        return self.encoder(batch['log_mag_fft'])   # (B, d_cond)

    # ── flow matching ─────────────────────────────────────────────────────────

    def _train_step(self, batch: dict):
        modes  = batch['modes']    # (B, M, 4)
        noise  = batch['noise']    # (B, M, 4)
        n_modes = batch['n_modes'] # (B,)

        B, M, _ = modes.shape
        mask = self._make_mask(n_modes, M)   # (B, M)

        # Encode IR
        z = self._encode(batch)                                     # (B, d_cond)
        z = self.vector_field.apply_dropout(z, self.hparams.cfg_dropout_rate)

        with torch.no_grad():
            t  = torch.rand(B, 1, device=modes.device)             # (B, 1)
            x0 = noise
            x1 = modes
            # Rectified flow path: x_t = (1-t)*x0 + t*x1
            t_exp = t.unsqueeze(-1)                                 # (B, 1, 1)
            x_t   = (1 - t_exp) * x0 + t_exp * x1                 # (B, M, 4)
            target = x1 - x0                                        # (B, M, 4)

        prediction = self.vector_field(x_t, t, z, mask=mask)       # (B, M, 4)

        # Loss only over real modes (not padding)
        loss_all = (prediction - target).square()                   # (B, M, 4)
        mask_exp = mask.unsqueeze(-1).float()                       # (B, M, 1)
        loss = (loss_all * mask_exp).sum() / (mask_exp.sum() * 4 + 1e-8)

        return loss

    def training_step(self, batch: dict, batch_idx: int):
        loss = self._train_step(batch)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ── sampling ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _sample(
        self,
        z:     Optional[torch.Tensor],   # (B, d_cond) or None
        noise: torch.Tensor,             # (B, M, 4)
        steps: int,
        cfg:   float,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sample = noise
        t      = torch.zeros(noise.shape[0], 1, device=noise.device)
        dt     = 1.0 / steps

        for _ in range(steps):
            sample = rk4_step(
                self.vector_field,
                sample, t, dt,
                conditioning=z,
                cfg_strength=cfg,
                mask=mask,
            )
            t = t + dt

        return sample

    def validation_step(self, batch: dict, batch_idx: int):
        modes   = batch['modes']
        n_modes = batch['n_modes']
        B, M, _ = modes.shape
        mask    = self._make_mask(n_modes, M)

        z = self._encode(batch)
        noise = torch.randn_like(modes)
        pred  = self._sample(z, noise, self.hparams.val_steps, self.hparams.val_cfg, mask)

        # MSE on real modes only
        mask_exp = mask.unsqueeze(-1).float()
        mse = ((pred - modes).square() * mask_exp).sum() / (mask_exp.sum() * 4 + 1e-8)
        self.log('val/mse', mse, on_epoch=True, prog_bar=True)

        # Per-parameter MSE (useful for debugging)
        for i, name in enumerate(['f0', 'sigma', 'gain', 'exists']):
            param_mse = ((pred[..., i] - modes[..., i]).square() * mask[..., None].squeeze(-1).float()).sum() \
                        / (mask.float().sum() + 1e-8)
            self.log(f'val/mse_{name}', param_mse, on_epoch=True)

        return mse

    def test_step(self, batch: dict, batch_idx: int):
        modes   = batch['modes']
        n_modes = batch['n_modes']
        B, M, _ = modes.shape
        mask    = self._make_mask(n_modes, M)

        z = self._encode(batch)
        noise = torch.randn_like(modes)
        pred  = self._sample(z, noise, self.hparams.test_steps, self.hparams.test_cfg, mask)

        mask_exp = mask.unsqueeze(-1).float()
        mse = ((pred - modes).square() * mask_exp).sum() / (mask_exp.sum() * 4 + 1e-8)
        self.log('test/mse', mse)
        return mse

    # ── optimizer ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-8, end_factor=1.0,
            total_iters=self.hparams.warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=100_000, eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[warmup, cosine],
            milestones=[self.hparams.warmup_steps],
        )

        return {
            'optimizer': opt,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, 2.0)
        self.log_dict(norms, on_step=True, on_epoch=False)

    def setup(self, stage: str):
        if self.hparams.compile:
            self.encoder      = torch.compile(self.encoder)
            self.vector_field = torch.compile(self.vector_field)