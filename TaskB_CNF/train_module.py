"""
train_module.py
---------------
Lightning module for Task B — works with both DeepSets and Mamba backends.

Pass --model deepsets  to use model.py (runs on MPS, for local testing)
Pass --model mamba     to use model_mamba.py (requires CUDA, for Apocrita)
"""

from functools import partial
from typing import Optional

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from model import LogMagFFTEncoder


def call_with_cfg(f, x_t, t, conditioning, cfg_strength, mask=None):
    y_c = f(x_t, t, conditioning, mask)
    y_u = f(x_t, t, None,         mask)
    return (1 - cfg_strength) * y_u + cfg_strength * y_c


def rk4_step(f, x, t, dt, conditioning, cfg_strength, mask=None):
    _f = partial(call_with_cfg, f,
                 conditioning=conditioning,
                 cfg_strength=cfg_strength,
                 mask=mask)
    k1 = _f(x,            t)
    k2 = _f(x + dt*k1/2,  t + dt/2)
    k3 = _f(x + dt*k2/2,  t + dt/2)
    k4 = _f(x + dt*k3,    t + dt)
    return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


class ModalFlowMatchingModule(LightningModule):
    def __init__(
        self,
        model_type:             str   = 'deepsets',
        n_fft:                  int   = 8192,
        d_cond:                 int   = 128,
        encoder_base_channels:  int   = 24,
        d_model:                int   = 128,
        d_ff:                   int   = 256,
        n_layers:               int   = 6,
        d_state:                int   = 16,
        d_conv:                 int   = 4,
        expand:                 int   = 2,
        lr:                     float = 1e-4,
        warmup_steps:           int   = 2000,
        cfg_dropout_rate:       float = 0.1,
        val_steps:              int   = 50,
        val_cfg:                float = 2.0,
        test_steps:             int   = 100,
        test_cfg:               float = 2.0,
        compile:                bool  = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = LogMagFFTEncoder(
            n_fft=n_fft,
            d_cond=d_cond,
            base_channels=encoder_base_channels,
        )

        if model_type == 'mamba':
            from model_mamba import BidirectionalMambaVectorField
            self.vector_field = BidirectionalMambaVectorField(
                d_model=d_model,
                d_cond=d_cond,
                d_ff=d_ff,
                n_layers=n_layers,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            from model import DeepSetsVectorField
            self.vector_field = DeepSetsVectorField(
                d_model=d_model,
                d_cond=d_cond,
                d_ff=d_ff,
                n_layers=n_layers,
            )

    def _encode(self, batch):
        return self.encoder(batch['log_mag_fft'])

    def _train_step(self, batch):
        modes   = batch['modes']
        noise   = batch['noise']
        
        # Optimal Transport: Sort the noise by the f0 slot
        sort_idx = noise[..., 0].argsort(dim=1)
        noise = torch.gather(noise, 1, sort_idx.unsqueeze(-1).expand_as(noise))
        
        n_modes = batch['n_modes']
        M       = modes.shape[1]
        
        # Fast GPU masking
        mask = torch.arange(M, device=modes.device)[None, :] < n_modes[:, None]

        # Flow Matching ODE target
        t = torch.rand(modes.shape[0], 1, device=modes.device)
        x_t = (1 - t).unsqueeze(-1) * noise + t.unsqueeze(-1) * modes
        target_drift = modes - noise

        z = self._encode(batch)
        
        # FIX 1: Corrected hyperparameter name
        z = self.vector_field.apply_dropout(z, self.hparams.cfg_dropout_rate)

        # Forward pass
        pred_drift = self.vector_field(x_t, t, z, mask=mask)

        # --- The Masked Physics & Unmasked Exists Loss Split ---
        mask_exp = mask.unsqueeze(-1).float()
        
        # 1. Physics (f0, sigma, gain) - MASKED
        loss_phys = (pred_drift[..., :3] - target_drift[..., :3]).square()
        loss_phys = (loss_phys * mask_exp).sum() / (mask_exp.sum() * 3 + 1e-8)
        
        # 2. Exists flag - UNMASKED (Trains the ghosts to vanish)
        loss_exists = (pred_drift[..., 3] - target_drift[..., 3]).square().mean()

        # Combine with 0.1 weighting
        loss = loss_phys + (0.1 * loss_exists)
        
        # Log them separately
        self.log('train/loss_phys', loss_phys)
        self.log('train/loss_exists', loss_exists)
        self.log('train/loss_step', loss)
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._train_step(batch)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @torch.no_grad()
    def _sample(self, z, noise, steps, cfg, mask=None):
        sample = noise
        t      = torch.zeros(noise.shape[0], 1, device=noise.device)
        dt     = 1.0 / steps
        for _ in range(steps):
            sample = rk4_step(
                self.vector_field, sample, t, dt,
                conditioning=z, cfg_strength=cfg, mask=mask,
            )
            t = t + dt
        return sample

    def validation_step(self, batch, batch_idx):
        modes   = batch['modes']
        n_modes = batch['n_modes']
        B, M, _ = modes.shape
        mask = torch.arange(M, device=modes.device)[None, :] < n_modes[:, None]

        z     = self._encode(batch)
        noise = torch.randn_like(modes)
        
        # Sort noise for OT
        sort_idx = noise[..., 0].argsort(dim=1)
        noise = torch.gather(noise, 1, sort_idx.unsqueeze(-1).expand_as(noise))

        pred  = self._sample(z, noise, self.hparams.val_steps,
                             self.hparams.val_cfg, mask)

        mask_exp = mask.unsqueeze(-1).float()
        mse = ((pred - modes).square() * mask_exp).sum() \
              / (mask_exp.sum() * 4 + 1e-8)
        self.log('val/mse', mse, on_epoch=True, prog_bar=True)

        # FIX 3: Unmask the 'exists' flag for validation logging
        for i, name in enumerate(['f0', 'sigma', 'gain', 'exists']):
            if name == 'exists':
                pm = (pred[..., i] - modes[..., i]).square().mean()
            else:
                pm = ((pred[..., i] - modes[..., i]).square()
                      * mask.float()).sum() / (mask.float().sum() + 1e-8)
            self.log(f'val/mse_{name}', pm, on_epoch=True)

        return mse

    def test_step(self, batch, batch_idx):
        modes   = batch['modes']
        n_modes = batch['n_modes']
        B, M, _ = modes.shape
        mask = torch.arange(M, device=modes.device)[None, :] < n_modes[:, None]

        z     = self._encode(batch)
        noise = torch.randn_like(modes)
        
        # FIX 2: Sort noise for inference test step
        sort_idx = noise[..., 0].argsort(dim=1)
        noise = torch.gather(noise, 1, sort_idx.unsqueeze(-1).expand_as(noise))
        
        pred  = self._sample(z, noise, self.hparams.test_steps,
                             self.hparams.test_cfg, mask)

        mask_exp = mask.unsqueeze(-1).float()
        mse = ((pred - modes).square() * mask_exp).sum() \
              / (mask_exp.sum() * 4 + 1e-8)
        self.log('test/mse', mse)
        return mse

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=1e-4)
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-8, end_factor=1.0,
            total_iters=self.hparams.warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=200_000, eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine],
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