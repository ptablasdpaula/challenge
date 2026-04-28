"""
train.py
--------
Training entrypoint for Task B DeepSets CNF.

Quick sanity check (overfit 10 samples):
  python train.py --data_folder ../random-IR-10000-4.0s --overfit --n_samples 10

Full training:
  python train.py --data_folder ../random-IR-10000-4.0s

Key args:
  --data_folder   : path to the dataset directory
  --overfit       : overfit on --n_samples samples (sanity check)
  --n_samples     : number of samples to use when --overfit is set (default 10)
  --max_modes     : pad/truncate all samples to this fixed number of modes.
                    Set to 0 to use dynamic padding (variable length per batch).
  --batch_size    : (default 4; reduce if OOM)
  --n_fft         : FFT size for encoder input (default 8192)
  --d_model       : DeepSets hidden dim (default 128)
  --n_layers      : number of DeepSets blocks (default 6)
  --max_epochs    : (default 500)
  --devices       : number of GPUs (default 1)
"""

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from dataset import ModalPlateDataset, variable_length_collate
from train_module import ModalFlowMatchingModule


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_folder', type=str, required=True)
    p.add_argument('--overfit',     action='store_true')
    p.add_argument('--n_samples',   type=int, default=10)
    p.add_argument('--max_modes',   type=int, default=0,
                   help='0 = dynamic padding per batch (recommended)')
    p.add_argument('--batch_size',  type=int, default=4)
    p.add_argument('--n_fft',       type=int, default=8192)
    p.add_argument('--d_model',     type=int, default=128)
    p.add_argument('--d_ff',        type=int, default=256)
    p.add_argument('--n_layers',    type=int, default=6)
    p.add_argument('--d_cond',      type=int, default=128)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--max_epochs',  type=int, default=500)
    p.add_argument('--devices',     type=int, default=1)
    p.add_argument('--wandb',       action='store_true')
    p.add_argument('--run_name',    type=str, default='deepsets-taskb')
    p.add_argument('--compile',     action='store_true')
    return p.parse_args()


def make_datasets(args):
    kwargs = dict(
        folder=args.data_folder,
        n_fft=args.n_fft,
        max_modes=args.max_modes if args.max_modes > 0 else None,
    )

    if args.overfit:
        # Use a tiny subset for the overfit sanity check
        ds = ModalPlateDataset(**kwargs, split='train')
        subset = Subset(ds, list(range(min(args.n_samples, len(ds)))))
        return subset, subset    # train == val for overfit check
    else:
        train_ds = ModalPlateDataset(**kwargs, split='train')
        val_ds   = ModalPlateDataset(**kwargs, split='val')
        return train_ds, val_ds


def main():
    args = parse_args()

    train_ds, val_ds = make_datasets(args)

    use_variable = (args.max_modes == 0)
    collate_fn   = variable_length_collate if use_variable else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=not args.overfit,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=False,
        persistent_workers=True,
    )

    model = ModalFlowMatchingModule(
        n_fft=args.n_fft,
        d_cond=args.d_cond,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        lr=args.lr,
        compile=args.compile,
    )

    # ── callbacks ────────────────────────────────────────────────────────────
    callbacks = [LearningRateMonitor(logging_interval='step')]
    if not args.overfit:
        callbacks.append(ModelCheckpoint(
            monitor='val/mse',
            mode='min',
            save_top_k=3,
            filename='epoch{epoch:03d}-val_mse{val/mse:.4f}',
            auto_insert_metric_name=False,
        ))

    # ── logger ────────────────────────────────────────────────────────────────
    logger = None
    if args.wandb and not args.overfit:
        logger = WandbLogger(project='dafx-taskb', name=args.run_name)

    # ── trainer ───────────────────────────────────────────────────────────────
    overfit_kwargs = dict(overfit_batches=1) if args.overfit else {}

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        devices=args.devices,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        val_check_interval=1.0,
        check_val_every_n_epoch=50 if args.overfit else 1,
        **overfit_kwargs,
    )

    print(f"\nModel parameter count: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Encoder parameters:    {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"VectorField parameters:{sum(p.numel() for p in model.vector_field.parameters()):,}")

    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()