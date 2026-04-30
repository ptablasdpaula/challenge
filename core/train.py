"""
train.py  —  Task B modal CNF training entrypoint

Local/Apocrita Training (DeepSets backend):
  python train.py --data_folder ../random-IR-10000-4.0s \
                  --batch_size 4 --d_model 128 --d_ff 256 \
                  --n_layers 6 --max_epochs 500
"""

import multiprocessing
import argparse
import os

import torch
from torch.utils.data import DataLoader, Subset
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from dataset import ModalPlateDataset, variable_length_collate
from train_module import ModalFlowMatchingModule

# Unlock A100 Tensor Cores
torch.set_float32_matmul_precision('medium')

def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument('--data_folder',  type=str,   required=True)
    p.add_argument('--n_fft',        type=int,   default=8192)
    p.add_argument('--max_modes',    type=int,   default=0,
                   help='0 = dynamic padding per batch')
    # model
    p.add_argument('--d_model',      type=int,   default=128)
    p.add_argument('--d_ff',         type=int,   default=256)
    p.add_argument('--n_layers',     type=int,   default=6)
    p.add_argument('--d_cond',       type=int,   default=128)
    # training
    p.add_argument('--batch_size',   type=int,   default=4)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--max_epochs',   type=int,   default=500)
    p.add_argument('--warmup_steps', type=int,   default=2000)
    p.add_argument('--devices',      type=int,   default=1)
    p.add_argument('--num_workers',  type=int,   default=4)
    p.add_argument('--compile',      action='store_true')
    # overfit
    p.add_argument('--overfit',      action='store_true')
    p.add_argument('--n_samples',    type=int,   default=10)
    # logging
    p.add_argument('--wandb',        action='store_true')
    p.add_argument('--run_name',     type=str,   default='modal-cnf')
    p.add_argument('--val_every',    type=int,   default=1,
                   help='validate every N epochs (use 500 for overfit runs)')
    return p.parse_args()


def make_datasets(args):
    kwargs = dict(
        folder=args.data_folder,
        n_fft=args.n_fft,
        max_modes=args.max_modes if args.max_modes > 0 else None,
    )
    if args.overfit:
        ds = ModalPlateDataset(**kwargs, split='train')
        sub = Subset(ds, list(range(min(args.n_samples, len(ds)))))
        return sub, sub
    return (
        ModalPlateDataset(**kwargs, split='train'),
        ModalPlateDataset(**kwargs, split='val'),
    )


def main():
    args = parse_args()

    # MPS doesn't support pin_memory
    is_mps = torch.backends.mps.is_available()
    pin = not is_mps

    train_ds, val_ds = make_datasets(args)
    collate = (variable_length_collate
               if args.max_modes == 0 else None)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=not args.overfit,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=pin,
        persistent_workers=(args.num_workers > 0),
    )

    model = ModalFlowMatchingModule(
        n_fft=args.n_fft,
        d_cond=args.d_cond,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        compile=args.compile,
    )

    print(f"\nModel:      deepsets")
    print(f"Params:     {sum(p.numel() for p in model.parameters()):,}")
    print(f"Encoder:    {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"VectorField:{sum(p.numel() for p in model.vector_field.parameters()):,}\n")

    # --- UNIFIED CALLBACKS & LOGGING ---
    # 1. Start with the Learning Rate Monitor
    callbacks = [LearningRateMonitor(logging_interval='step')]

    # 2. Add the Checkpoint Manager (Only if NOT overfitting)
    if not args.overfit:
        checkpoint_callback = ModelCheckpoint(
            dirpath='checkpoints/',
            filename='deepsets-{epoch:04d}-{val/RE:.4f}',
            monitor='val/RE',
            mode='min',
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False
        )
        callbacks.append(checkpoint_callback)

    # 3. Setup W&B Logger
    logger = None
    if args.wandb and not args.overfit:
        logger = WandbLogger(project='dafx-taskb', name=args.run_name)
    # -----------------------------------

    overfit_kwargs = dict(overfit_batches=1) if args.overfit else {}

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        devices=args.devices,
        callbacks=callbacks,
        accumulate_grad_batches=16,
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        val_check_interval=1.0,
        check_val_every_n_epoch=args.val_every,
        num_sanity_val_steps=0,
        **overfit_kwargs,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    # Fix the SLURM Dataloader deadlock
    multiprocessing.set_start_method('spawn', force=True)
    main()