"""
smoke_test.py
-------------
Run this first to verify the dataset loads correctly and the model forward
pass works before starting any training.

Usage (from the taskb_deepsets folder):
  python smoke_test.py --data_folder ../random-IR-10000-4.0s
"""

import argparse
import time
import torch
from torch.utils.data import DataLoader, Subset

from dataset import ModalPlateDataset, variable_length_collate
from model import LogMagFFTEncoder, DeepSetsVectorField
from train_module import ModalFlowMatchingModule


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_folder', type=str, required=True)
    p.add_argument('--n_fft', type=int, default=8192)
    args = p.parse_args()

    print("=" * 60)
    print("1. Dataset loading test")
    print("=" * 60)

    ds = ModalPlateDataset(
        folder=args.data_folder,
        n_fft=args.n_fft,
        max_modes=None,   # variable length
        split='train',
    )
    print(f"  Train set size: {len(ds)}")

    # Load a few samples and print shapes/stats
    for i in [0, 1, 2]:
        sample = ds[i]
        m = sample['modes']
        print(f"  Sample {i}: log_mag_fft={sample['log_mag_fft'].shape}, "
              f"modes={m.shape}, n_modes={sample['n_modes']}")
        print(f"    f0    range: [{m[:,0].min():.3f}, {m[:,0].max():.3f}]  (should be ~[-1,1])")
        print(f"    sigma range: [{m[:,1].min():.3f}, {m[:,1].max():.3f}]")
        print(f"    gain  range: [{m[:,2].min():.3f}, {m[:,2].max():.3f}]")
        print(f"    exists     : [{m[:,3].min():.3f}, {m[:,3].max():.3f}]  (all 1.0)")

    print("\n" + "=" * 60)
    print("2. DataLoader (variable-length collate) test")
    print("=" * 60)

    subset = Subset(ds, list(range(8)))
    loader = DataLoader(
        subset,
        batch_size=4,
        collate_fn=variable_length_collate,
        num_workers=0,
    )
    batch = next(iter(loader))
    print(f"  log_mag_fft : {batch['log_mag_fft'].shape}")
    print(f"  modes       : {batch['modes'].shape}")
    print(f"  n_modes     : {batch['n_modes']}")
    print(f"  noise       : {batch['noise'].shape}")

    print("\n" + "=" * 60)
    print("3. Model forward pass test")
    print("=" * 60)

    model = ModalFlowMatchingModule(n_fft=args.n_fft)
    model.eval()

    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Encoder:      {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"  VectorField:  {sum(p.numel() for p in model.vector_field.parameters()):,}")

    with torch.no_grad():
        t0 = time.time()
        # Encoder
        z = model.encoder(batch['log_mag_fft'])
        print(f"  Encoder output shape: {z.shape}  ({time.time()-t0:.2f}s)")

        # Vector field
        t0 = time.time()
        modes  = batch['modes']
        n_modes = batch['n_modes']
        B, M, _ = modes.shape
        mask   = model._make_mask(n_modes, M)
        t_in   = torch.rand(B, 1)
        drift  = model.vector_field(modes, t_in, z, mask=mask)
        print(f"  VectorField output shape: {drift.shape}  ({time.time()-t0:.2f}s)")

    print("\n" + "=" * 60)
    print("4. Training step test")
    print("=" * 60)

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    for step in range(3):
        t0 = time.time()
        loss = model._train_step(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"  Step {step}: loss={loss.item():.6f}  ({time.time()-t0:.2f}s)")

    print("\n✓ All tests passed. Ready to run train.py.")


if __name__ == '__main__':
    main()