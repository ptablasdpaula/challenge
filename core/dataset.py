"""
dataset.py
----------
PyTorch Dataset for DAFx Task B modal parameter estimation.

Each sample contains:
  - log_mag_fft : (N_fft,) float32  — log-magnitude spectrum of the IR
  - modes       : (M, 4) float32    — normalised (f0, sigma, gain, exists) per mode
  - n_modes     : int               — number of real modes M (before padding)

Normalisation (all to approximately [-1, 1]):
  f0    : log-scale,    bounds from empirical stats (1.5, 10000) Hz
  sigma : log-scale,    bounds (1.15, 922)
  gain  : signed-log,   symmetric around zero
  exists: 0.0 or 1.0   (left as-is, flow treats it as continuous)

The gains from the plate model are O(1e-10). We use signed-log:
  signed_log(x) = sign(x) * log(1 + |x| / scale)
where scale is set to the p1 absolute value so the bulk of the distribution
maps to a reasonable range. We store the scale as a dataset attribute.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple


# ── fixed normalisation bounds derived from dataset statistics ────────────────
F0_LOG_MIN  = np.log(1.5)
F0_LOG_MAX  = np.log(10000.0)

SIGMA_LOG_MIN = np.log(1.15)
SIGMA_LOG_MAX = np.log(923.0)

# Empirical p1 absolute gain value used as the signed-log scale factor.
# Anything smaller than this maps to ≈ 0; anything at p99 maps to ≈ log(2) ≈ 0.69
# We then divide by log(2) so the p99 ≈ 1.0.
GAIN_SCALE = 8.2e-11   # ≈ p1 of |gain|
GAIN_NORM  = np.log(1 + 6.5e-10 / GAIN_SCALE)  # ≈ log(1 + p99/scale)


def normalise_f0(f0: np.ndarray) -> np.ndarray:
    """Log-scale, map [F0_LOG_MIN, F0_LOG_MAX] -> [-1, 1]."""
    log_f0 = np.log(np.clip(f0, 1e-6, None))
    return 2.0 * (log_f0 - F0_LOG_MIN) / (F0_LOG_MAX - F0_LOG_MIN) - 1.0


def denormalise_f0(f0_norm: np.ndarray) -> np.ndarray:
    log_f0 = (f0_norm + 1.0) / 2.0 * (F0_LOG_MAX - F0_LOG_MIN) + F0_LOG_MIN
    return np.exp(log_f0)


def normalise_sigma(sigma: np.ndarray) -> np.ndarray:
    """Log-scale, map [SIGMA_LOG_MIN, SIGMA_LOG_MAX] -> [-1, 1]."""
    log_s = np.log(np.clip(sigma, 1e-6, None))
    return 2.0 * (log_s - SIGMA_LOG_MIN) / (SIGMA_LOG_MAX - SIGMA_LOG_MIN) - 1.0


def denormalise_sigma(sigma_norm: np.ndarray) -> np.ndarray:
    log_s = (sigma_norm + 1.0) / 2.0 * (SIGMA_LOG_MAX - SIGMA_LOG_MIN) + SIGMA_LOG_MIN
    return np.exp(log_s)


def normalise_gain(gain: np.ndarray) -> np.ndarray:
    """Signed-log normalisation, output approximately in [-1, 1]."""
    signed_log = np.sign(gain) * np.log(1.0 + np.abs(gain) / GAIN_SCALE)
    return signed_log / GAIN_NORM


def denormalise_gain(gain_norm: np.ndarray) -> np.ndarray:
    signed_log = gain_norm * GAIN_NORM
    return np.sign(signed_log) * GAIN_SCALE * (np.exp(np.abs(signed_log)) - 1.0)


def compute_log_mag_fft(ir: np.ndarray, n_fft: int = 8192) -> np.ndarray:
    """
    Compute log-magnitude FFT of the IR.
    Returns the one-sided spectrum, shape (n_fft // 2 + 1,).
    Normalised to zero mean, unit std using fixed empirical constants
    (just clamp + log for now; dataset-level stats can refine this).
    """
    # Zero-pad or truncate to n_fft
    if len(ir) >= n_fft:
        ir_windowed = ir[:n_fft]
    else:
        ir_windowed = np.pad(ir, (0, n_fft - len(ir)))

    spec = np.fft.rfft(ir_windowed)
    log_mag = np.log(np.abs(spec) + 1e-10)
    return log_mag.astype(np.float32)


class ModalPlateDataset(Dataset):
    """
    Dataset of plate impulse responses with modal parameter targets.

    Args:
        folder      : path to the dataset folder (random-IR-XXXX-XXs)
        n_fft       : FFT size for the log-magnitude encoder input
        max_modes   : if set, pad/truncate modes to this fixed size.
                      If None, each sample has variable length — only
                      works with batch_size=1 or a custom collate_fn.
        split       : 'train', 'val', or 'test'
        val_frac    : fraction of data to use for validation
        test_frac   : fraction of data to use for testing
        seed        : random seed for split
    """

    def __init__(
        self,
        folder: str,
        n_fft: int = 8192,
        max_modes: Optional[int] = None,
        split: str = 'train',
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
    ):
        self.folder = Path(folder)
        self.n_fft = n_fft
        self.max_modes = max_modes

        # Discover all npz files
        all_npz = sorted(self.folder.glob("random_IR_[0-9]*.npz"))
        assert len(all_npz) > 0, f"No NPZ files found in {folder}"

        # Deterministic split
        rng = np.random.default_rng(seed)
        indices = np.arange(len(all_npz))
        rng.shuffle(indices)

        n_total = len(indices)
        n_test  = int(n_total * test_frac)
        n_val   = int(n_total * val_frac)

        if split == 'test':
            chosen = indices[:n_test]
        elif split == 'val':
            chosen = indices[n_test:n_test + n_val]
        else:  # train
            chosen = indices[n_test + n_val:]

        self.npz_files = [all_npz[i] for i in chosen]
        print(f"[ModalPlateDataset] split={split}, n_samples={len(self.npz_files)}")

    def __len__(self) -> int:
        return len(self.npz_files)

    def _load_modes(self, npz_path: Path) -> Tuple[np.ndarray, int]:
        """Load and normalise modes from the corresponding CSV."""
        stem = npz_path.stem  # e.g. "random_IR_0001"
        idx  = stem.split('_')[-1]  # "0001"
        csv_path = npz_path.parent / f"random_IR_modes_{idx}.csv"

        df = pd.read_csv(csv_path)
        f0    = df['f0'].values.astype(np.float64)
        sigma = df['sigma'].values.astype(np.float64)
        gain  = df['gain'].values.astype(np.float64)
        M = len(f0)

        # Sort by f0 (natural canonical order for plate modes)
        sort_idx = np.argsort(f0)
        f0, sigma, gain = f0[sort_idx], sigma[sort_idx], gain[sort_idx]

        # Normalise
        f0_n    = normalise_f0(f0).astype(np.float32)
        sigma_n = normalise_sigma(sigma).astype(np.float32)
        gain_n  = normalise_gain(gain).astype(np.float32)
        exists  = np.ones(M, dtype=np.float32)

        modes = np.stack([f0_n, sigma_n, gain_n, exists], axis=1)  # (M, 4)
        return modes, M

    def __getitem__(self, idx: int) -> dict:
        npz_path = self.npz_files[idx]
        data = np.load(npz_path)
        ir   = data['ir'].astype(np.float64)

        # Encoder input
        log_mag = compute_log_mag_fft(ir, self.n_fft)  # (n_fft//2+1,)

        # Modal targets
        modes, n_modes = self._load_modes(npz_path)  # (M, 4), int

        if self.max_modes is not None:
            if n_modes <= self.max_modes:
                # Pad with zeros (exists=0 for padding rows)
                pad = np.zeros((self.max_modes - n_modes, 4), dtype=np.float32)
                modes = np.concatenate([modes, pad], axis=0)
            else:
                # Truncate to max_modes (sorted by f0, so keep lowest)
                modes = modes[:self.max_modes]
                n_modes = self.max_modes

        return {
            'log_mag_fft': torch.from_numpy(log_mag),          # (N_fft//2+1,)
            'modes':       torch.from_numpy(modes),             # (M, 4) or (max_modes, 4)
            'n_modes':     n_modes,                             # int
            'noise':       torch.randn_like(torch.from_numpy(modes)),  # pre-sampled noise
        }


def variable_length_collate(batch: list) -> dict:
    """
    Collate function for variable-length mode sequences.
    Pads all samples in the batch to the longest sequence in the batch.
    This is more memory-efficient than a global max_modes.
    """
    max_m = max(item['n_modes'] for item in batch)

    log_mags, modes_list, n_modes_list, noises = [], [], [], []

    for item in batch:
        log_mags.append(item['log_mag_fft'])
        n_modes_list.append(item['n_modes'])

        m = item['modes']       # (M_i, 4)
        M_i = m.shape[0]
        if M_i < max_m:
            pad = torch.zeros(max_m - M_i, 4)
            m   = torch.cat([m, pad], dim=0)
        modes_list.append(m)

        # Re-sample noise to match padded length
        noises.append(torch.randn(max_m, 4))

    return {
        'log_mag_fft': torch.stack(log_mags),          # (B, N_fft//2+1)
        'modes':       torch.stack(modes_list),          # (B, max_m, 4)
        'n_modes':     torch.tensor(n_modes_list),       # (B,)
        'noise':       torch.stack(noises),              # (B, max_m, 4)
    }