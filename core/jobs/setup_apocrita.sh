#!/bin/bash
module load python/3.11
module load cuda/12.4.0-gcc-12.2.0

rm -rf ~/envs/dafx
python -m venv ~/envs/dafx
source ~/envs/dafx/bin/activate

pip install --upgrade pip
pip install packaging wheel ninja

pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install causal-conv1d mamba-ssm --no-build-isolation

pip install lightning einops wandb pandas numpy scipy soundfile

python -c "from mamba_ssm import Mamba; print('Mamba OK')"
