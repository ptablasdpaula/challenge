#!/bin/bash
#SBATCH --job-name=taskb_mamba
#SBATCH --output=qlogs/%j.out
#SBATCH --error=qlogs/%j.err
#SBATCH --account=pilot_andrena
#SBATCH --partition=andrena
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@300

set -euo pipefail

REPO=/data/home/acw794/challenge/TaskB_CNF
DATA=/data/home/acw794/challenge/random-IR-10000-4.0s

cd $REPO
mkdir -p qlogs

# 1. Load exact modules
module load python/3.11
module load cuda/12.4.0-gcc-12.2.0

# 2. Smart Venv Creation (Don't nuke it if it exists!)
if [ ! -d "$HOME/envs/dafx" ]; then
    echo "Creating fresh virtual environment..."
    python -m venv ~/envs/dafx
fi
source ~/envs/dafx/bin/activate

# 3. Core dependencies and build tools
pip install --upgrade pip
pip install packaging wheel ninja

# 4. Install PyTorch locked to CUDA 12.4
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 5. The Mamba-SSM Installation (Pinned to 1.2.2 for PyTorch 2.4 compatibility)
if ! python -c "import mamba_ssm" &> /dev/null; then
    echo "Compiling causal-conv1d and mamba-ssm 1.2.2 from source against torch 2.4..."
    echo "This wiall take ~5-10 minutes. Please wait..."
    
    # Notice the .post1 on causal-conv1d!
    pip install causal-conv1d==1.2.2.post1 --no-build-isolation --no-cache-dir
    pip install mamba-ssm==1.2.2 --no-build-isolation --no-cache-dir
else
    echo "Mamba-SSM is already installed! Skipping compile."
fi

# 6. Install other project deps (Pin transformers to 4.39.3)
pip install transformers==4.39.3 lightning einops wandb pandas numpy scipy soundfile

# 7. Verification (Test GPU Fast Path)
echo "=== Environment Check ==="
python -c "import torch; print('Torch:', torch.__version__, '| CUDA OK:', torch.cuda.is_available())"
python -c "from mamba_ssm import Mamba; import torch; m = Mamba(64).cuda(); x = torch.randn(1,100,64).cuda(); print('Mamba fwd OK:', m(x).shape)"
echo "========================="

# 8. Start the Run
echo "Starting mamba training..."
python train.py \
  --data_folder $DATA \
  --model mamba \
  --batch_size 1 \
  --max_modes 0 \
  --max_epochs 500 \
  --val_every 10 \
  --warmup_steps 500 \
  --lr 3e-4 \
  --d_model 256 --d_ff 512 --n_layers 8 --d_cond 256 \
  --num_workers 0 \
  --wandb