#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J evaluate_reconstruction
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need 8GB of memory per core/slot --
#BSUB -R "rusage[mem=8GB]"
### -- specify that we want the job to get killed if it exceeds 9GB per core/slot --
#BSUB -M 9GB
### -- request 1 GPU --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --
#BSUB -W 4:00
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
#BSUB -o Output_%J.out
#BSUB -e Output_%J.err

echo "=========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

# Make sure uv is on PATH (installed to ~/.local/bin by default)
export PATH="$HOME/.local/bin:$PATH"

# Navigate to the project root (adjust if your path differs)
cd ~/Advanced-DL-CV || { echo "ERROR: project directory not found"; exit 1; }

# Load CUDA (check available versions with: module avail cuda)
module load cuda/12.1

# Verify GPU is visible
echo "--- nvidia-smi ---"
nvidia-smi

# Point HuggingFace cache to a scratch/work directory to avoid home quota issues
# Change this path to your actual scratch space on the cluster
export HF_HOME="$HOME/.cache/huggingface"

# Sync dependencies (installs/updates the venv from pyproject.toml)
echo "--- uv sync ---"
uv sync

# Confirm CUDA is available to PyTorch before running the full job
echo "--- PyTorch CUDA check ---"
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

echo "--- Starting evaluation ---"
uv run python Exercise2.3/src/evaluate_reconstruction.py

echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
