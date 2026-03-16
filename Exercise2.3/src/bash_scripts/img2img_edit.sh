#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J ddim_editing
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

export PATH="$HOME/.local/bin:$PATH"

cd ~/Advanced-DL-CV || { echo "ERROR: project directory not found"; exit 1; }

module load cuda/12.1

echo "--- nvidia-smi ---"
nvidia-smi

export HF_HOME="$HOME/.cache/huggingface"

echo "--- uv sync ---"
uv sync

echo "--- PyTorch CUDA check ---"
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

echo "--- Starting DDIM editing ---"
uv run python Exercise2.3/src/ddim_inversion_simple_editing.py

echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
