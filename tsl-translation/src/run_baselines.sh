#!/bin/bash

# Optional SLURM directives for HPC usage:
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/base_dual_%j.out

# Optional benchmark launcher for baseline models.
# This script is not required for reproducing the codebase itself; it is a
# convenience wrapper for batch execution on Unix-like systems or HPC clusters.

set -euo pipefail

# --- [ ENVIRONMENT SETUP ] ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a && source "$PROJECT_ROOT/.env" && set +a
fi

module load miniconda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-soziasign}"

JOB_SUFFIX="${SLURM_JOB_ID:-manual}"
export HF_HOME="/tmp/${USER}_base_dual_${JOB_SUFFIX}"
mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:?Please set HF_TOKEN in .env}"

MODELS=("google/gemma-2-9b-it" "meta-llama/Meta-Llama-3.1-8B-Instruct" "Trendyol/Trendyol-LLM-7b-chat-v1.8")

for M in "${MODELS[@]}"; do
    rm -rf "$HF_HOME"/*
    python src/base_model_bench.py --model_id "$M"
done

rm -rf "$HF_HOME"
