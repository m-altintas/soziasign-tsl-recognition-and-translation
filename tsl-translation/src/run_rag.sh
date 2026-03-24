#!/bin/bash

# Optional SLURM directives for HPC usage:
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/rag_bench_%j.out

# Optional launcher for the RAG benchmark.

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

# Use a job-specific temporary cache to avoid mutating unrelated user caches.
JOB_SUFFIX="${SLURM_JOB_ID:-manual}"
export HF_HOME="/tmp/${USER}_rag_cache_${JOB_SUFFIX}"
mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:?Please set HF_TOKEN in .env}"

python src/rag_bench.py

rm -rf "$HF_HOME"
