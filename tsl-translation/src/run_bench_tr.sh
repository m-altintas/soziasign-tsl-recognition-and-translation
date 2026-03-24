#!/bin/bash

# Optional SLURM directives for HPC usage:
#SBATCH --partition=compute
#SBATCH --job-name=SLT_TR_MIRROR
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/tr_bench_%j.out

# Optional batch launcher for Turkish benchmark runs.

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
export HF_HOME="/tmp/${USER}_tr_cache_${JOB_SUFFIX}"
mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:?Please set HF_TOKEN in .env}"

# Models and their best Turkish prompt strategies
declare -A model_strat
model_strat["google/gemma-2-9b-it"]="P3_TR"
model_strat["meta-llama/Meta-Llama-3.1-8B-Instruct"]="P2_TR"
model_strat["Trendyol/Trendyol-LLM-7b-chat-v1.8"]="P2_TR"

for M in "${!model_strat[@]}"; do
    S=${model_strat[$M]}
    echo "=========================================="
    echo "[START] TR Mirror: $M with $S"
    
    rm -rf "$HF_HOME"/* # Clear previous model cache
    python src/unified_bench_tr.py --model_id "$M" --strategy "$S"
done

rm -rf "$HF_HOME"
echo "ALL TR EXPERIMENTS DONE."
