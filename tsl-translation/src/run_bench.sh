#!/bin/bash

# Optional SLURM directives for HPC usage:
#SBATCH --partition=compute
#SBATCH --job-name=SLT_SOTA_POLISHED
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bench_%j.out

# Optional benchmark launcher for multi-model runs.
# The Python entry points in src/ are the canonical implementation; this file is
# only a convenience wrapper for batch execution.

set -euo pipefail

# --- [ 1. ENVIRONMENT SETUP ] ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a && source "$PROJECT_ROOT/.env" && set +a
fi

module load miniconda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-soziasign}"

# --- [ 2. LOCAL CACHE SETUP ] ---
# Download models to a temporary job-specific directory rather than the home directory.
JOB_SUFFIX="${SLURM_JOB_ID:-manual}"
export HF_HOME="/tmp/${USER}_hf_cache_${JOB_SUFFIX}"
mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:?Please set HF_TOKEN in .env}"

# --- [ 3. BENCHMARK MATRIX ] ---
# Prioritizing the champion model: Gemma-2 P3_EN
MODELS=("google/gemma-2-9b-it" "meta-llama/Meta-Llama-3.1-8B-Instruct" "Trendyol/Trendyol-LLM-7b-chat-v1.8")
STRATEGIES=("P3_EN" "P2_EN" "P1_EN")

# --- [ 4. EXECUTION LOOP ] ---
for M in "${MODELS[@]}"; do
    # Create a slug for the model folder name (e.g., google/gemma-2-9b-it -> gemma_2_9b_it)
    MODEL_SLUG=$(echo "$M" | rev | cut -d'/' -f1 | rev | tr '[:upper:]' '[:lower:]' | tr '-' '_')
    
    echo "===================================================="
    echo "[NEW MODEL BLOCK] $M"
    echo "===================================================="

    for S in "${STRATEGIES[@]}"; do
        STRAT_LOWER=$(echo $S | tr '[:upper:]' '[:lower:]')
        RESULT_FILE="experiments/benchmarks/$MODEL_SLUG/$STRAT_LOWER/result.json"
        
        # SKIP LOGIC: If result.json already exists, don't waste H100 time
        if [ -f "$RESULT_FILE" ]; then
            echo "[SKIP] Strategy $S already completed for $M. Skipping..."
            continue
        fi
        
        echo "[EXEC] Training and Evaluating: $M | Strategy: $S"
        # Run the Unified Engine (Training + Polished Eval + Latency)
        python src/unified_bench.py --model_id "$M" --strategy "$S"
    done
    
    # CRITICAL CLEANUP: Remove the heavy model weights (15-20GB) after 
    # completing all strategies for this model to free up /tmp space.
    echo "[CLEANUP] Removing weights for $M from temporary storage..."
    rm -rf "$HF_HOME"/*
done

# Final system cleanup
rm -rf "$HF_HOME"
echo "===================================================="
echo "ALL MODELS AND STRATEGIES COMPLETED SUCCESSFULLY."
echo "Check experiments/benchmarks/ for your results."
echo "===================================================="
