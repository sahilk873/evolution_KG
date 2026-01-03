#!/bin/bash
#
# Submit with: sbatch run_experiments.sh

#SBATCH --job-name=evo-poc
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%jquick.out
#SBATCH --error=logs/%x_%jquick.err

set -euo pipefail

DEVICE="${DEVICE:-cuda}"

echo "========== SLURM JOB INFO =========="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node(s):       $SLURM_NODELIST"
echo "GPUs:          ${SLURM_GPUS:-N/A}"
echo "CPUs/task:     $SLURM_CPUS_PER_TASK"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "Working dir:   $(pwd)"
echo "Start time:    $(date)"
echo "===================================="

# Make sure log directory exists
mkdir -p logs

# ---- Activate virtual environment ----
source .venv/bin/activate

# ---- Sanity check that GPU is visible ----
echo "---- nvidia-smi ----"
nvidia-smi || true

# ---- Performance-related env vars ----
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1

# ---- GPU memory monitor (background) ----
GPU_LOG="logs/gpu_mem_${SLURM_JOB_ID}.log"

monitor_gpu_mem() {
    echo "timestamp,used_MiB,total_MiB" > "$GPU_LOG"
    while true; do
        nvidia-smi \
          --query-gpu=timestamp,memory.used,memory.total \
          --format=csv,noheader,nounits >> "$GPU_LOG" 2>/dev/null || true
        sleep 2
    done
}

monitor_gpu_mem &
GPU_MONITOR_PID=$!

# ---- Run experiments ----

python -m scripts.run_experiments \
    --split random \
    --method all \
    --model rotate \
    --device cuda \
    --budget 10 \
    --pop_size 5 \
    --generations 2 \
    --topk 2 \
    --seeds 1 \
    --epochs 1 \
    --batch_size 16 \
    --max-rows 5 \
    --max-cases 1 \
    --baseline-method full_embed \
    --classifier logistic \
    --reuse-artifacts \
    --skip-prep

# ---- Stop GPU monitor ----
kill $GPU_MONITOR_PID || true
wait $GPU_MONITOR_PID 2>/dev/null || true

# ---- Report peak GPU memory usage ----
echo "========== GPU MEMORY SUMMARY =========="
awk -F',' '
NR>1 {
    total = $3;
    if ($2 > max) {
        max = $2;
    }
}
END {
    percent = total ? (max/total)*100 : 0;
    printf("Peak GPU memory used: %d MiB / %d MiB (%.1f%%)\n",
           max, total, percent);
}
' "$GPU_LOG"
echo "Detailed log: $GPU_LOG"
echo "========================================"

echo "End time: $(date)"
