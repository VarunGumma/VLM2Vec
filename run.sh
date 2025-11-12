
#!/bin/bash

#PBS -l select=1:mem=128gb:ncpus=112:ngpus=8
#PBS -o /scratch_aisg/peerat_main/VLM2Vec/log-pbs/
#PBS -e /scratch_aisg/peerat_main/VLM2Vec/log-pbs/
#PBS -j oe
#PBS -N Qwen-2B
#PBS -q AISG_large
#PBS -l walltime=48:00:00

# Generate timestamp for unique log files


# Create unique log file with timestamp and job ID
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/scratch_aisg/peerat_main/VLM2Vec/log-pbs/training_${TIMESTAMP}_${PBS_JOBID}.log"

echo "Job started at: $(date)" | tee $LOG_FILE
echo "Job ID: $PBS_JOBID" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE" | tee -a $LOG_FILE

# Set all Hugging Face and ML cache directories to your .cache folder
export HF_HOME=/scratch_aisg/peerat_main/.cache
export HUGGINGFACE_HUB_CACHE=/scratch_aisg/peerat_main/.cache/huggingface/hub
export HF_DATASETS_CACHE=/scratch_aisg/peerat_main/.cache/huggingface/datasets
export SENTENCE_TRANSFORMERS_HOME=/scratch_aisg/peerat_main/.cache/sentence_transformers
export GIT_CONFIG_GLOBAL=/scratch_aisg/peerat_main/.gitconfig

# Additional cache directories
export TORCH_HOME=/scratch_aisg/peerat_main/.cache/torch
export XDG_CACHE_HOME=/scratch_aisg/peerat_main/.cache

# Remove the deprecated TRANSFORMERS_CACHE to avoid the warning
unset TRANSFORMERS_CACHE

# Change to your directory to ensure output goes to the right place
cd /scratch_aisg/peerat_main/VLM2Vec

export NCCL_TIMEOUT=3600000  # 1 hour timeout (in milliseconds)
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

# Additional stability settings
export NCCL_IB_DISABLE=1  # Disable InfiniBand if causing issues
export NCCL_P2P_DISABLE=1  # Disable P2P if causing issues

bash experiments/public/train/train_v2-qwen2vl-2B-series-S.sh 2>&1 | tee -a $LOG_FILE

# torchrun --nproc_per_node=8 pre_training_training.py 2>&1 | tee -a $LOG_FILE