#!/bin/bash
#SBATCH --job-name=tests
#SBATCH --account=project_2020507
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=../logs/tests_%j.out
#SBATCH --error=../logs/tests_%j.err

set -euo pipefail

module purge
module load python-vllm/0.19.1

# Set the number of CPU threads based on cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Unset CC and CXX to avoid conflicts with the Singularity container's compilers.
#unset CC CXX

# Keep expected third-party startup warnings out of the SLURM error log.
export PYTHONWARNINGS="ignore::FutureWarning"
export TRANSFORMERS_VERBOSITY="error"

# Set up Hugging Face cache directories in the scratch space.
export HF_HOME="/scratch/project_2020507/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

# Define paths to the base directory, Singularity image, and Python script.
base_dir="/scratch/project_2020507/users/tarkkaot/MultiWebOrganizerPlus"

# Set vLLM cache root to a directory in the scratch space
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$base_dir/cache/vllm}"
mkdir -p "$VLLM_CACHE_ROOT"

srun python -m pytest -q "$base_dir/scripts/tests/test_label_pipeline.py"