#!/bin/bash
#SBATCH --job-name=discover_topic_labels
#SBATCH --account=project_462001516
#SBATCH --partition=small-g
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --gpus-per-node=4
#SBATCH -o ../logs/discover_%j.out
#SBATCH -e ../logs/discover_%j.err

module purge >/dev/null 2>&1
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# Keep expected third-party startup warnings out of the SLURM error log.
export PYTHONWARNINGS="ignore::FutureWarning"
export TRANSFORMERS_VERBOSITY="error"

# Set up Hugging Face cache directories in the scratch space.
export HF_HOME="/scratch/project_462001516/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

# Define paths to the base directory, Singularity image, and Python script.
base_dir="/scratch/project_462001516/users/tarkkaot/MultiWebOrganizerPlus"
SIF="$base_dir/environments/lumi/LABEL_PIPELINE.sif"
python_script="$base_dir/scripts/label_pipeline.py"

# Set vLLM cache root to a directory in the scratch space
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$base_dir/cache/vllm}"
mkdir -p "$VLLM_CACHE_ROOT"

# Run inside the Singularity container.
# Bind the scratch space to the container to ensure access to files and directories.
# Give arguments either here with flags or in the config file.
# Arguments given here will override those in the config file.
srun singularity run -B /scratch/project_462001516 "$SIF" python "$python_script" \
                        --config "$base_dir/configs/creative2.yaml" \
                        --mode discover \
                        --jsonl-input "$base_dir/data/FineWebSmallSample.jsonl" \
