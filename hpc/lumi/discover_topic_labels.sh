#!/bin/bash
#SBATCH --job-name=discover_topic_labels
#SBATCH --account=project_462001516
#SBATCH --partition=dev-g
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gpus-per-node=2
#SBATCH -o ../logs/discover_%j.out
#SBATCH -e ../logs/discover_%j.err

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export HF_HOME="/scratch/project_462001516/users/tarkkaot/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"

base_dir="/scratch/project_462001516/users/tarkkaot/MultiWebOrganizerPlus"
SIF="$base_dir/environments/lumi/BASE.sif"
python_script="$base_dir/scripts/label_pipeline.py"

srun singularity run -B /scratch/project_462001516 $SIF python "$python_script" \
                        --config "$base_dir/configs/pipeline_defaults.yaml" \
                        --mode discover \
                        --input "$base_dir/data/FineWebSample.jsonl" \
                        --reset-discovery
