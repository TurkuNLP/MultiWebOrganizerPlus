#!/bin/bash
#SBATCH --job-name=tests
#SBATCH --account=project_462001516
#SBATCH --partition=debug
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/project_462001516/users/tarkkaot/MultiWebOrganizerPlus/logs/tests_%j.out
#SBATCH --error=/scratch/project_462001516/users/tarkkaot/MultiWebOrganizerPlus/logs/tests_%j.err

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

base_dir="/scratch/project_462001516/users/tarkkaot/MultiWebOrganizerPlus"
SIF="$base_dir/environments/lumi/BASE.sif"

[[ -f "$SIF" ]]
mkdir -p "$base_dir/logs"

export PYTHONWARNINGS="ignore::FutureWarning"
export TRANSFORMERS_VERBOSITY="error"

srun singularity exec -B /scratch/project_462001516 "$SIF" \
    env PYTHONPATH="$base_dir/scripts" \
    python -m pytest -q "$base_dir/scripts/tests/test_label_pipeline.py"