#!/bin/bash
#SBATCH --job-name=WO_classify
#SBATCH --account=project_462001516
#SBATCH --partition=standard-g
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
# LUMI-G exposes 56 usable CPU cores; 8 ranks x 7 cores fits one node.
#SBATCH --cpus-per-task=7
#SBATCH --mem=450G
#SBATCH --gpus-per-node=8
#SBATCH -o /scratch/project_462001516/users/tarkkaot/MultiWebOrganizerPlus/logs/wo_classify_%j.out
#SBATCH -e /scratch/project_462001516/users/tarkkaot/MultiWebOrganizerPlus/logs/wo_classify_%j.err

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load Local-LAIF lumi-aif-singularity-bindings

export HF_HOME="/scratch/project_462001516/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"

base_dir="/scratch/project_462001516/users/tarkkaot/MultiWebOrganizerPlus"
SIF="$base_dir/environments/lumi/WO_inference.sif"
python_script="$base_dir/scripts/run_weborganizer.py"
gpu_wrapper="$base_dir/hpc/lumi/select_gpu.sh"
input_file="$base_dir/data/FineWebSample.jsonl"
output_file="$base_dir/results/WO_MultiSynt_formats.jsonl"
rank_count=8

[[ -f "$SIF" ]]
[[ -f "$python_script" ]]
[[ -x "$gpu_wrapper" ]]
[[ -f "$input_file" ]]
mkdir -p "$base_dir/logs" "$base_dir/results" "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

srun --cpu-bind=cores "$gpu_wrapper" \
                    singularity exec -B /scratch/project_462001516 "$SIF" \
                    python "$python_script" \
                    --dataset-input "MultiSynt/MT-Nemotron-CC" \
                    --output "$output_file" \
                    --aspect format \
                    --batch-size 256 \
                    --max-length 2048 \
                    --log-every-batches 100 \
                    --log-level INFO \
                    --memory-efficient-attention \
                    --bfloat16 \
                    --compact-output \
                    --length-aware-batching \

# Post-processing: concatenate rank output files and verify line counts match.
shopt -s nullglob
all_rank_outputs=("$output_file".rank*)
progress_files=("$output_file".rank*.progress.json)
rank_outputs=()
for rank_output in "${all_rank_outputs[@]}"; do
    if [[ "$rank_output" != *.progress.json ]]; then
        rank_outputs+=("$rank_output")
    fi
done
if (( ${#rank_outputs[@]} != rank_count )); then
    echo "Expected $rank_count rank output files, found ${#rank_outputs[@]}" >&2
    exit 1
fi

# Calculate the total number of lines in the rank output files
# and compare it to the concatenated output file.
rank_line_count=0
for rank_output in "${rank_outputs[@]}"; do
    rank_line_count=$((rank_line_count + $(wc -l < "$rank_output")))
done

cat "${rank_outputs[@]}" > "$output_file"
output_line_count=$(wc -l < "$output_file")

echo "Rank output lines: $rank_line_count; concatenated output lines: $output_line_count"

# Exit with an error if the line counts do not match
if (( rank_line_count != output_line_count )); then
    echo "Rank output line count does not match concatenated output" >&2
    exit 1
fi

# Clean up the rank output files and progress files if no errors occurred
if (( ${#progress_files[@]} > 0 )); then
    rm -- "${progress_files[@]}"
fi
