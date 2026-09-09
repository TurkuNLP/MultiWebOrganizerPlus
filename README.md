# MultiWebOrganizerPlus

MultiWebOrganizerPlus is a two-stage, vLLM-based pipeline for assigning topic
labels to documents. The first stage discovers and reconciles an evolving
taxonomy. The second stage classifies the corpus against the resulting frozen
taxonomy. The seed labels are derived from [WebOrganizer](https://github.com/CodeCreator/WebOrganizer), but the pipeline is designed to work with any seed taxonomy.

## Pipeline Logic

The entrypoint is `scripts/label_pipeline.py`, which dispatches to one of two modes: discovery or classification. The discovery mode iteratively discovers and reconciles labels, while the classification mode assigns labels to documents using a frozen taxonomy. The following diagram illustrates the pipeline flow:

```text
input JSONL + seed labels
						 |
						 v
			 discovery mode
						 |
	model labels documents and
	proposes missing topic labels
						 |
						 v
		periodic reconciliation
	defer / reject / map / create
	merge or revise dynamic labels
						 |
						 v
	frozen taxonomy_state.json
						 |
						 v
			 classify mode
						 |
						 v
		 final_labels.jsonl
```

### Discovery

For each input document, the model receives the current active taxonomy and
returns:

- `assigned_label_ids`: existing seed or dynamic labels;
- `proposed_labels`: candidate labels with a name and definition.

The validated result is appended to `discovery.jsonl`. Every
`--reconcile-every` documents, the pipeline groups equivalent proposals and
asks the model to resolve every group. A proposal can be deferred, rejected,
mapped to an existing label, or created as a new dynamic label. Existing
dynamic labels may also be merged or revised. The taxonomy state records the
schema version, aliases, deferred proposals, reconciliation history, and the
last processed discovery sequence.

After the input is exhausted, a final reconciliation resolves all deferred
proposals without allowing new deferrals. The taxonomy is then marked frozen
and receives a content hash. A frozen taxonomy cannot be used to resume
discovery; start a new run if a new taxonomy is required.

Discovery is resumable. Existing `discovery.jsonl` records are checked for
contiguous sequence numbers and duplicate document IDs, and completed
documents are skipped. Resuming requires the same input fingerprint, seed
labels, model settings, and discovery settings that are stored in the
taxonomy state.

### Classification

Classification loads a frozen `taxonomy_state.json` and takes a snapshot of its
active labels. It then sends documents to the model in batches and appends
validated records to `final_labels.jsonl`. Each record includes the taxonomy
version and taxonomy hash. A companion `final_labels.jsonl.meta.json` stores
the input fingerprint and model/runtime settings, so an interrupted run can be
resumed safely. Use `--overwrite-output` only when deliberately starting the
classification output again.

## Repository Layout

- `scripts/label_pipeline.py`: command-line entrypoint.
- `scripts/label_pipeline_lib/`: configuration, Pydantic schemas, model calls, discovery,
	reconciliation, classification, persistence, and validation logic.
- `configs/seed_labels/topics.yaml`: seed taxonomy labels.
- `configs/pipeline_defaults.yaml`: example defaults for a debug run.
- `configs/full_run1.yaml`: LUMI/full-run configuration template.
- `data/FineWebSample.jsonl`: example input corpus.
- `hpc/lumi/discover_topic_labels.sh`: Slurm script for discovery on LUMI.
- `results/`: taxonomy, intermediate discovery, and final classification
	artifacts.

## Requirements

The pipeline requires Python with the packages used by the scripts, including
vLLM, PyTorch, Transformers, Pydantic, and PyYAML, plus a compatible GPU
environment and access to the configured Hugging Face model. On LUMI, use the
provided container image described in `environments/lumi/README.md`.

The model is downloaded and cached by Hugging Face. Set `HF_HOME` (or the
relevant Hugging Face cache variables) to a persistent location with enough
space before submitting a long run.

## Prepare a Run

1. Change to the repository root:

	 ```bash
	 cd /path/to/MultiWebOrganizerPlus
	 ```

2. Check the input corpus. It must be JSONL with at least one record per line,
	 containing string `doc_id` and `text` fields. Additional fields are allowed.

	 ```json
	 {"doc_id": "doc-001", "text": "A document to label."}
	 ```

3. Check the seed-label YAML. It must be a list of objects with unique `id`,
	 `name`, and `definition` values. The configured
	 `expected_seed_label_count` must match the number of entries (the supplied
	 topic file contains 24 labels).

4. Copy a configuration template and edit its `paths` section. Use absolute
	 paths on a cluster, or paths that are valid from the directory where the
	 command will run. Set `taxonomy` and `discovery_output` to a new results
	 directory for a fresh run:

	 ```bash
	 cp configs/full_run1.yaml configs/my_run.yaml
	 $EDITOR configs/my_run.yaml
	 ```

	 Important settings include `model.name`, `model.tensor_parallel_size`,
	 `model.batch_size` and `reconcile.reconcile_every`.

## Run on LUMI

1. Edit `hpc/lumi/discover_topic_labels.sh` and replace its project-specific
	 `base_dir`, model/cache locations, and Slurm resources as needed. The script
	 currently uses four GPUs and points at `configs/full_run1.yaml`.

2. Submit discovery:

	 ```bash
	 cd /scratch/project_<id>/users/<user>/MultiWebOrganizerPlus
	 sbatch hpc/lumi/discover_topic_labels.sh
	 ```

3. Monitor the job and logs:

	 ```bash
	 squeue -u "$USER"
	 tail -f hpc/logs/discover_<jobid>.out
	 tail -f hpc/logs/discover_<jobid>.err
	 ```

4. Wait for the log message indicating that discovery completed and the
	 taxonomy is frozen. The configured results directory should contain
	 `taxonomy_state.json` and `discovery.jsonl`.

5. Run classification in a Slurm job using the same container and model
	 environment. From the repository root, the command inside that job is:

	 ```bash
	 srun singularity run -B /scratch/project_<id> "$SIF" \
		 python scripts/label_pipeline.py \
			 --config configs/my_run.yaml \
			 --mode classify \
			 --input /scratch/project_<id>/users/<user>/MultiWebOrganizerPlus/data/FineWebSample.jsonl \
			 --taxonomy /scratch/project_<id>/users/<user>/MultiWebOrganizerPlus/results/my_run/taxonomy_state.json \
			 --output /scratch/project_<id>/users/<user>/MultiWebOrganizerPlus/results/my_run/final_labels.jsonl
	 ```

	 Classification requires the taxonomy to be frozen. It may be run as a
	 separate job after discovery; use resources appropriate for the model and
	 set `--tensor-parallel-size` to the number of GPUs allocated to that job.
