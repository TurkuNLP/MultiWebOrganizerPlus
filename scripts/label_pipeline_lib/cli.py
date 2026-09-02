from __future__ import annotations

import argparse

import vllm  # type: ignore

from label_pipeline_lib.classification import run_classification
from label_pipeline_lib.common import LOGGER, configure_logging
from label_pipeline_lib.discovery import run_discovery


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover an evolving taxonomy or classify against a frozen one."
    )
    parser.add_argument(
        "--config",
        default="configs/pipeline_defaults.yaml",
        help="Path to YAML config file with defaults",
    )
    parser.add_argument("--mode", default=None, choices=("discover", "classify"))
    parser.add_argument(
        "--input", default=None, help="Input JSONL with doc_id and text"
    )
    parser.add_argument(
        "--taxonomy",
        default="MultiWebOrganizerPlus/results/taxonomy_state.json",
        help="Taxonomy state JSON path",
    )

    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-document-tokens", type=int, default=8192)
    parser.add_argument("--classification-max-tokens", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--thinking-mode",
        choices=("disabled", "enabled", "template-default"),
        default="disabled",
        help=(
            "Qwen thinking behavior. 'disabled' passes enable_thinking=False to "
            "the chat template; 'template-default' passes no override."
        ),
    )

    parser.add_argument("--aspect", default="topics")
    parser.add_argument(
        "--seed-labels",
        default="MultiWebOrganizerPlus/config/seed_labels/topics.yaml",
    )
    parser.add_argument(
        "--discovery-output",
        default="MultiWebOrganizerPlus/results/discovery.jsonl",
    )
    parser.add_argument("--reconcile-every", type=int, default=64)
    parser.add_argument("--reconcile-max-tokens", type=int, default=8192)
    parser.add_argument("--max-reconcile-groups", type=int, default=128)
    parser.add_argument("--min-create-support", type=int, default=2)
    parser.add_argument("--max-assigned-labels-per-doc", type=int, default=5)
    parser.add_argument("--max-proposals-per-doc", type=int, default=3)
    parser.add_argument("--max-total-labels", type=int, default=100)
    parser.add_argument("--expected-seed-label-count", type=int, default=24)
    parser.add_argument(
        "--reset-discovery",
        action="store_true",
        help="Delete discovery output and taxonomy state before starting",
    )

    parser.add_argument(
        "--output",
        default="MultiWebOrganizerPlus/results/final_labels.jsonl",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete existing final output and its metadata before classifying",
    )
    return parser


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.mode is None:
        raise ValueError("mode must be specified via --mode or config")

    if args.input is None:
        raise ValueError("input must be specified via --input or config")

    positive_int_fields = (
        "batch_size",
        "max_model_len",
        "max_document_tokens",
        "classification_max_tokens",
        "tensor_parallel_size",
    )
    for field in positive_int_fields:
        if getattr(args, field) < 1:
            raise ValueError(f"--{field.replace('_', '-')} must be >= 1")

    if not (0.0 < args.gpu_memory_utilization <= 1.0):
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")

    if args.mode == "discover":
        for field in (
            "reconcile_every",
            "reconcile_max_tokens",
            "max_reconcile_groups",
            "min_create_support",
            "max_proposals_per_doc",
            "max_total_labels",
            "expected_seed_label_count",
        ):
            if getattr(args, field) < 1:
                raise ValueError(f"--{field.replace('_', '-')} must be >= 1")
        if args.max_total_labels < args.expected_seed_label_count:
            raise ValueError(
                "--max-total-labels cannot be smaller than --expected-seed-label-count"
            )


def main() -> None:
    # First parse only to get --config (if provided), then load YAML defaults and reparse.
    parser = build_arg_parser()
    prelim_args, _ = parser.parse_known_args()

    # Load config file if it exists and is readable.
    config_defaults: dict = {}
    try:
        import yaml
        from pathlib import Path

        cfg_path = Path(prelim_args.config)
        if cfg_path.exists():
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

            # Map top-level config keys to argparse names
            if "mode" in raw:
                config_defaults["mode"] = raw["mode"]
            # Map nested config keys to argparse names
            mapping = {
                ("model", "name"): "model",
                ("model", "batch_size"): "batch_size",
                ("model", "max_model_len"): "max_model_len",
                ("model", "max_document_tokens"): "max_document_tokens",
                ("model", "classification_max_tokens"): "classification_max_tokens",
                ("model", "reconcile_max_tokens"): "reconcile_max_tokens",
                ("model", "tensor_parallel_size"): "tensor_parallel_size",
                ("model", "gpu_memory_utilization"): "gpu_memory_utilization",
                ("model", "dtype"): "dtype",
                ("model", "seed"): "seed",
                ("model", "thinking_mode"): "thinking_mode",
                ("paths", "input"): "input",
                ("paths", "seed_labels"): "seed_labels",
                ("paths", "taxonomy"): "taxonomy",
                ("paths", "discovery_output"): "discovery_output",
                ("paths", "output"): "output",
                (
                    "discovery",
                    "max_assigned_labels_per_doc",
                ): "max_assigned_labels_per_doc",
                ("discovery", "max_proposals_per_doc"): "max_proposals_per_doc",
                ("reconcile", "reconcile_every"): "reconcile_every",
                ("reconcile", "max_reconcile_groups"): "max_reconcile_groups",
                ("reconcile", "min_create_support"): "min_create_support",
                ("reconcile", "max_total_labels"): "max_total_labels",
                ("reconcile", "expected_seed_label_count"): "expected_seed_label_count",
                ("runtime", "log_level"): "log_level",
                ("runtime", "reset_discovery"): "reset_discovery",
                ("runtime", "overwrite_output"): "overwrite_output",
            }

            for (section, key), dest in mapping.items():
                if section in raw and key in raw[section]:
                    config_defaults[dest] = raw[section][key]

    except Exception:
        raise RuntimeError(f"Failed to load config file {prelim_args.config}")

    # Apply config-derived defaults to the parser so CLI flags override them.
    if config_defaults:
        parser.set_defaults(**config_defaults)

    args = parser.parse_args()

    # Configure logging according to config / CLI override (default INFO)
    log_level = getattr(args, "log_level", "INFO") or "INFO"
    configure_logging(log_level)

    validate_cli_args(args)

    LOGGER.info("Run args:")
    for k, v in vars(args).items():
        LOGGER.info("  %s: %s", k, v)

    LOGGER.info("Using vLLM %s", vllm.__version__)
    if args.mode == "discover":
        run_discovery(args)
    elif args.mode == "classify":
        run_classification(args)
    else:
        raise AssertionError(args.mode)
