from __future__ import annotations

import argparse
import re

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
        "--run-name",
        default=None,
        help="Run name substituted for ${run_name} in configured paths",
    )
    parser.add_argument(
        "--input", default=None, help="Input JSONL with doc_id and text"
    )
    parser.add_argument(
        "--taxonomy",
        default="MultiWebOrganizerPlus/results/taxonomy_state.json",
        help="Taxonomy state JSON path",
    )

    # Shared model/runtime settings.
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
        "--creativity",
        choices=("low", "high"),
        default="low",
        help="Discovery creativity level. Adjusts how often the model proposes new labels. (ignored for frozen classification)",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    parser.add_argument(
        "--compute-logging",
        action="store_true",
        help="Enable periodic compute logging (elapsed time and GPU-hours estimates)",
    )
    parser.add_argument(
        "--compute-logging-interval",
        type=float,
        default=300.0,
        help="Seconds between compute-logging messages (default: 300)",
    )

    # Discovery paths/settings.
    parser.add_argument(
        "--seed-labels",
        default="MultiWebOrganizerPlus/config/seed_labels/topics.yaml",
    )
    parser.add_argument(
        "--discovery-output",
        default="MultiWebOrganizerPlus/results/discovery.jsonl",
    )

    parser.add_argument(
        "--screen-every",
        type=int,
        default=64,
        help="Run proposal screening after this many newly persisted discovery documents",
    )
    parser.add_argument(
        "--maintenance-max-tokens",
        type=int,
        default=8192,
        help="Maximum output tokens for each maintenance sub-task",
    )
    parser.add_argument(
        "--max-screening-groups",
        type=int,
        default=32,
        help="Maximum proposal groups in one screening call",
    )
    parser.add_argument(
        "--max-promotion-groups",
        type=int,
        default=16,
        help="Maximum candidate groups in one promotion call",
    )
    parser.add_argument(
        "--max-revision-labels",
        type=int,
        default=16,
        help="Maximum dynamic labels eligible for revision in one model call",
    )
    parser.add_argument(
        "--min-promotion-support",
        type=int,
        default=2,
        help="Minimum accumulated support_count before a kept candidate can be promoted",
    )
    parser.add_argument("--max-assigned-labels-per-doc", type=int, default=5)
    parser.add_argument("--max-proposals-per-doc", type=int, default=3)
    parser.add_argument("--max-total-labels", type=int, default=100)
    parser.add_argument("--expected-seed-label-count", type=int, default=24)
    parser.add_argument(
        "--reset-discovery",
        action="store_true",
        help="Delete discovery output and taxonomy state before starting",
    )

    # Frozen classification settings.
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
    if args.mode not in {"discover", "classify"}:
        raise ValueError("mode must be 'discover' or 'classify'")
    if args.creativity not in {"low", "high"}:
        raise ValueError("creativity must be 'low' or 'high'")
    if args.run_name is not None and (
        not isinstance(args.run_name, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name)
    ):
        raise ValueError(
            "--run-name must contain only letters, numbers, '_', '-', and '.' "
            "and must start with a letter or number"
        )

    shared_string_fields = (
        "input",
        "taxonomy",
        "model",
        "aspect",
        "dtype",
        "creativity",
    )
    for field in shared_string_fields:
        value = getattr(args, field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"--{field.replace('_', '-')} must be a non-empty string")

    mode_path_fields = (
        ("seed_labels", "discovery_output") if args.mode == "discover" else ("output",)
    )
    for field in mode_path_fields:
        value = getattr(args, field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"--{field.replace('_', '-')} must be a non-empty string")

    for field in ("compute_logging", "reset_discovery", "overwrite_output"):
        value = getattr(args, field)
        if not isinstance(value, bool):
            raise ValueError(f"--{field.replace('_', '-')} must be boolean")

    if isinstance(args.seed, bool) or not isinstance(args.seed, int) or args.seed < 0:
        raise ValueError("--seed must be an integer >= 0 for reproducible runs")

    if args.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError("--log-level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")

    if args.thinking_mode not in {"disabled", "enabled", "template-default"}:
        raise ValueError(
            "--thinking-mode must be disabled, enabled, or template-default"
        )

    positive_int_fields = (
        "batch_size",
        "max_model_len",
        "max_document_tokens",
        "classification_max_tokens",
        "max_assigned_labels_per_doc",
        "tensor_parallel_size",
    )
    for field in positive_int_fields:
        value = getattr(args, field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"--{field.replace('_', '-')} must be an integer >= 1")

    if args.classification_max_tokens >= args.max_model_len:
        raise ValueError(
            "--classification-max-tokens must be smaller than --max-model-len"
        )

    if (
        isinstance(args.gpu_memory_utilization, bool)
        or not isinstance(args.gpu_memory_utilization, (int, float))
        or not (0.0 < float(args.gpu_memory_utilization) <= 1.0)
    ):
        raise ValueError("--gpu-memory-utilization must be numeric and in (0, 1]")

    if (
        isinstance(args.compute_logging_interval, bool)
        or not isinstance(args.compute_logging_interval, (int, float))
        or float(args.compute_logging_interval) <= 0.0
    ):
        raise ValueError("--compute-logging-interval must be numeric and > 0")

    if args.mode == "discover":
        for field in (
            "screen_every",
            "maintenance_max_tokens",
            "max_screening_groups",
            "max_promotion_groups",
            "max_revision_labels",
            "min_promotion_support",
            "max_proposals_per_doc",
            "max_total_labels",
            "expected_seed_label_count",
        ):
            value = getattr(args, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"--{field.replace('_', '-')} must be an integer >= 1")

        if args.maintenance_max_tokens >= args.max_model_len:
            raise ValueError(
                "--maintenance-max-tokens must be smaller than --max-model-len"
            )

        if args.max_total_labels < args.expected_seed_label_count:
            raise ValueError(
                "--max-total-labels cannot be smaller than --expected-seed-label-count"
            )


def _load_config_defaults(parser: argparse.ArgumentParser) -> dict:
    prelim_args, _ = parser.parse_known_args()

    try:
        import yaml
        from pathlib import Path

        cfg_path = Path(prelim_args.config)
        if not cfg_path.exists():
            return {}

        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("Top-level config must be a mapping")

        # Fail clearly on the old all-in-one reconciliation configuration rather
        # than reporting it as a generic unknown key.
        if "reconcile" in raw:
            raise ValueError(
                "Config section 'reconcile' is obsolete. Replace it with the new "
                "'maintenance' section (screen_every, max_screening_groups, "
                "max_promotion_groups, max_revision_labels, min_promotion_support, max_total_labels, "
                "expected_seed_label_count)."
            )
        if (
            isinstance(raw.get("model"), dict)
            and "reconcile_max_tokens" in raw["model"]
        ):
            raise ValueError(
                "Config key model.reconcile_max_tokens is obsolete; use "
                "model.maintenance_max_tokens"
            )

        allowed_top_level = {
            "mode",
            "aspect",
            "run_name",
            "model",
            "paths",
            "discovery",
            "maintenance",
            "runtime",
        }
        unknown_top_level = set(raw) - allowed_top_level
        if unknown_top_level:
            raise ValueError(
                f"Unknown top-level config keys: {sorted(unknown_top_level)}"
            )

        allowed_section_keys = {
            "model": {
                "name",
                "batch_size",
                "max_model_len",
                "max_document_tokens",
                "classification_max_tokens",
                "maintenance_max_tokens",
                "tensor_parallel_size",
                "gpu_memory_utilization",
                "dtype",
                "seed",
                "thinking_mode",
            },
            "paths": {
                "input",
                "seed_labels",
                "taxonomy",
                "discovery_output",
                "output",
            },
            "discovery": {
                "max_assigned_labels_per_doc",
                "max_proposals_per_doc",
                "creativity",
            },
            "maintenance": {
                "screen_every",
                "max_screening_groups",
                "max_promotion_groups",
                "max_revision_labels",
                "min_promotion_support",
                "max_total_labels",
                "expected_seed_label_count",
            },
            "runtime": {
                "log_level",
                "compute_logging",
                "compute_logging_interval",
                "reset_discovery",
                "overwrite_output",
            },
        }

        for section, allowed_keys in allowed_section_keys.items():
            if section not in raw:
                continue
            section_value = raw[section]
            if not isinstance(section_value, dict):
                raise ValueError(f"Config section {section!r} must be a mapping")
            unknown_keys = set(section_value) - allowed_keys
            if unknown_keys:
                raise ValueError(
                    f"Unknown keys in config section {section!r}: "
                    f"{sorted(unknown_keys)}"
                )

        config_defaults: dict = {}
        if "mode" in raw:
            config_defaults["mode"] = raw["mode"]
        if "aspect" in raw:
            config_defaults["aspect"] = raw["aspect"]
        if "run_name" in raw and prelim_args.run_name is None:
            config_defaults["run_name"] = raw["run_name"]

        run_name = prelim_args.run_name or raw.get("run_name")
        if run_name is not None and (
            not isinstance(run_name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name)
        ):
            raise ValueError(
                "run_name must contain only letters, numbers, '_', '-', and '.' "
                "and must start with a letter or number"
            )

        mapping = {
            ("model", "name"): "model",
            ("model", "batch_size"): "batch_size",
            ("model", "max_model_len"): "max_model_len",
            ("model", "max_document_tokens"): "max_document_tokens",
            ("model", "classification_max_tokens"): "classification_max_tokens",
            ("model", "maintenance_max_tokens"): "maintenance_max_tokens",
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
            ("discovery", "max_assigned_labels_per_doc"): "max_assigned_labels_per_doc",
            ("discovery", "max_proposals_per_doc"): "max_proposals_per_doc",
            ("discovery", "creativity"): "creativity",
            ("maintenance", "screen_every"): "screen_every",
            ("maintenance", "max_screening_groups"): "max_screening_groups",
            ("maintenance", "max_promotion_groups"): "max_promotion_groups",
            ("maintenance", "max_revision_labels"): "max_revision_labels",
            ("maintenance", "min_promotion_support"): "min_promotion_support",
            ("maintenance", "max_total_labels"): "max_total_labels",
            ("maintenance", "expected_seed_label_count"): "expected_seed_label_count",
            ("runtime", "log_level"): "log_level",
            ("runtime", "compute_logging"): "compute_logging",
            ("runtime", "compute_logging_interval"): "compute_logging_interval",
            ("runtime", "reset_discovery"): "reset_discovery",
            ("runtime", "overwrite_output"): "overwrite_output",
        }

        for (section, key), dest in mapping.items():
            section_value = raw.get(section)
            if isinstance(section_value, dict) and key in section_value:
                value = section_value[key]
                if section == "paths" and isinstance(value, str):
                    if "${run_name}" in value:
                        if run_name is None:
                            raise ValueError(
                                f"paths.{key} uses ${{run_name}}, but run_name is not configured"
                            )
                        value = value.replace("${run_name}", run_name)
                    if "${" in value:
                        raise ValueError(
                            f"paths.{key} contains an unresolved config placeholder"
                        )
                config_defaults[dest] = value

        return config_defaults

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load config file {prelim_args.config}: {exc}"
        ) from exc


def main() -> None:
    parser = build_arg_parser()
    config_defaults = _load_config_defaults(parser)

    # Config-derived values are defaults so explicit CLI flags always win.
    if config_defaults:
        parser.set_defaults(**config_defaults)

    args = parser.parse_args()

    log_level = getattr(args, "log_level", "INFO") or "INFO"
    configure_logging(log_level)

    validate_cli_args(args)

    LOGGER.info("Run args:")
    for key, value in vars(args).items():
        LOGGER.info("  %s: %s", key, value)

    LOGGER.info("Using vLLM %s", vllm.__version__)
    if args.mode == "discover":
        run_discovery(args)
    elif args.mode == "classify":
        run_classification(args)
    else:  # pragma: no cover - argparse/validation make this unreachable.
        raise AssertionError(args.mode)
