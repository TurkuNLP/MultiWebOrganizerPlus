from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, TypeVar

import yaml
from pydantic import BaseModel, ValidationError  # type: ignore

from label_pipeline_lib.schemas import InputDocument, LabelDef, TaxonomyState

LOGGER = logging.getLogger("label_pipeline")
T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)


def configure_logging(level: str = "DEBUG") -> None:
    """Configure root logging from a standard logging level name."""
    if not isinstance(level, str):
        raise TypeError("logging level must be a string")

    normalized = level.upper()
    lvl = getattr(logging, normalized, None)
    if not isinstance(lvl, int):
        raise ValueError(f"Unknown logging level: {level!r}")

    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_model(path: Path, model: BaseModel) -> None:
    atomic_write_text(path, model.model_dump_json(indent=2) + "\n")


def load_model_file(path: Path, model_type: type[M]) -> M:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = path.read_text(encoding="utf-8")
    try:
        return model_type.model_validate_json(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid {model_type.__name__} in {path}: {exc}") from exc


def append_models_jsonl(path: Path, records: Sequence[BaseModel]) -> None:
    """Append a validated batch and fsync before returning."""
    if not records:
        return
    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def iter_jsonl_models(path: Path, model_type: type[M]) -> Iterator[M]:
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Empty JSONL line in {path}:{line_no}")
            try:
                yield model_type.model_validate_json(line)
            except ValidationError as exc:
                raise ValueError(
                    f"Invalid {model_type.__name__} in {path}:{line_no}: {exc}"
                ) from exc


def stream_documents(path: Path) -> Iterator[InputDocument]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Empty JSONL line in {path}:{line_no}")
            try:
                yield InputDocument.model_validate_json(line)
            except ValidationError as exc:
                raise ValueError(
                    f"Invalid input record in {path}:{line_no}: {exc}"
                ) from exc


def count_jsonl_records(path: Path) -> int:
    """Count JSONL records without parsing each record into a model."""
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "rb") as handle:
        return sum(1 for _ in handle)


def batch_iter(iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    if size < 1:
        raise ValueError("batch size must be >= 1")
    batch: list[T] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def file_fingerprint(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path.resolve()), stat.st_size, stat.st_mtime_ns


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def seed_labels_hash(labels: Sequence[LabelDef]) -> str:
    payload = [label.model_dump(mode="json") for label in labels]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def taxonomy_hash(state: TaxonomyState) -> str:
    """Hash only the effective taxonomy, not discovery/audit state."""
    payload = {
        "aspect": state.aspect,
        "seed_labels": [label.model_dump(mode="json") for label in state.seed_labels],
        "dynamic_labels": [
            {
                "id": label.id,
                "name": label.name,
                "definition": label.definition,
            }
            for label in state.dynamic_labels
        ],
        "alias_map": state.alias_map,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def validate_unique_labels(labels: Sequence[LabelDef], *, context: str) -> None:
    ids = [label.id for label in labels]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate label IDs in {context}")

    normalized_names = [normalize_text(label.name) for label in labels]
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError(f"Duplicate label names in {context}")


def load_seed_labels(path: Path, expected_count: int) -> list[LabelDef]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, list):
        raise TypeError(f"Seed label file must contain a YAML list: {path}")

    try:
        labels = [LabelDef.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise ValueError(f"Invalid seed label definition in {path}: {exc}") from exc

    if len(labels) != expected_count:
        raise ValueError(
            f"Expected {expected_count} seed labels, found {len(labels)} in {path}"
        )
    validate_unique_labels(labels, context=str(path))
    return labels


def active_labels(state: TaxonomyState) -> list[LabelDef]:
    labels: list[LabelDef] = [*state.seed_labels, *state.dynamic_labels]
    validate_unique_labels(labels, context="active taxonomy")
    return labels


def resolve_alias(label_id: str, alias_map: dict[str, str]) -> str:
    seen: set[str] = set()
    current = label_id
    while current in alias_map:
        if current in seen:
            raise ValueError(f"Alias cycle detected at label {current}")
        seen.add(current)
        current = alias_map[current]
    return current


def validate_taxonomy_state(
    state: TaxonomyState,
    *,
    max_total_labels: int | None = None,
    expected_seed_count: int | None = None,
) -> None:
    """Validate state-dependent invariants not expressible in Pydantic fields."""
    if max_total_labels is None:
        max_total_labels = state.discovery_settings.max_total_labels
    if expected_seed_count is None:
        expected_seed_count = state.discovery_settings.expected_seed_label_count

    if len(state.seed_labels) != expected_seed_count:
        raise ValueError(
            f"Taxonomy contains {len(state.seed_labels)} seed labels; "
            f"expected {expected_seed_count}"
        )

    labels = active_labels(state)
    active_ids = {label.id for label in labels}

    for label in state.dynamic_labels:
        if label.created_version < 2:
            raise ValueError(
                f"Dynamic label {label.id} has impossible created_version "
                f"{label.created_version}; dynamic labels are first created at schema version 2"
            )
        if label.created_version > label.last_modified_version:
            raise ValueError(
                f"Dynamic label {label.id} has created_version "
                f"{label.created_version} after last_modified_version "
                f"{label.last_modified_version}"
            )
        if label.last_modified_version > state.schema_version:
            raise ValueError(
                f"Dynamic label {label.id} has last_modified_version "
                f"{label.last_modified_version} newer than taxonomy schema_version "
                f"{state.schema_version}"
            )

    if len(labels) > max_total_labels:
        raise ValueError(
            f"Taxonomy has {len(labels)} active labels, exceeding cap {max_total_labels}"
        )

    # The ID allocator is monotonic. A rolled-back counter could otherwise reuse
    # a previously active or retired dynamic ID after state corruption.
    numeric_used_ids = [
        int(label_id[1:])
        for label_id in active_ids | set(state.alias_map)
        if label_id.startswith("L") and label_id[1:].isdigit()
    ]
    if numeric_used_ids and state.next_dynamic_label_num <= max(numeric_used_ids):
        raise ValueError(
            "next_dynamic_label_num must be greater than every active or retired "
            f"numeric L### ID; got {state.next_dynamic_label_num}, "
            f"maximum used is {max(numeric_used_ids)}"
        )

    # Every alias source must be retired and every chain must terminate at an
    # active label. resolve_alias() also detects cycles.
    for source, target in state.alias_map.items():
        if source in active_ids:
            raise ValueError(f"Alias source {source} is still an active label")
        resolved = resolve_alias(target, state.alias_map)
        if resolved not in active_ids:
            raise ValueError(
                f"Alias {source}->{target} does not resolve to an active label"
            )

    # Candidate proposals are discovery evidence, not active labels. They may
    # remain after freezing, but their persistent IDs and exact normalized
    # signatures must be unique.
    candidate_ids = [group.group_id for group in state.candidate_proposals]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Duplicate candidate proposal group IDs in taxonomy state")

    candidate_signatures = [
        (normalize_text(group.name), normalize_text(group.definition))
        for group in state.candidate_proposals
    ]
    if len(candidate_signatures) != len(set(candidate_signatures)):
        raise ValueError(
            "Duplicate normalized candidate proposal signatures in taxonomy state"
        )

    expected_hash = seed_labels_hash(state.seed_labels)
    if expected_hash != state.seed_labels_sha256:
        raise ValueError(
            "Seed-label hash stored in taxonomy does not match seed labels"
        )

    if state.frozen:
        if state.frozen_at_unix is None or state.frozen_taxonomy_hash is None:
            raise ValueError("Frozen taxonomy is missing frozen metadata")
        if taxonomy_hash(state) != state.frozen_taxonomy_hash:
            raise ValueError("Frozen taxonomy hash does not match taxonomy contents")
    elif state.frozen_at_unix is not None or state.frozen_taxonomy_hash is not None:
        raise ValueError("Unfrozen taxonomy must not contain frozen metadata")


def next_dynamic_id(state: TaxonomyState) -> str:
    used = {label.id for label in active_labels(state)} | set(state.alias_map)
    while True:
        candidate = f"L{state.next_dynamic_label_num:03d}"
        state.next_dynamic_label_num += 1
        if candidate not in used:
            return candidate
