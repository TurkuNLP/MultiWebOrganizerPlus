from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterator

import vllm  # type: ignore

from label_pipeline_lib.common import (
    LOGGER,
    active_labels,
    append_models_jsonl,
    atomic_write_model,
    batch_iter,
    file_fingerprint,
    iter_jsonl_models,
    load_model_file,
    stream_documents,
    taxonomy_hash,
    validate_taxonomy_state,
)
from label_pipeline_lib.model import ModelClient
from schemas import (
    ClassificationRunMetadata,
    FinalClassificationRecord,
    InputDocument,
    TaxonomyState,
)


def metadata_path_for_output(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".meta.json")


def build_classification_metadata(
    args: argparse.Namespace,
    input_path: Path,
    taxonomy_path: Path,
    state: TaxonomyState,
) -> ClassificationRunMetadata:
    resolved, size, mtime_ns = file_fingerprint(input_path)
    return ClassificationRunMetadata(
        input_path=resolved,
        input_size_bytes=size,
        input_mtime_ns=mtime_ns,
        taxonomy_path=str(taxonomy_path.resolve()),
        taxonomy_version=state.schema_version,
        taxonomy_hash=taxonomy_hash(state),
        model_name=args.model,
        vllm_version=vllm.__version__,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        max_document_tokens=args.max_document_tokens,
        classification_max_tokens=args.classification_max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        seed=args.seed,
        thinking_mode=args.thinking_mode,
        created_at_unix=int(time.time()),
    )


def ensure_classification_metadata(
    path: Path,
    expected: ClassificationRunMetadata,
) -> None:
    if not path.exists():
        atomic_write_model(path, expected)
        return
    existing = load_model_file(path, ClassificationRunMetadata)
    existing_cmp = existing.model_dump(exclude={"created_at_unix"})
    expected_cmp = expected.model_dump(exclude={"created_at_unix"})
    if existing_cmp != expected_cmp:
        raise ValueError(
            "Classification run metadata does not match existing output. "
            "Use --overwrite-output for a deliberate fresh run."
        )


def load_result_index(path: Path, model_type: type) -> tuple[set[str], int]:
    ids: set[str] = set()
    expected_seq = 1
    for record in iter_jsonl_models(path, model_type):
        seq = getattr(record, "seq")
        doc_id = getattr(record, "doc_id")
        if seq != expected_seq:
            raise ValueError(
                f"Non-contiguous sequence in {path}: expected {expected_seq}, got {seq}"
            )
        expected_seq += 1
        if doc_id in ids:
            raise ValueError(f"Duplicate doc_id {doc_id!r} in {path}")
        ids.add(doc_id)
    return ids, expected_seq - 1


def run_classification(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    taxonomy_path = Path(args.taxonomy)
    output_path = Path(args.output)
    meta_path = metadata_path_for_output(output_path)

    state = load_model_file(taxonomy_path, TaxonomyState)
    validate_taxonomy_state(state)
    if not state.frozen:
        raise ValueError(
            f"Taxonomy {taxonomy_path} is not frozen. Complete --mode discover first."
        )

    if args.overwrite_output:
        for path in (output_path, meta_path):
            if path.exists():
                path.unlink()

    if output_path.exists() and not meta_path.exists():
        raise ValueError(
            f"Existing classification output {output_path} has no metadata file "
            f"{meta_path}. Use --overwrite-output for a fresh run."
        )

    metadata = build_classification_metadata(args, input_path, taxonomy_path, state)
    ensure_classification_metadata(meta_path, metadata)

    completed_ids, last_seq = load_result_index(output_path, FinalClassificationRecord)
    for record in iter_jsonl_models(output_path, FinalClassificationRecord):
        if record.taxonomy_version != state.schema_version:
            raise ValueError(
                f"Existing output record {record.doc_id} uses taxonomy version "
                f"{record.taxonomy_version}, expected {state.schema_version}"
            )
        if record.taxonomy_hash != state.frozen_taxonomy_hash:
            raise ValueError(
                f"Existing output record {record.doc_id} has a different taxonomy hash"
            )

    labels_snapshot = active_labels(state)
    model = ModelClient(
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        seed=args.seed,
        thinking_mode=args.thinking_mode,
    )

    seen_input_ids: set[str] = set()
    next_seq = last_seq + 1
    processed_this_run = 0
    total_input = 0

    def pending_docs() -> Iterator[InputDocument]:
        nonlocal total_input
        for doc in stream_documents(input_path):
            total_input += 1
            if doc.doc_id in seen_input_ids:
                raise ValueError(f"Duplicate doc_id {doc.doc_id!r} in input corpus")
            seen_input_ids.add(doc.doc_id)
            if doc.doc_id in completed_ids:
                continue
            yield doc

    for batch in batch_iter(pending_docs(), args.batch_size):
        outputs = model.classify_batch(
            batch,
            labels_snapshot,
            aspect=state.aspect,
            max_document_tokens=args.max_document_tokens,
            output_max_tokens=args.classification_max_tokens,
            max_assigned_labels_per_doc=args.max_assigned_labels_per_doc,
        )
        if len(outputs) != len(batch):
            raise RuntimeError(
                f"Expected {len(batch)} validated outputs, got {len(outputs)}"
            )

        records = [
            FinalClassificationRecord(
                seq=next_seq + index,
                doc_id=doc.doc_id,
                taxonomy_version=state.schema_version,
                taxonomy_hash=state.frozen_taxonomy_hash or taxonomy_hash(state),
                assigned_label_ids=output.assigned_label_ids,
            )
            for index, (doc, output) in enumerate(zip(batch, outputs, strict=True))
        ]
        append_models_jsonl(output_path, records)
        for record in records:
            completed_ids.add(record.doc_id)
        next_seq += len(records)
        processed_this_run += len(records)
        LOGGER.info(
            "Classification: persisted %d documents this run; latest seq=%d",
            processed_this_run,
            next_seq - 1,
        )

    if total_input == 0:
        raise ValueError("Input corpus contains no documents")

    LOGGER.info(
        "Frozen classification complete. %d documents processed this run; "
        "%d total output records.",
        processed_this_run,
        len(completed_ids),
    )
