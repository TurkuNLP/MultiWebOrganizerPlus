from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
import torch  # type: ignore
import json
import argparse
import os
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any
import logging

LOGGER = logging.getLogger(__name__)
SKIP_LOG_INTERVAL = 10_000


def visible_gpu_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


@dataclass
class Document:
    fields: dict[str, Any]

    @property
    def text(self) -> str:
        return self.fields["text"]

    @property
    def url(self) -> str:
        return self.fields.get("url", "")


@dataclass
class ClassifiedDocument(Document):
    predicted_label: str

    def to_record(self, compact: bool = False) -> dict[str, Any]:
        if compact:
            record = {
                field: self.fields[field]
                for field in ("doc_id", "warc_record_id")
                if field in self.fields
            }
            record["predicted_label"] = self.predicted_label
            return record
        record = dict(self.fields)
        record["predicted_label"] = self.predicted_label
        return record


def process_rank() -> tuple[int, int, int]:
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", "1")))
    local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", "0")))
    return rank, world_size, local_rank


def validate_document_fields(fields: dict[str, Any], source: str) -> None:
    missing_fields = []
    if "text" not in fields:
        missing_fields.append("text")
    if "doc_id" not in fields and "warc_record_id" not in fields:
        missing_fields.append("doc_id or warc_record_id")
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"Each {source} must contain required field(s): {missing}")


def stream_jsonl_file(
    file_path: str,
    batch_size: int = 128,
    skip_documents: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[list[Document]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if skip_documents < 0:
        raise ValueError("skip_documents must not be negative")

    total_skip_documents = skip_documents
    skipped_documents = 0
    if total_skip_documents:
        LOGGER.info(
            "Skipping %d existing documents before classification",
            total_skip_documents,
        )

    with open(file_path, "r", encoding="utf-8") as f:
        batch = []
        source_index = 0
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            if not isinstance(doc, dict):
                raise ValueError("Each JSONL record must be a JSON object")
            validate_document_fields(doc, "JSONL record")
            selected = source_index % world_size == rank
            source_index += 1
            if not selected:
                continue
            if skip_documents:
                skip_documents -= 1
                skipped_documents += 1
                if skipped_documents % SKIP_LOG_INTERVAL == 0 or skip_documents == 0:
                    LOGGER.info(
                        "Skipped %d/%d existing documents",
                        skipped_documents,
                        total_skip_documents,
                    )
                continue
            batch.append(Document(fields=doc))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def stream_huggingface_dataset(
    dataset_name: str,
    dataset_config: str,
    dataset_split: str,
    batch_size: int = 128,
    skip_documents: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[list[Document]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "The 'datasets' package is required for --dataset input"
        ) from error

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if skip_documents < 0:
        raise ValueError("skip_documents must not be negative")

    total_skip_documents = skip_documents
    skipped_documents = 0
    if total_skip_documents:
        LOGGER.info(
            "Skipping %d existing documents before classification",
            total_skip_documents,
        )

    LOGGER.info(
        "Streaming Hugging Face dataset %s/%s [%s]",
        dataset_name,
        dataset_config,
        dataset_split,
    )
    rows = load_dataset(
        dataset_name,
        name=dataset_config,
        split=dataset_split,
        streaming=True,
    )
    batch = []
    for source_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("Each Hugging Face dataset row must be a mapping")
        if source_index % world_size != rank:
            continue
        if skip_documents:
            skip_documents -= 1
            skipped_documents += 1
            if skipped_documents % SKIP_LOG_INTERVAL == 0 or skip_documents == 0:
                LOGGER.info(
                    "Skipped %d/%d existing documents",
                    skipped_documents,
                    total_skip_documents,
                )
            continue
        validate_document_fields(row, "Hugging Face dataset row")
        batch.append(Document(fields=dict(row)))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def length_aware_batches(
    document_batches: Iterator[list[Document]],
    batch_size: int,
    bucket_size: int,
) -> Iterator[list[Document]]:
    if bucket_size < batch_size:
        raise ValueError("length bucket size must be at least batch size")
    buffer: list[Document] = []
    for documents in document_batches:
        buffer.extend(documents)
        while len(buffer) >= bucket_size:
            buffer.sort(key=lambda document: len(document.text))
            while len(buffer) >= batch_size:
                yield buffer[:batch_size]
                del buffer[:batch_size]
    if buffer:
        buffer.sort(key=lambda document: len(document.text))
        while buffer:
            yield buffer[:batch_size]
            del buffer[:batch_size]


def count_output_records(file_path: str) -> int:
    """Count complete JSONL records without parsing every historical record."""
    with open(file_path, "rb") as f:
        record_count = sum(
            chunk.count(b"\n") for chunk in iter(lambda: f.read(8 * 1024 * 1024), b"")
        )
        f.seek(0, os.SEEK_END)
        if f.tell():
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                raise ValueError("Cannot resume: output ends with an incomplete record")
    return record_count


def load_progress(progress_path: str) -> int | None:
    if not os.path.isfile(progress_path):
        return None
    with open(progress_path, "r", encoding="utf-8") as f:
        progress = json.load(f)
    completed_count = progress.get("completed_count")
    if not isinstance(completed_count, int) or completed_count < 0:
        raise ValueError(f"Invalid resume checkpoint: {progress_path}")
    return completed_count


def save_progress(progress_path: str, completed_count: int) -> None:
    temporary_path = f"{progress_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump({"completed_count": completed_count}, f)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary_path, progress_path)


def load_model_and_tokenizer(
    aspect: str = "topic",
    with_url: bool = False,
    use_memory_efficient_attention: bool = False,
    use_bfloat16: bool = False,
    local_rank: int = 0,
):
    if not aspect in ["topic", "format"]:
        raise ValueError(f"Invalid aspect: {aspect}. Must be 'topic' or 'format'.")

    classifier_names = {
        "topic": {
            "no_url": "WebOrganizer/TopicClassifier-NoURL",
            "with_url": "WebOrganizer/TopicClassifier",
        },
        "format": {
            "no_url": "WebOrganizer/FormatClassifier-NoURL",
            "with_url": "WebOrganizer/FormatClassifier",
        },
    }

    classifier_name = (
        classifier_names[aspect]["with_url"]
        if with_url
        else classifier_names[aspect]["no_url"]
    )

    LOGGER.info("Loading %s", classifier_name)
    tokenizer = AutoTokenizer.from_pretrained(
        classifier_name,
        trust_remote_code=True,
    )
    load_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": "eager",
        "use_memory_efficient_attention": False,
    }
    if use_memory_efficient_attention:
        load_kwargs.update(
            unpad_inputs=True,
            use_memory_efficient_attention=True,
        )
    if torch.cuda.is_available() and use_bfloat16:
        load_kwargs["dtype"] = torch.bfloat16
    LOGGER.info("Loading model weights")
    model = AutoModelForSequenceClassification.from_pretrained(
        classifier_name,
        **load_kwargs,
    )
    # Slurm/ROCR masks each rank to its assigned GPU, so that GPU is always
    # local device zero inside the process.
    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )
    LOGGER.info("Moving model to %s", device)
    model.to(device)
    LOGGER.info("Model moved to %s", device)
    model.eval()
    LOGGER.info("Using rank-local device %s", device)
    LOGGER.info("Model ready on %s", next(model.parameters()).device)
    return tokenizer, model


def classify_batch(
    documents: Sequence[Document],
    tokenizer,
    model,
    with_url: bool = False,
    max_length: int = 1024,
) -> list[ClassifiedDocument]:
    if max_length < 1:
        raise ValueError("max_length must be at least 1")
    LOGGER.debug("Tokenizing batch of %d documents", len(documents))
    tokenization_kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "max_length": max_length,
    }
    texts = [f"{doc.url}\n\n{doc.text}" if with_url else doc.text for doc in documents]
    inputs = tokenizer(
        texts,
        **tokenization_kwargs,
    )
    device = next(model.parameters()).device
    LOGGER.debug("Moving batch inputs to %s", device)
    inputs = {name: value.to(device) for name, value in inputs.items()}
    LOGGER.debug("Running inference for batch of %d documents", len(documents))
    with torch.inference_mode():
        outputs = model(**inputs)
    LOGGER.debug("Inference completed for batch of %d documents", len(documents))

    predicted_labels = outputs.logits.argmax(dim=-1).tolist()
    id_to_label = getattr(model.config, "id2label", {})

    classified_documents = []
    for doc, label in zip(documents, predicted_labels):
        predicted_label = id_to_label.get(
            label, id_to_label.get(str(label), str(label))
        )
        classified_documents.append(
            ClassifiedDocument(
                fields=dict(doc.fields),
                predicted_label=predicted_label,
            )
        )

    return classified_documents


def arguments():
    parser = argparse.ArgumentParser(
        description="Classify documents using a pre-trained model."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--jsonl-input", help="Path to the input JSONL file.")
    input_group.add_argument(
        "--dataset-input", help="Hugging Face dataset repository to stream."
    )
    parser.add_argument(
        "--dataset-config",
        default="eng_Latn",
        help="Hugging Face dataset configuration (default: eng_Latn).",
    )
    parser.add_argument(
        "--dataset-split",
        choices=["eng_all", "parallel", "additional"],
        default="parallel",
        help="Hugging Face dataset split (default: parallel).",
    )
    parser.add_argument(
        "--output", required=True, help="Path to the output JSONL file."
    )
    parser.add_argument(
        "--compact-output",
        action="store_true",
        help="Write only doc_id/warc_record_id and predicted_label.",
    )
    parser.add_argument(
        "--length-aware-batching",
        action="store_true",
        help="Group nearby documents by text length to reduce padding.",
    )
    parser.add_argument(
        "--length-bucket-size",
        type=int,
        default=0,
        help="Buffered documents for length-aware batching; 0 means 8x batch size.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume by skipping records already present in the output JSONL.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output JSONL file if it already exists.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Global batch size across all GPUs (default: 128).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum tokenizer length per document (default: 2048).",
    )
    parser.add_argument(
        "--aspect",
        choices=["topic", "format"],
        default="topic",
        help="Aspect to classify: 'topic' or 'format'.",
    )
    parser.add_argument(
        "--with-url",
        action="store_true",
        help="Use model that considers URL in classification.",
    )
    parser.add_argument(
        "--log-every-batches",
        type=int,
        default=10,
        help="Log progress after this many batches (default: 10).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--memory-efficient-attention",
        action="store_true",
        help="Enable optional memory-efficient attention kernels.",
    )
    parser.add_argument(
        "--bfloat16",
        action="store_true",
        help="Load the model in bfloat16 instead of float32.",
    )
    return parser.parse_args()


def main():
    args = arguments()
    if args.log_every_batches < 1:
        raise ValueError("log-every-batches must be at least 1")
    if args.batch_size < 1:
        raise ValueError("batch-size must be at least 1")
    if args.max_length < 1:
        raise ValueError("max-length must be at least 1")
    if args.length_bucket_size < 0:
        raise ValueError("length-bucket-size must not be negative")
    if args.jsonl_input and args.dataset_input:
        raise ValueError("Cannot specify both --jsonl-input and --dataset-input")
    rank, world_size, local_rank = process_rank()
    output_path = f"{args.output}.rank{rank}" if world_size > 1 else args.output
    progress_path = f"{output_path}.progress.json"
    if args.resume and not os.path.isfile(output_path):
        raise ValueError(f"Cannot resume: output file does not exist: {output_path}")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    LOGGER.info("Starting classification:")
    for key, value in vars(args).items():
        LOGGER.info("  %s: %s", key, value)
    gpu_count = visible_gpu_count()
    LOGGER.info(
        "Rank %d/%d using local rank %d; detected %d visible GPU(s)",
        rank,
        world_size,
        local_rank,
        gpu_count,
    )
    if world_size > 1 and gpu_count != 1:
        LOGGER.warning("Expected one visible GPU per rank, found %d", gpu_count)
    start_time = time.perf_counter()

    # Resume or overwrite logic
    if args.resume and args.overwrite:
        raise ValueError("Cannot use both --resume and --overwrite at the same time.")
    if args.resume:
        LOGGER.info(
            "Attempting to resume from existing checkpoint; checking progress file: %s",
            progress_path,
        )
        completed_count = load_progress(progress_path)
        if completed_count is None:
            LOGGER.info("No checkpoint found; counting existing output records")
            completed_count = count_output_records(output_path)
            save_progress(progress_path, completed_count)
        else:
            LOGGER.info("Loaded checkpoint from %s", progress_path)
    else:
        completed_count = 0
    if completed_count > 0 and not args.resume and not args.overwrite:
        raise ValueError(
            f"Output file already exists with {completed_count} records: {output_path}. "
            "Use --resume to continue or --overwrite to replace it."
        )
    if completed_count > 0 and args.resume:
        LOGGER.info("Resuming after %d completed documents", completed_count)
    elif completed_count > 0 and args.overwrite:
        LOGGER.warning("Overwriting existing output file: %s", output_path)

    batch_count = 0
    document_count = completed_count
    tokenizer, model = load_model_and_tokenizer(
        aspect=args.aspect,
        with_url=args.with_url,
        use_memory_efficient_attention=args.memory_efficient_attention,
        use_bfloat16=args.bfloat16,
        local_rank=local_rank,
    )
    processing_start = time.perf_counter()
    last_log_time = processing_start
    last_log_count = 0

    if args.dataset_input:
        document_batches = stream_huggingface_dataset(
            args.dataset_input,
            args.dataset_config,
            args.dataset_split,
            batch_size=args.batch_size,
            skip_documents=completed_count,
            rank=rank,
            world_size=world_size,
        )
    else:
        document_batches = stream_jsonl_file(
            args.jsonl_input,
            batch_size=args.batch_size,
            skip_documents=completed_count,
            rank=rank,
            world_size=world_size,
        )

    if args.length_aware_batching:
        bucket_size = args.length_bucket_size or args.batch_size * 8
        document_batches = length_aware_batches(
            document_batches,
            batch_size=args.batch_size,
            bucket_size=bucket_size,
        )
        LOGGER.info("Length-aware batching enabled with bucket size %d", bucket_size)

    output_mode = "a" if args.resume else "w"
    with open(output_path, output_mode, encoding="utf-8") as f:
        for documents in document_batches:
            classified_documents = classify_batch(
                documents,
                tokenizer,
                model,
                with_url=args.with_url,
                max_length=args.max_length,
            )
            for doc in classified_documents:
                f.write(
                    json.dumps(
                        doc.to_record(compact=args.compact_output),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            f.flush()
            batch_count += 1
            document_count += len(documents)
            save_progress(progress_path, document_count)
            if batch_count % args.log_every_batches == 0:
                now = time.perf_counter()
                elapsed = now - processing_start
                window_elapsed = now - last_log_time
                processed_count = document_count - completed_count
                window_count = processed_count - last_log_count
                cumulative_rate = processed_count / elapsed if elapsed else 0.0
                window_rate = window_count / window_elapsed if window_elapsed else 0.0
                LOGGER.info(
                    "Processed %d documents in %d batches (%.1f docs/s cumulative, "
                    "%.1f docs/s last window)",
                    document_count,
                    batch_count,
                    cumulative_rate,
                    window_rate,
                )
                last_log_time = now
                last_log_count = processed_count

    elapsed = time.perf_counter() - processing_start
    processed_count = document_count - completed_count
    rate = processed_count / elapsed if elapsed else 0.0
    LOGGER.info(
        "Finished classification: %d documents in %d batches (%.1f docs/s, %.1f s; "
        "model/load setup %.1f s)",
        document_count,
        batch_count,
        rate,
        elapsed,
        processing_start - start_time,
    )


if __name__ == "__main__":
    main()
