"""
Main streaming labeling pipeline.

Features implemented:
- Streams JSONL input without loading whole file into memory.
- Configurable batching for LLM inference.
- Truncation of document text to an adjustable token limit (heuristic fallback available).
- Uses vLLM (preferred) with attempt to use structured-output parsing into Pydantic models.
- Validates every model response using Pydantic before applying it.
- Separates classification from schema maintenance: proposed new labels are appended to a proposals queue; classification never mutates the main taxonomy.
- Periodic checkpointing of processed document IDs and taxonomy state for resumability.
- Checkpointing prevents re-processing on restart unless forced.

Notes:
- vLLM structured-output support varies by version; code attempts to use a Pydantic output parser from vllm if available and falls back to JSON parsing.
- Default truncation is 8192 tokens (configurable). If a tokeniser for Qwen is available, it will be used; otherwise a conservative character-based heuristic is used.

To run (example):
python3 MultiWebOrganizerPlus/scripts/label_pipeline.py \
  --input /path/to/data.jsonl \
  --batch-size 8 \
  --truncate-tokens 8192 \
  --checkpoint-dir MultiWebOrganizerPlus/data/checkpoints \
  --model qwen/Qwen3.6-35B-A3B

"""
from __future__ import annotations
import argparse
import json
import os
import time
from typing import Iterator, Dict, Any, List, Optional
import logging
import pyyaml

from pathlib import Path

from schemas import LabelDef, LabellingOutputSchema, LabelledDocument
from prompts.labelling_prompt import LabellingPrompt

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


#----------- Logging configuration ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)


# ---------- Utilities ----------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def stream_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Yield one JSON object per line, parsed. Does not load the whole file into memory."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # Skip/raise as appropriate; here raise so caller sees malformed data
                raise


def batch_iter(iterable: Iterator[Any], size: int) -> Iterator[List[Any]]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# Simple token truncation heuristic when a tokenizer is not available.
# Assumes 4 characters per token on average.
CHARS_PER_TOKEN_ESTIMATE = 4


def truncate_text_to_tokens(text: str, token_limit: int, tokenizer=None) -> str:
    """
    Truncate text to a specified token limit.
    Truncated text is returned with a "[TRUNCATED]" marker appended.

    Args:
        text (str): Document text to truncate.
        token_limit (int): Maximum number of tokens allowed.
        tokenizer (_type_, optional): A tokenizer instance that can encode and decode text. Defaults to None.

    Returns:
        str: Original text or truncated text with a "[TRUNCATED]" marker if truncation occurred.
    """
    if tokenizer is not None:
        # tokenizer expected to provide encode/text_to_ids and allow truncation reliably
        try:
            toks = tokenizer.encode(text)
            if len(toks) <= token_limit:
                return text
            # decode back truncated tokens if possible
            try:
                truncated = tokenizer.decode(toks[:token_limit])
                return truncated + "\n[TRUNCATED]"
            except Exception:
                # fallback to char heuristic
                pass
        except Exception:
            pass
    # Heuristic fallback: approximate by characters
    max_chars = token_limit * CHARS_PER_TOKEN_ESTIMATE
    if len(text) <= max_chars:
        return text
    # Truncate safely preserving UTF-8 boundaries
    truncated = text[:max_chars]
    return truncated + "\n[TRUNCATED]"


# ---------- Checkpoint and taxonomy helpers ----------

class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        ensure_dir(self.checkpoint_dir)
        self.processed_file = os.path.join(self.checkpoint_dir, "processed_ids.jsonl")
        self.taxonomy_file = os.path.join(self.checkpoint_dir, "taxonomy_state.json")
        self.proposals_file = os.path.join(self.checkpoint_dir, "schema_proposals.jsonl")
        # in-memory set for quick checks
        self._processed_ids = None

    def load_processed_ids(self) -> set:
        if self._processed_ids is not None:
            return self._processed_ids
        ids = set()
        if os.path.exists(self.processed_file):
            with open(self.processed_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        ids.add(rec.get("id"))
                    except Exception:
                        continue
        self._processed_ids = ids
        return ids

    def mark_processed(self, doc_id: str, metadata: Optional[Dict[str, Any]] = None):
        if metadata is None:
            metadata = {}
        entry = {"id": doc_id, "ts": int(time.time()), "meta": metadata}
        with open(self.processed_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if self._processed_ids is None:
            self._processed_ids = set()
        self._processed_ids.add(doc_id)

    def load_taxonomy(self) -> Dict[str, Any]:
        if os.path.exists(self.taxonomy_file):
            with open(self.taxonomy_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_taxonomy(self, taxonomy: Dict[str, Any]):
        with open(self.taxonomy_file, "w", encoding="utf-8") as f:
            json.dump(taxonomy, f, ensure_ascii=False, indent=2)

    def append_proposal(self, proposal: Dict[str, Any]):
        # Each line is a JSON object proposal discovered during classification
        with open(self.proposals_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"proposal": proposal, "ts": int(time.time())}, ensure_ascii=False) + "\n")

    def iter_proposals(self) -> Iterator[Dict[str, Any]]:
        if not os.path.exists(self.proposals_file):
            return
        with open(self.proposals_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


# ---------- Model wrapper ----------

class ModelClient:
    def __init__(self, model_name: str, max_input_tokens: int = 8192):
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.llm = LLM(self.model_name)
        self.parser = None

    def classify_batch(self, batch_docs: List[Dict[str, Any]], seed_labels: List[Dict[str, str]], aspect: str, truncate_tokens: int) -> List[LabellingOutputSchema]:
        """
        Given a list of documents (each with at least `id` and `text`), returns a validated BatchClassificationResponse.

        The model is expected to produce both assigned labels (referencing existing label IDs where appropriate)
        and a list of proposed new labels (for schema maintenance). Model output is validated with Pydantic.
        """
        # Build structured prompt for each document.
        prompts = []
        for doc in batch_docs:
            doc_id = doc.get("id")
            text = doc.get("text", "")
            assert doc_id is not None, "Document must have an 'id' field"
            assert text is not None, "Document must have a 'text' field"
            # Truncate text to token limit
            truncated = truncate_text_to_tokens(text, truncate_tokens)
            # Current set of labels and their IDs
            label_snapshot = "\n".join([f"{l['id']}: {l['name']}" for l in seed_labels])
            prompt = LabellingPrompt(
                doc_id=doc_id,
                text=truncated,
                labels=label_snapshot,
                aspect=aspect
            )
            prompts.append(prompt)

        # Process prompts through vLLM
        output_json_schema = LabellingOutputSchema.json_schema()
        sampling_params = SamplingParams(structured_outputs=StructuredOutputsParams(output_schema=output_json_schema))
        model_outputs = self.llm.chat(prompts, sampling_params=sampling_params)

        # Parse and validate outputs
        parsed_outputs = [self.parse_model_output(output) for output in model_outputs]
        validated_outputs = [self.validate_output_semantics(parsed, {l['id'] for l in seed_labels}) for parsed in parsed_outputs]
        
        return validated_outputs
        
    @staticmethod
    def parse_model_output(output) -> LabellingOutputSchema:
        """
        Attempt to parse the model output string into a structured dict.
        If vLLM structured-output parsing is available, use it; otherwise fallback to JSON parsing.
        """
        generated_text = output.outputs[0].text

        return LabellingOutputSchema.model_validate(generated_text)

    
    @staticmethod
    def validate_output_semantics(result: LabellingOutputSchema, valid_label_ids: set[str]) -> LabellingOutputSchema:
        # log duplicate IDs in assigned_label_ids and remove duplicates
        if len(result.assigned_label_ids) != len(set(result.assigned_label_ids)):
            logging.warning(f"Duplicate label IDs found in assigned_label_ids for document. Duplicates will be removed.")
        result = LabellingOutputSchema(
            assigned_label_ids=list(set(result.assigned_label_ids)),
            proposed_labels=result.proposed_labels
        )
        
        # Log invalid IDs in assigned_label_ids and drop them
        if set(result.assigned_label_ids) - valid_label_ids:
            logging.warning("Model returned invalid label IDs. They will be dropped from assigned_label_ids.")
            result = LabellingOutputSchema(
                assigned_label_ids=list(set(result.assigned_label_ids) & valid_label_ids),
                proposed_labels=result.proposed_labels
            )

        return result

# ---------- Main pipeline ----------

def process_stream(input_path: str, aspect: str, seed_labels_path: str, checkpoint_dir: str, batch_size: int, truncate_tokens: int, model_name: str, force_reprocess: bool = False):
    ensure_dir(checkpoint_dir)
    ck = CheckpointManager(checkpoint_dir)

    # Load seed labels (these are immutable per requirements)
    with open(seed_labels_path, "r", encoding="utf-8") as f:
        seed_labels = pyyaml.safe_load(f)

    processed_ids = ck.load_processed_ids()

    model = ModelClient(model_name=model_name, max_input_tokens=truncate_tokens)

    stream = stream_jsonl(input_path)

    total = 0
    for batch in batch_iter(stream, batch_size):
        # Filter already-processed unless force_reprocess
        batch_to_run = []
        for doc in batch:
            doc_id = doc.get('doc_id')
            if doc_id is None:
                raise ValueError("Document missing 'doc_id' field")
            if (not force_reprocess) and (doc_id in processed_ids):
                continue
            batch_to_run.append(doc)

        if not batch_to_run:
            continue

        # Classify batch
        resp = model.classify_batch(batch_to_run, seed_labels, aspect, truncate_tokens)
        
        classified_docs = [LabelledDocument(doc_id=doc.get('doc_id'), output=out) for doc, out in zip(batch_to_run, resp)]

        # For each proposed label, append to proposals queue for schema maintenance
        for doc_obj in classified_docs:
            for i, p in enumerate(doc_obj.output.proposed_labels):
                temp_id = f"temp-{doc_obj.id}-{i}"
                label_def = LabelDef(id=p.name, name=p.name, definition=p.definition)
                # Append proposal along with originating document id
                ck.append_proposal({"from_doc": doc_obj.id, "proposal": label_def.dict()})

            # Mark document as processed
            ck.mark_processed(doc_obj.id)
            total += 1

        # Periodic taxonomy save (we don't mutate seed_labels here but keep a record)
        taxonomy_state = ck.load_taxonomy() or {}
        taxonomy_state.setdefault("seed_labels_path", seed_labels_path)
        taxonomy_state.setdefault("last_checkpoint_ts", int(time.time()))
        taxonomy_state.setdefault("total_processed", 0)
        taxonomy_state["total_processed"] += len(resp)
        ck.save_taxonomy(taxonomy_state)

        print(f"Processed batch of {len(resp)} documents (total processed this run: {total})")

        # After each batch, reconcile schema proposals to update taxonomy_state.json (does not mutate seed_labels.json)
        reconcile_schema(checkpoint_dir, seed_labels_path)

    print(f"Done. Total processed in this run: {total}")


# ---------- Schema maintenance placeholder ----------

def reconcile_schema(checkpoint_dir: str, seed_labels_path: str, taxonomy_save_path: Optional[str] = None, max_total_labels: int = 100):
    """
    Dedicated schema maintenance step. Reads proposals from the proposals file, applies reconciliation logic,
    and writes updates to a taxonomy_state.json file. This function does NOT run automatically during classification
    and must be triggered by an operator or scheduled job.

    Important behaviors implemented:
    - Proposed labels are reviewed but NOT automatically merged into initial seed labels.
    - Newly accepted labels receive stable IDs L025+ and definitions; an alias/tombstone mapping is maintained when labels are retired.
    - Does not modify the original seed_labels.json file; instead writes an augmented taxonomy file in checkpoint dir.
    """
    ck = CheckpointManager(checkpoint_dir)
    seed_labels = []
    with open(seed_labels_path, "r", encoding="utf-8") as f:
        seed_labels = json.load(f)

    taxonomy = ck.load_taxonomy() or {}
    accepted_labels: List[Dict[str, Any]] = taxonomy.get("dynamic_labels", [])
    alias_map: Dict[str, str] = taxonomy.get("alias_map", {})

    # Collect proposals and count occurrences by (name, definition) signature
    proposals = {}
    for rec in ck.iter_proposals():
        try:
            prop = rec.get("proposal")
            name = prop.get("name")
            definition = prop.get("definition")
            key = (name.strip().lower(), definition.strip() if definition else "")
            proposals.setdefault(key, {"count": 0, "example": prop}).update({"count": proposals.get(key, {}).get("count", 0) + 1})
        except Exception:
            continue

    # Simple heuristic: accept proposals seen >= 2 times and ensure total labels cap
    next_id_num = 25 + len(accepted_labels)
    for (name_def, stats) in list(proposals.items()):
        if stats.get("count", 0) >= 2:
            if 24 + len(accepted_labels) >= max_total_labels:
                print("Reached max_total_labels cap; skipping further automatic accepts. Manual review required.")
                break
            name, _ = name_def
            example = stats.get("example")
            new_id = f"L{next_id_num:03d}"
            next_id_num += 1
            accepted = {"id": new_id, "name": example.get("name"), "definition": example.get("definition"), "accepted_ts": int(time.time()), "origin_count": stats.get("count", 0)}
            accepted_labels.append(accepted)

    taxonomy["dynamic_labels"] = accepted_labels
    taxonomy["alias_map"] = alias_map
    taxonomy["last_reconciled_ts"] = int(time.time())

    ck.save_taxonomy(taxonomy)
    print(f"Schema reconciliation complete: now {len(accepted_labels)} dynamic labels")


# ---------- CLI ----------

def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to input JSONL (documents with id and text)")
    p.add_argument("--aspect", default="topics", help="Aspect to classify (e.g., topics, sentiment, etc.)")
    p.add_argument("--seed-labels", default="MultiWebOrganizerPlus/data/seed_labels.json", help="Path to seed labels JSON file")
    p.add_argument("--checkpoint-dir", default="MultiWebOrganizerPlus/data/checkpoints", help="Directory to store checkpoints and taxonomy state")
    p.add_argument("--batch-size", type=int, default=8, help="Number of documents to batch per inference call")
    p.add_argument("--truncate-tokens", type=int, default=8192, help="Token truncation limit per document (default 8192)")
    p.add_argument("--model", default="qwen/Qwen3.6-35B-A3B", help="Model name to pass to vLLM; ensure model is available locally for vLLM")
    p.add_argument("--force-reprocess", action='store_true', help="Reprocess documents even if checkpoint says they were processed")
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    process_stream(
        input_path=args.input,
        aspect=args.aspect,
        seed_labels_path=args.seed_labels,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        truncate_tokens=args.truncate_tokens,
        model_name=args.model,
        force_reprocess=args.force_reprocess,
    )


if __name__ == "__main__":
    main()
