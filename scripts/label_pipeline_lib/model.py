from __future__ import annotations

import time
from typing import Any, Sequence, TypeVar

from pydantic import BaseModel, ValidationError  # type: ignore
from vllm import LLM, SamplingParams  # type: ignore
from vllm.sampling_params import StructuredOutputsParams  # type: ignore

try:
    from transformers.tokenization_utils_base import BatchEncoding  # type: ignore
except ImportError:
    BatchEncoding = None  # type: ignore

from label_pipeline_lib.common import LOGGER
from label_pipeline_lib.prompts import (
    build_discovery_messages,
    build_frozen_classification_messages,
    build_reconciliation_messages,
)
from label_pipeline_lib.structured_schemas import (
    StructuredSchemaSpec,
    build_discovery_output_schema,
    build_classification_output_schema,
    build_reconciliation_output_schema,
)
from schemas import (
    FrozenClassificationOutputSchema,
    InputDocument,
    LabelDef,
    LabellingOutputSchema,
    ProposalGroup,
    ReconciliationOutput,
    TaxonomyState,
    FinalReconciliationOutput,
)

M = TypeVar("M", bound=BaseModel)


class ModelClient:
    def __init__(
        self,
        *,
        model_name: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        dtype: str,
        max_model_len: int,
        seed: int,
        thinking_mode: str,
    ) -> None:
        self.model_name = model_name
        self.seed = seed
        self.thinking_mode = thinking_mode

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            max_model_len=max_model_len,
            seed=seed,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.max_model_len = int(self.llm.model_config.max_model_len)

    @property
    def chat_template_kwargs(self) -> dict[str, Any] | None:
        if self.thinking_mode == "disabled":
            return {"enable_thinking": False}
        if self.thinking_mode == "enabled":
            return {"enable_thinking": True}
        if self.thinking_mode == "template-default":
            return None
        raise ValueError(f"Unknown thinking mode: {self.thinking_mode}")

    def _sampling_params(
        self,
        json_schema: dict,
        *,
        max_tokens: int,
    ) -> SamplingParams:
        structured = StructuredOutputsParams(json=json_schema)
        return SamplingParams(
            temperature=0.0,
            seed=self.seed,
            max_tokens=max_tokens,
            structured_outputs=structured,
        )

    def _chat_token_count(self, messages: list[dict[str, str]]) -> int:
        kwargs = self.chat_template_kwargs or {}
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **kwargs,
        )

        # Handle BatchEncoding objects from transformers
        if BatchEncoding is not None and isinstance(tokenized, BatchEncoding):
            if "input_ids" not in tokenized:
                raise TypeError(
                    "Tokenizer chat template returned BatchEncoding without input_ids"
                )
            tokenized = tokenized["input_ids"]
        elif isinstance(tokenized, dict):
            if "input_ids" not in tokenized:
                raise TypeError(
                    "Tokenizer chat template returned a dict without input_ids"
                )
            tokenized = tokenized["input_ids"]

        # Convert tensor-like objects to list (e.g., PyTorch tensors, numpy arrays)
        if hasattr(tokenized, "tolist"):
            tokenized = tokenized.tolist()

        if not isinstance(tokenized, (list, tuple)):
            raise TypeError(
                "Tokenizer apply_chat_template() returned unsupported type "
                f"{type(tokenized).__name__}"
            )
        return len(tokenized)

    def _log_model_throughput(
        self, *, input_tokens: int, generated_tokens: int, elapsed_seconds: float
    ) -> None:
        if elapsed_seconds <= 0:
            LOGGER.warning(
                "Elapsed time is non-positive (%.3f seconds); cannot compute throughput",
                elapsed_seconds,
            )
            return
        total_throughput = (input_tokens + generated_tokens) / elapsed_seconds
        input_throughput = input_tokens / elapsed_seconds
        output_throughput = generated_tokens / elapsed_seconds
        LOGGER.info(
            "Processed %d total tokens in %.3f seconds (%.2f tokens/sec (input: %.2f tokens/sec, output: %.2f tokens/sec))",
            input_tokens + generated_tokens,
            elapsed_seconds,
            total_throughput,
            input_throughput,
            output_throughput,
        )

    def fit_document_messages(
        self,
        *,
        text: str,
        labels: Sequence[LabelDef],
        aspect: str,
        discovery: bool,
        max_document_tokens: int,
        output_max_tokens: int,
    ) -> list[dict[str, str]]:
        doc_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if not isinstance(doc_tokens, (list, tuple)):
            raise TypeError("Tokenizer encode() did not return token IDs")

        upper = min(len(doc_tokens), max_document_tokens)
        prompt_budget = self.max_model_len - output_max_tokens
        if prompt_budget <= 0:
            raise ValueError(
                f"output_max_tokens={output_max_tokens} leaves no prompt budget "
                f"for max_model_len={self.max_model_len}"
            )

        builder = (
            build_discovery_messages
            if discovery
            else build_frozen_classification_messages
        )

        def messages_for(n_tokens: int) -> list[dict[str, str]]:
            if n_tokens >= len(doc_tokens):
                candidate_text = text
            else:
                candidate_text = self.tokenizer.decode(doc_tokens[:n_tokens])
                candidate_text += "\n[TRUNCATED]"
            return builder(candidate_text, labels, aspect)

        full_candidate = messages_for(upper)
        if self._chat_token_count(full_candidate) <= prompt_budget:
            return full_candidate

        empty_candidate = messages_for(0)
        if self._chat_token_count(empty_candidate) > prompt_budget:
            raise ValueError(
                "Prompt containing only instructions and taxonomy exceeds model context "
                f"budget ({prompt_budget} prompt tokens). Reduce taxonomy/prompt size or "
                "increase --max-model-len."
            )

        low, high = 0, upper
        while low < high:
            mid = (low + high + 1) // 2
            if self._chat_token_count(messages_for(mid)) <= prompt_budget:
                low = mid
            else:
                high = mid - 1

        return messages_for(low)

    def _run_structured_chat(
        self,
        conversations: Sequence[list[dict[str, str]]],
        *,
        schema_spec: StructuredSchemaSpec,
        max_tokens: int,
    ) -> list[M]:
        if not conversations:
            return []

        sampling_params = self._sampling_params(
            schema_spec.json_schema, max_tokens=max_tokens
        )
        kwargs: dict[str, Any] = {}
        if self.chat_template_kwargs is not None:
            kwargs["chat_template_kwargs"] = self.chat_template_kwargs

        generate_start_time = time.time()

        # Generate outputs using the LLM's chat method
        outputs = self.llm.chat(
            messages=list(conversations),
            sampling_params=sampling_params,
            use_tqdm=False,
            **kwargs,
        )

        # Log throughput
        generate_end_time = time.time()
        input_tokens = sum(len(output.prompt_token_ids) for output in outputs)
        generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        self._log_model_throughput(
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
            elapsed_seconds=generate_end_time - generate_start_time,
        )
        if len(outputs) != len(conversations):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(conversations)} prompts"
            )

        parsed: list[M] = []
        for index, request_output in enumerate(outputs):
            if len(request_output.outputs) != 1:
                raise RuntimeError(
                    f"Prompt {index}: expected exactly one completion, "
                    f"got {len(request_output.outputs)}"
                )
            completion = request_output.outputs[0]
            if completion.finish_reason != "stop":
                raise RuntimeError(
                    f"Prompt {index}: generation ended with finish_reason="
                    f"{completion.finish_reason!r}; output may be truncated.\n"
                    f"Configured max_tokens={sampling_params.max_tokens}\n"
                    f"Generated tokens: {len(completion.token_ids)}\n"
                    f"Generated text:\n{completion.text!r}"
                )
            generated_text = completion.text
            if not generated_text:
                raise RuntimeError(f"Prompt {index}: model returned empty output")
            try:
                parsed.append(
                    schema_spec.model_type.model_validate_json(generated_text)
                )
            except ValidationError as exc:
                preview = generated_text[:2000]
                raise ValueError(
                    f"Prompt {index}: model output failed {schema_spec.model_type.__name__} "
                    f"validation: {exc}. Output preview: {preview!r}"
                ) from exc
        return parsed

    def discover_batch(
        self,
        docs: Sequence[InputDocument],
        labels: Sequence[LabelDef],
        *,
        aspect: str,
        max_document_tokens: int,
        output_max_tokens: int,
        max_assigned_labels_per_doc: int,
        max_proposals_per_doc: int,
    ) -> list[LabellingOutputSchema]:
        conversations = [
            self.fit_document_messages(
                text=doc.text,
                labels=labels,
                aspect=aspect,
                discovery=True,
                max_document_tokens=max_document_tokens,
                output_max_tokens=output_max_tokens,
            )
            for doc in docs
        ]

        valid_label_ids = [label.id for label in labels]
        discovery_schema_spec = build_discovery_output_schema(
            valid_label_ids=valid_label_ids,
            max_assigned_labels=max_assigned_labels_per_doc,
            max_proposed_labels=max_proposals_per_doc,
        )

        results = self._run_structured_chat(
            conversations,
            schema_spec=discovery_schema_spec,
            max_tokens=output_max_tokens,
        )
        valid_ids = {label.id for label in labels}
        for doc, result in zip(docs, results, strict=True):
            unknown = set(result.assigned_label_ids) - valid_ids
            if unknown:
                raise ValueError(
                    f"Document {doc.doc_id}: model returned unknown label IDs "
                    f"{sorted(unknown)}"
                )
            if len(result.proposed_labels) > max_proposals_per_doc:
                raise ValueError(
                    f"Document {doc.doc_id}: model proposed {len(result.proposed_labels)} "
                    f"labels, exceeding --max-proposals-per-doc={max_proposals_per_doc}"
                )
        return results

    def classify_batch(
        self,
        docs: Sequence[InputDocument],
        labels: Sequence[LabelDef],
        *,
        aspect: str,
        max_document_tokens: int,
        output_max_tokens: int,
        max_assigned_labels_per_doc: int,
    ) -> list[FrozenClassificationOutputSchema]:
        conversations = [
            self.fit_document_messages(
                text=doc.text,
                labels=labels,
                aspect=aspect,
                discovery=False,
                max_document_tokens=max_document_tokens,
                output_max_tokens=output_max_tokens,
            )
            for doc in docs
        ]

        classification_schema_spec = self.build_classification_output_schema(
            valid_label_ids=[label.id for label in labels],
            max_assigned_labels=max_assigned_labels_per_doc,
        )

        results = self._run_structured_chat(
            conversations,
            schema_spec=classification_schema_spec,
            max_tokens=output_max_tokens,
        )
        valid_ids = {label.id for label in labels}
        for doc, result in zip(docs, results, strict=True):
            unknown = set(result.assigned_label_ids) - valid_ids
            if unknown:
                raise ValueError(
                    f"Document {doc.doc_id}: model returned unknown label IDs "
                    f"{sorted(unknown)}"
                )
        return results

    def reconcile(
        self,
        state: TaxonomyState,
        proposal_groups: Sequence[ProposalGroup],
        *,
        max_total_labels: int,
        min_create_support: int,
        max_tokens: int,
        final_maintenance: bool,
    ) -> ReconciliationOutput | FinalReconciliationOutput:

        # Build short model-facing ID aliases for proposal groups to reduce token usage in the prompt
        global_to_local = {
            group.group_id: f"P{i:03d}"
            for i, group in enumerate(proposal_groups, start=1)
        }

        local_to_global = {
            local_id: global_id for global_id, local_id in global_to_local.items()
        }

        model_proposal_groups = [
            group.model_copy(update={"group_id": global_to_local[group.group_id]})
            for group in proposal_groups
        ]

        messages = build_reconciliation_messages(
            state,
            model_proposal_groups,
            max_total_labels=max_total_labels,
            min_create_support=min_create_support,
            final_maintenance=final_maintenance,
        )

        prompt_tokens = self._chat_token_count(messages)
        prompt_budget = self.max_model_len - max_tokens
        if prompt_tokens > prompt_budget:
            raise ValueError(
                f"Reconciliation prompt is {prompt_tokens} tokens but only "
                f"{prompt_budget} prompt tokens are available. Lower "
                "--reconcile-every/--max-reconcile-groups or increase "
                "--max-model-len."
            )

        schema_spec = build_reconciliation_output_schema(
            final_maintenance=final_maintenance,
            valid_proposal_group_ids=list(local_to_global.keys()),
        )

        result = self._run_structured_chat(
            [messages],
            schema=schema_spec,
            max_tokens=max_tokens,
        )[0]

        # Translate local IDs back to persistent global IDs.
        translated_resolutions = []

        for resolution in result.proposal_resolutions:
            translated_ids = []

            for local_id in resolution.proposal_group_ids:
                if local_id not in local_to_global:
                    raise ValueError(
                        f"Reconciler returned unknown local proposal ID "
                        f"{local_id!r}"
                    )

                translated_ids.append(local_to_global[local_id])

            translated_resolutions.append(
                resolution.model_copy(update={"proposal_group_ids": translated_ids})
            )

        result = result.model_copy(
            update={"proposal_resolutions": translated_resolutions}
        )

        return result
