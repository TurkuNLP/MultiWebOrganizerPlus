from __future__ import annotations

import json
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
    build_candidate_promotion_messages,
    build_discovery_messages,
    build_final_merge_messages,
    build_final_revision_messages,
    build_frozen_classification_messages,
    build_proposal_screening_messages,
)
from label_pipeline_lib.structured_schemas import (
    PromptItem,
    StructuredSchemaSpec,
    build_candidate_promotion_output_schema,
    build_classification_output_schema,
    build_discovery_output_schema,
    build_final_merge_output_schema,
    build_final_revision_output_schema,
    build_proposal_screening_output_schema,
)
from label_pipeline_lib.schemas import (
    CandidatePromotionOutput,
    FinalMergeOutput,
    FinalRevisionOutput,
    FrozenClassificationOutputSchema,
    InputDocument,
    LabelDef,
    LabellingOutputSchema,
    ProposalGroup,
    ProposalScreeningOutput,
    TaxonomyState,
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
            performance_mode="throughput",
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
        creativity: str | None,
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
            return builder(candidate_text, labels, aspect, creativity)

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
        prompt_items: Sequence[PromptItem],
        *,
        schema_spec: StructuredSchemaSpec[M],
        max_tokens: int,
    ) -> list[M]:
        if not prompt_items:
            return []

        sampling_params = self._sampling_params(
            schema_spec.json_schema, max_tokens=max_tokens
        )
        kwargs: dict[str, Any] = {}
        if self.chat_template_kwargs is not None:
            kwargs["chat_template_kwargs"] = self.chat_template_kwargs

        generate_start_time = time.time()

        messages = [prompt_item.messages for prompt_item in prompt_items]
        # Generate outputs using the LLM's chat method
        outputs = self.llm.chat(
            messages=messages,
            sampling_params=sampling_params,
            use_tqdm=False,
            **kwargs,
        )

        generate_end_time = time.time()
        if len(outputs) != len(messages):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(messages)} prompts"
            )

        # Validate output cardinality before indexing completions for throughput
        # accounting, so malformed vLLM responses fail with the prompt debug ID
        # instead of an unrelated IndexError.
        for index, request_output in enumerate(outputs):
            if len(request_output.outputs) != 1:
                raise RuntimeError(
                    f"{prompt_items[index].debug_id}: expected exactly one completion, "
                    f"got {len(request_output.outputs)}"
                )

        input_tokens = sum(len(output.prompt_token_ids) for output in outputs)
        generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        self._log_model_throughput(
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
            elapsed_seconds=generate_end_time - generate_start_time,
        )

        parsed: list[M] = []
        for index, request_output in enumerate(outputs):
            completion = request_output.outputs[0]
            if completion.finish_reason != "stop":
                raise RuntimeError(
                    f"{prompt_items[index].debug_id}: generation ended with finish_reason="
                    f"{completion.finish_reason!r}; output may be truncated.\n"
                    f"Configured max_tokens={sampling_params.max_tokens}\n"
                    f"Generated tokens: {len(completion.token_ids)}\n"
                    f"Generated text:\n{completion.text!r}"
                )
            generated_text = completion.text
            if not generated_text:
                raise RuntimeError(
                    f"{prompt_items[index].debug_id}: model returned empty output"
                )
            try:
                payload = json.loads(generated_text)

                # Forcefully deduplicate assigned_label_ids to avoid validation errors due to duplicates
                # This is a workaround for cases where the model may return duplicate label IDs, which would violate the schema's constraints.
                # Would be better to enforce uniqueness in the output itself, but that is not easily doable.
                if "assigned_label_ids" in payload and isinstance(
                    payload["assigned_label_ids"], list
                ):
                    num_assigned_label_ids = len(payload["assigned_label_ids"])
                    payload["assigned_label_ids"] = list(
                        dict.fromkeys(payload["assigned_label_ids"])
                    )
                    num_deduplicated = len(payload["assigned_label_ids"])
                    if num_deduplicated < num_assigned_label_ids:
                        LOGGER.warning(
                            f"{prompt_items[index].debug_id}: "
                            f"model returned {num_assigned_label_ids - num_deduplicated} duplicate assigned_label_ids; "
                            f"deduplicated to {num_deduplicated}."
                        )
                parsed.append(schema_spec.model_type.model_validate(payload))
            except ValidationError as exc:
                preview = generated_text[:2000]
                raise ValueError(
                    f"{prompt_items[index].debug_id}: model output failed {schema_spec.model_type.__name__} "
                    f"validation: {exc}. Output preview: {preview!r}"
                ) from exc
        return parsed

    def discover_batch(
        self,
        docs: Sequence[InputDocument],
        labels: Sequence[LabelDef],
        *,
        aspect: str,
        creativity: str | None,
        max_document_tokens: int,
        output_max_tokens: int,
        max_assigned_labels_per_doc: int,
        max_proposals_per_doc: int,
    ) -> list[LabellingOutputSchema]:

        prompt_items: list[PromptItem] = []

        messages = [
            self.fit_document_messages(
                text=doc.text,
                labels=labels,
                aspect=aspect,
                creativity=creativity,
                discovery=True,
                max_document_tokens=max_document_tokens,
                output_max_tokens=output_max_tokens,
            )
            for doc in docs
        ]

        prompt_items.extend(
            PromptItem(
                debug_id=f"doc:{doc.doc_id}",
                messages=messages,
            )
            for doc, messages in zip(docs, messages, strict=True)
        )

        valid_label_ids = [label.id for label in labels]

        discovery_schema_spec = build_discovery_output_schema(
            valid_label_ids=valid_label_ids,
            max_assigned_labels=max_assigned_labels_per_doc,
            max_proposed_labels=max_proposals_per_doc,
        )

        results = self._run_structured_chat(
            prompt_items,
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
            if len(result.assigned_label_ids) > max_assigned_labels_per_doc:
                raise ValueError(
                    f"Document {doc.doc_id}: model assigned {len(result.assigned_label_ids)} "
                    f"labels, exceeding --max-assigned-labels-per-doc="
                    f"{max_assigned_labels_per_doc}"
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

        prompt_items: list[PromptItem] = []

        messages = [
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

        prompt_items.extend(
            PromptItem(
                debug_id=f"doc:{doc.doc_id}",
                messages=messages,
            )
            for doc, messages in zip(docs, messages, strict=True)
        )

        classification_schema_spec = build_classification_output_schema(
            valid_label_ids=[label.id for label in labels],
            max_assigned_labels=max_assigned_labels_per_doc,
        )

        results = self._run_structured_chat(
            prompt_items,
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
            if len(result.assigned_label_ids) > max_assigned_labels_per_doc:
                raise ValueError(
                    f"Document {doc.doc_id}: model assigned {len(result.assigned_label_ids)} "
                    f"labels, exceeding --max-assigned-labels-per-doc="
                    f"{max_assigned_labels_per_doc}"
                )
        return results

    def _run_single_structured_task(
        self,
        *,
        debug_id: str,
        messages: list[dict[str, str]],
        schema_spec: StructuredSchemaSpec[M],
        max_tokens: int,
    ) -> M:
        prompt_tokens = self._chat_token_count(messages)
        prompt_budget = self.max_model_len - max_tokens
        if prompt_tokens > prompt_budget:
            raise ValueError(
                f"{debug_id}: prompt is {prompt_tokens} tokens but only "
                f"{prompt_budget} prompt tokens are available. Reduce the task "
                "size or increase --max-model-len."
            )

        results = self._run_structured_chat(
            [PromptItem(debug_id=debug_id, messages=messages)],
            schema_spec=schema_spec,
            max_tokens=max_tokens,
        )
        if len(results) != 1:
            raise RuntimeError(
                f"{debug_id}: expected exactly one structured result, got {len(results)}"
            )
        return results[0]

    @staticmethod
    def _proposal_aliases(
        proposal_groups: Sequence[ProposalGroup],
    ) -> tuple[dict[str, str], dict[str, str], list[ProposalGroup]]:
        global_to_local = {
            group.group_id: f"P{i:03d}"
            for i, group in enumerate(proposal_groups, start=1)
        }
        local_to_global = {
            local_id: global_id for global_id, local_id in global_to_local.items()
        }
        model_groups = [
            group.model_copy(update={"group_id": global_to_local[group.group_id]})
            for group in proposal_groups
        ]
        return global_to_local, local_to_global, model_groups

    @staticmethod
    def _translate_screening_ids(
        result: ProposalScreeningOutput,
        local_to_global: dict[str, str],
    ) -> ProposalScreeningOutput:
        translated = []
        for decision in result.decisions:
            local_id = decision.proposal_group_id
            if local_id not in local_to_global:
                raise ValueError(
                    f"Proposal screener returned unknown local proposal ID {local_id!r}"
                )
            translated.append(
                decision.model_copy(
                    update={"proposal_group_id": local_to_global[local_id]}
                )
            )
        return result.model_copy(update={"decisions": translated})

    @staticmethod
    def _translate_promotion_ids(
        result: CandidatePromotionOutput,
        local_to_global: dict[str, str],
    ) -> CandidatePromotionOutput:
        translated_labels = []
        for new_label in result.new_labels:
            translated_ids = []
            for local_id in new_label.candidate_group_ids:
                if local_id not in local_to_global:
                    raise ValueError(
                        f"Candidate promoter returned unknown local candidate ID {local_id!r}"
                    )
                translated_ids.append(local_to_global[local_id])
            translated_labels.append(
                new_label.model_copy(update={"candidate_group_ids": translated_ids})
            )
        return result.model_copy(update={"new_labels": translated_labels})

    def screen_proposals(
        self,
        state: TaxonomyState,
        proposal_groups: Sequence[ProposalGroup],
        aspect: str,
        *,
        max_tokens: int,
    ) -> ProposalScreeningOutput:
        if not proposal_groups:
            raise ValueError("screen_proposals requires at least one proposal group")

        if not aspect:
            raise ValueError("screen_proposals requires a non-empty aspect")

        _, local_to_global, model_groups = self._proposal_aliases(proposal_groups)
        messages = build_proposal_screening_messages(state, model_groups, aspect=aspect)
        schema_spec = build_proposal_screening_output_schema(
            valid_proposal_group_ids=list(local_to_global),
            valid_target_label_ids=[label.id for label in state.seed_labels]
            + [label.id for label in state.dynamic_labels],
        )

        result = self._run_single_structured_task(
            debug_id=f"proposal_screening:taxonomy_v{state.schema_version}",
            messages=messages,
            schema_spec=schema_spec,
            max_tokens=max_tokens,
        )
        return self._translate_screening_ids(result, local_to_global)

    def promote_candidates(
        self,
        state: TaxonomyState,
        candidate_groups: Sequence[ProposalGroup],
        aspect: str,
        *,
        max_new_labels: int,
        max_tokens: int,
    ) -> CandidatePromotionOutput:
        if not candidate_groups:
            raise ValueError("promote_candidates requires at least one candidate group")
        if max_new_labels < 1:
            raise ValueError("max_new_labels must be >= 1")
        if not aspect:
            raise ValueError("promote_candidates requires a non-empty aspect")

        _, local_to_global, model_groups = self._proposal_aliases(candidate_groups)
        messages = build_candidate_promotion_messages(
            state, model_groups, max_new_labels=max_new_labels, aspect=aspect
        )
        schema_spec = build_candidate_promotion_output_schema(
            valid_candidate_group_ids=list(local_to_global),
            max_new_labels=max_new_labels,
        )

        result = self._run_single_structured_task(
            debug_id=f"candidate_promotion:taxonomy_v{state.schema_version}",
            messages=messages,
            schema_spec=schema_spec,
            max_tokens=max_tokens,
        )
        return self._translate_promotion_ids(result, local_to_global)

    def final_merge_pass(
        self,
        state: TaxonomyState,
        aspect: str,
        *,
        max_tokens: int,
    ) -> FinalMergeOutput:
        if not state.dynamic_labels:
            return FinalMergeOutput(merges=[])

        if not aspect:
            raise ValueError("final_merge_pass requires a non-empty aspect")

        messages = build_final_merge_messages(state, aspect=aspect)
        schema_spec = build_final_merge_output_schema(
            valid_dynamic_label_ids=[label.id for label in state.dynamic_labels],
            valid_target_label_ids=[label.id for label in state.seed_labels]
            + [label.id for label in state.dynamic_labels],
        )
        return self._run_single_structured_task(
            debug_id=f"final_merge:taxonomy_v{state.schema_version}",
            messages=messages,
            schema_spec=schema_spec,
            max_tokens=max_tokens,
        )

    def final_revision_pass(
        self,
        state: TaxonomyState,
        aspect: str,
        review_label_ids: Sequence[str],
        *,
        max_tokens: int,
    ) -> FinalRevisionOutput:
        review_ids = list(review_label_ids)
        if not review_ids:
            return FinalRevisionOutput(revisions=[])
        if len(review_ids) != len(set(review_ids)):
            raise ValueError("review_label_ids contains duplicates")
        if not aspect:
            raise ValueError("final_revision_pass requires a non-empty aspect")

        dynamic_ids = {label.id for label in state.dynamic_labels}
        unknown = set(review_ids) - dynamic_ids
        if unknown:
            raise ValueError(
                f"Final revision requested non-dynamic labels {sorted(unknown)}"
            )

        messages = build_final_revision_messages(state, review_ids, aspect=aspect)
        schema_spec = build_final_revision_output_schema(
            valid_dynamic_label_ids=review_ids,
        )
        return self._run_single_structured_task(
            debug_id=(
                f"final_revision:taxonomy_v{state.schema_version}:"
                f"{review_ids[0]}-{review_ids[-1]}"
            ),
            messages=messages,
            schema_spec=schema_spec,
            max_tokens=max_tokens,
        )
