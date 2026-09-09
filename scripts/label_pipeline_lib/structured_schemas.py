from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel  # type: ignore

from label_pipeline_lib.schemas import (
    CandidatePromotionOutput,
    FinalMergeOutput,
    FinalRevisionOutput,
    FrozenClassificationOutputSchema,
    LabellingOutputSchema,
    ProposalScreeningOutput,
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass(frozen=True)
class PromptItem:
    debug_id: str
    messages: list[dict[str, str]]


@dataclass(frozen=True)
class StructuredSchemaSpec(Generic[SchemaT]):
    model_type: type[SchemaT]
    json_schema: dict


def _require_unique_nonempty(values: list[str], *, field_name: str) -> None:
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} contains duplicates")


def build_discovery_output_schema(
    valid_label_ids: list[str],
    max_assigned_labels: int,
    max_proposed_labels: int,
) -> StructuredSchemaSpec[LabellingOutputSchema]:
    _require_unique_nonempty(valid_label_ids, field_name="valid_label_ids")
    if max_assigned_labels < 1:
        raise ValueError("max_assigned_labels must be >= 1")
    if max_proposed_labels < 1:
        raise ValueError("max_proposed_labels must be >= 1")

    schema = LabellingOutputSchema.model_json_schema()
    assigned = schema["properties"]["assigned_label_ids"]
    assigned["items"] = {"type": "string", "enum": valid_label_ids}
    assigned["maxItems"] = max_assigned_labels

    schema["properties"]["proposed_labels"]["maxItems"] = max_proposed_labels

    return StructuredSchemaSpec(
        model_type=LabellingOutputSchema,
        json_schema=schema,
    )


def build_classification_output_schema(
    valid_label_ids: list[str],
    max_assigned_labels: int,
) -> StructuredSchemaSpec[FrozenClassificationOutputSchema]:
    _require_unique_nonempty(valid_label_ids, field_name="valid_label_ids")
    if max_assigned_labels < 1:
        raise ValueError("max_assigned_labels must be >= 1")

    schema = FrozenClassificationOutputSchema.model_json_schema()
    assigned = schema["properties"]["assigned_label_ids"]
    assigned["items"] = {"type": "string", "enum": valid_label_ids}
    assigned["maxItems"] = max_assigned_labels

    return StructuredSchemaSpec(
        model_type=FrozenClassificationOutputSchema,
        json_schema=schema,
    )


def build_proposal_screening_output_schema(
    valid_proposal_group_ids: list[str],
    valid_target_label_ids: list[str],
) -> StructuredSchemaSpec[ProposalScreeningOutput]:
    _require_unique_nonempty(
        valid_proposal_group_ids, field_name="valid_proposal_group_ids"
    )
    _require_unique_nonempty(
        valid_target_label_ids, field_name="valid_target_label_ids"
    )

    schema = ProposalScreeningOutput.model_json_schema()
    n_groups = len(valid_proposal_group_ids)

    decisions = schema["properties"]["decisions"]
    # Exactly one decision per supplied proposal group. Cross-item uniqueness is
    # still validated by Pydantic/Python after generation.
    decisions["minItems"] = n_groups
    decisions["maxItems"] = n_groups

    for def_name in (
        "KeepCandidateDecision",
        "DiscardCandidateDecision",
        "MapExistingDecision",
    ):
        schema["$defs"][def_name]["properties"]["proposal_group_id"] = {
            "type": "string",
            "enum": valid_proposal_group_ids,
        }

    schema["$defs"]["MapExistingDecision"]["properties"]["target_label_id"] = {
        "type": "string",
        "enum": valid_target_label_ids,
    }

    return StructuredSchemaSpec(
        model_type=ProposalScreeningOutput,
        json_schema=schema,
    )


def build_candidate_promotion_output_schema(
    valid_candidate_group_ids: list[str],
    *,
    max_new_labels: int,
) -> StructuredSchemaSpec[CandidatePromotionOutput]:
    _require_unique_nonempty(
        valid_candidate_group_ids, field_name="valid_candidate_group_ids"
    )
    if max_new_labels < 1:
        raise ValueError("max_new_labels must be >= 1")

    schema = CandidatePromotionOutput.model_json_schema()
    n_groups = len(valid_candidate_group_ids)

    new_labels = schema["properties"]["new_labels"]
    new_labels["minItems"] = 1
    new_labels["maxItems"] = min(max_new_labels, n_groups)

    group_ids = schema["$defs"]["PromotedCandidateLabel"]["properties"][
        "candidate_group_ids"
    ]
    group_ids["items"] = {
        "type": "string",
        "enum": valid_candidate_group_ids,
    }
    # Prevent an unbounded repetition loop even though uniqueItems is not
    # available in the vLLM 0.22.1 XGrammar path.
    group_ids["maxItems"] = n_groups

    return StructuredSchemaSpec(
        model_type=CandidatePromotionOutput,
        json_schema=schema,
    )


def build_final_merge_output_schema(
    valid_dynamic_label_ids: list[str],
    valid_target_label_ids: list[str],
) -> StructuredSchemaSpec[FinalMergeOutput]:
    _require_unique_nonempty(
        valid_dynamic_label_ids, field_name="valid_dynamic_label_ids"
    )
    _require_unique_nonempty(
        valid_target_label_ids, field_name="valid_target_label_ids"
    )

    schema = FinalMergeOutput.model_json_schema()
    schema["properties"]["merges"]["maxItems"] = len(valid_dynamic_label_ids)

    merge_def = schema["$defs"]["FinalMergeDecision"]["properties"]
    merge_def["source_label_id"] = {
        "type": "string",
        "enum": valid_dynamic_label_ids,
    }
    merge_def["target_label_id"] = {
        "type": "string",
        "enum": valid_target_label_ids,
    }

    return StructuredSchemaSpec(
        model_type=FinalMergeOutput,
        json_schema=schema,
    )


def build_final_revision_output_schema(
    valid_dynamic_label_ids: list[str],
) -> StructuredSchemaSpec[FinalRevisionOutput]:
    _require_unique_nonempty(
        valid_dynamic_label_ids, field_name="valid_dynamic_label_ids"
    )

    schema = FinalRevisionOutput.model_json_schema()
    schema["properties"]["revisions"]["maxItems"] = len(valid_dynamic_label_ids)
    schema["$defs"]["FinalRevisionDecision"]["properties"]["label_id"] = {
        "type": "string",
        "enum": valid_dynamic_label_ids,
    }

    return StructuredSchemaSpec(
        model_type=FinalRevisionOutput,
        json_schema=schema,
    )
