from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, model_validator  # type: ignore

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


class _KeyedProposalScreeningOutput(ProposalScreeningOutput):
    """Model-boundary representation for proposal screening.

    The LLM is constrained to emit::

        {
          "decisions": {
            "P001": {"action": "keep_candidate"},
            "P002": {
              "action": "map_existing",
              "target_label_id": "L003"
            }
          }
        }

    The canonical application model remains ProposalScreeningOutput, whose
    ``decisions`` field is a list and whose decisions contain
    ``proposal_group_id`` explicitly.  This before-validator converts the
    model-boundary keyed object into that canonical representation before the
    inherited ProposalScreeningOutput validators run.
    """

    expected_proposal_group_ids: ClassVar[tuple[str, ...]] = ()
    expected_target_label_ids: ClassVar[tuple[str, ...]] = ()

    @model_validator(mode="before")
    @classmethod
    def keyed_decisions_to_canonical(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        decisions = value.get("decisions")
        if not isinstance(decisions, dict):
            return value

        expected = cls.expected_proposal_group_ids
        if not expected:
            raise RuntimeError(
                "_KeyedProposalScreeningOutput was used without expected proposal IDs"
            )

        returned_ids = set(decisions)
        expected_ids = set(expected)
        if returned_ids != expected_ids:
            raise ValueError(
                "Proposal screening key coverage mismatch; "
                f"missing={sorted(expected_ids - returned_ids)}, "
                f"unknown={sorted(returned_ids - expected_ids)}"
            )

        canonical_decisions: list[dict[str, Any]] = []
        # Use caller-supplied order rather than model-emitted object-key order so
        # the canonical internal result is deterministic.
        for proposal_group_id in expected:
            decision = decisions[proposal_group_id]
            if not isinstance(decision, dict):
                raise TypeError(f"Decision for {proposal_group_id!r} must be an object")
            if "proposal_group_id" in decision:
                raise ValueError(
                    "Model-boundary screening decisions must not contain "
                    "proposal_group_id; the proposal ID is the decisions object key"
                )
            canonical_decisions.append(
                {
                    "proposal_group_id": proposal_group_id,
                    **decision,
                }
            )

        converted = dict(value)
        converted["decisions"] = canonical_decisions
        return converted

    @model_validator(mode="after")
    def target_label_ids_must_be_allowed(self) -> "_KeyedProposalScreeningOutput":
        allowed = set(self.expected_target_label_ids)
        if not allowed:
            raise RuntimeError(
                "_KeyedProposalScreeningOutput was used without expected target label IDs"
            )
        for decision in self.decisions:
            if (
                decision.action == "map_existing"
                and decision.target_label_id not in allowed
            ):
                raise ValueError(
                    f"Proposal screening mapped to unknown target label "
                    f"{decision.target_label_id!r}"
                )
        return self


def _keyed_proposal_screening_model(
    expected_proposal_group_ids: list[str],
    expected_target_label_ids: list[str],
) -> type[_KeyedProposalScreeningOutput]:
    """Create a boundary model that knows the exact required IDs."""

    return type(
        "KeyedProposalScreeningOutput",
        (_KeyedProposalScreeningOutput,),
        {
            "expected_proposal_group_ids": tuple(expected_proposal_group_ids),
            "expected_target_label_ids": tuple(expected_target_label_ids),
            "__module__": __name__,
        },
    )


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
    """Build a screening schema with one required object key per proposal.

    Using proposal IDs as JSON object keys makes duplicate proposal decisions
    structurally impossible in the normal constrained-decoding path.  This
    avoids relying on ``uniqueItems``, which is not available in the vLLM
    0.19.1 XGrammar path used by this pipeline.

    The returned model type converts this boundary representation back into
    the canonical ProposalScreeningOutput list representation, so schemas.py,
    taxonomy history, and downstream application logic do not need to change.
    """

    _require_unique_nonempty(
        valid_proposal_group_ids, field_name="valid_proposal_group_ids"
    )
    _require_unique_nonempty(
        valid_target_label_ids, field_name="valid_target_label_ids"
    )

    keep_candidate = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "const": "keep_candidate",
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    }
    discard_candidate = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "const": "discard_candidate",
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    }
    map_existing = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "const": "map_existing",
            },
            "target_label_id": {
                "type": "string",
                "enum": valid_target_label_ids,
            },
        },
        "required": ["action", "target_label_id"],
        "additionalProperties": False,
    }

    decision_schema = {
        "oneOf": [
            keep_candidate,
            discard_candidate,
            map_existing,
        ]
    }

    schema = {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "object",
                "properties": {
                    proposal_group_id: decision_schema
                    for proposal_group_id in valid_proposal_group_ids
                },
                "required": list(valid_proposal_group_ids),
                "additionalProperties": False,
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }

    boundary_model = _keyed_proposal_screening_model(
        valid_proposal_group_ids, valid_target_label_ids
    )

    return StructuredSchemaSpec(
        model_type=boundary_model,
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
    # Bound repetition even though uniqueItems is unavailable in the vLLM
    # 0.19.1 XGrammar path. Pydantic/Python still enforce uniqueness.
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
