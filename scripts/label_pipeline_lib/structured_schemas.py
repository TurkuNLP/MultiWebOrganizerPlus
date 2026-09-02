from dataclasses import dataclass
from typing import Generic, TypeVar
from schemas import (
    LabellingOutputSchema,
    FrozenClassificationOutputSchema,
    ReconciliationOutput,
    FinalReconciliationOutput,
)

from pydantic import BaseModel  # type: ignore

SchemaT = TypeVar(
    "SchemaT",
    bound=BaseModel,
)


@dataclass(frozen=True)
class StructuredSchemaSpec(Generic[SchemaT]):
    model_type: type[SchemaT]
    json_schema: dict


def build_discovery_output_schema(
    valid_label_ids: list[str],
    max_assigned_labels: int,
    max_proposed_labels: int,
) -> StructuredSchemaSpec:

    if not valid_label_ids:
        raise ValueError("valid_label_ids must not be empty")

    schema = LabellingOutputSchema.model_json_schema()

    assigned = schema["properties"]["assigned_label_ids"]

    assigned["items"] = {
        "type": "string",
        "enum": valid_label_ids,
    }
    assigned["maxItems"] = max_assigned_labels

    proposed = schema["properties"]["proposed_labels"]
    proposed["maxItems"] = max_proposed_labels

    return StructuredSchemaSpec(
        model_type=LabellingOutputSchema,
        json_schema=schema,
    )


def build_classification_output_schema(
    valid_label_ids: list[str],
    max_assigned_labels: int,
) -> StructuredSchemaSpec:

    if not valid_label_ids:
        raise ValueError("valid_label_ids must not be empty")

    schema = FrozenClassificationOutputSchema.model_json_schema()

    assigned = schema["properties"]["assigned_label_ids"]

    assigned["items"] = {
        "type": "string",
        "enum": valid_label_ids,
    }
    assigned["maxItems"] = max_assigned_labels

    return StructuredSchemaSpec(
        model_type=FrozenClassificationOutputSchema,
        json_schema=schema,
    )


def build_reconciliation_output_schema(
    final_maintenance: bool,
    valid_proposal_group_ids: list[str] | None = None,
) -> StructuredSchemaSpec:

    schema_type = (
        FinalReconciliationOutput if final_maintenance else ReconciliationOutput
    )

    schema = schema_type.model_json_schema()

    if valid_proposal_group_ids is not None:
        if not valid_proposal_group_ids:
            raise ValueError("valid_proposal_group_ids must not be empty")

        if len(valid_proposal_group_ids) != len(set(valid_proposal_group_ids)):
            raise ValueError("valid_proposal_group_ids contains duplicates")

        resolution_defs = [
            "RejectResolution",
            "MapExistingResolution",
            "CreateResolution",
        ]

        if not final_maintenance:
            resolution_defs.append("DeferResolution")

        for def_name in resolution_defs:
            try:
                group_ids_schema = schema["$defs"][def_name]["properties"][
                    "proposal_group_ids"
                ]
            except KeyError as exc:
                raise RuntimeError(
                    f"Unexpected reconciliation JSON schema: "
                    f"could not find proposal_group_ids in {def_name}"
                ) from exc

            group_ids_schema["items"] = {
                "type": "string",
                "enum": valid_proposal_group_ids,
            }

    return StructuredSchemaSpec(
        model_type=schema_type,
        json_schema=schema,
    )
