from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator  # type: ignore[import]
    
class StrictModel(BaseModel):
    """Base class for data that must validate without silent coercion."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

class LabelDef(StrictModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    definition: str = Field(min_length=1)


class DynamicLabelDef(LabelDef):
    created_version: int = Field(ge=1)
    last_modified_version: int = Field(ge=1)
    usage_count: int = Field(default=0, ge=0)
    proposal_support_count: int = Field(default=0, ge=0)


class ProposedLabel(StrictModel):
    name: str = Field(min_length=1, max_length=50)
    definition: str = Field(min_length=1, max_length=500)


class ProposedLabelRecord(ProposedLabel):
    proposal_id: str = Field(min_length=1)
    

class LabellingOutputSchema(StrictModel):
    """
    Output used during taxonomy discovery.

    Both top-level fields are intentionally required. Missing fields are an
    invalid model response rather than something the pipeline repairs.
    """

    assigned_label_ids: list[str] = Field(max_length=10)
    proposed_labels: list[ProposedLabel] = Field(max_length=3)

    @field_validator("assigned_label_ids")
    @classmethod
    def assigned_ids_must_be_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("assigned_label_ids contains duplicates")
        return value

    @model_validator(mode="after")
    def require_some_output(self) -> "LabellingOutputSchema":
        if not self.assigned_label_ids and not self.proposed_labels:
            raise ValueError(
                "At least one assigned label or proposed label is required"
            )
        return self


class FrozenClassificationOutputSchema(StrictModel):
    """Output used for the final pass against a frozen taxonomy."""

    assigned_label_ids: list[str] = Field(min_length=1)

    @field_validator("assigned_label_ids")
    @classmethod
    def assigned_ids_must_be_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("assigned_label_ids contains duplicates")
        return value


class DiscoveryRecord(StrictModel):
    seq: int = Field(ge=1)
    doc_id: str = Field(min_length=1)
    taxonomy_version: int = Field(ge=1)
    assigned_label_ids: list[str]
    proposed_labels: list[ProposedLabelRecord]


class FinalClassificationRecord(StrictModel):
    seq: int = Field(ge=1)
    doc_id: str = Field(min_length=1)
    taxonomy_version: int = Field(ge=1)
    taxonomy_hash: str = Field(min_length=1)
    assigned_label_ids: list[str] = Field(min_length=1)


class ProposalGroup(StrictModel):
    group_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    support_count: int = Field(ge=1)


class DeferResolution(StrictModel):
    """Keep one proposal group pending until more evidence accumulates."""

    action: Literal["defer"]
    proposal_group_ids: list[str] = Field(min_length=1, max_length=1)

    @field_validator("proposal_group_ids")
    @classmethod
    def group_ids_must_be_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("proposal_group_ids contains duplicates")
        return value


class RejectResolution(StrictModel):
    """Reject one or more proposal groups as unsuitable taxonomy categories."""

    action: Literal["reject"]
    proposal_group_ids: list[str] = Field(min_length=1)

    @field_validator("proposal_group_ids")
    @classmethod
    def group_ids_must_be_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("proposal_group_ids contains duplicates")
        return value


class MapExistingResolution(StrictModel):
    """Map one or more proposal groups to the same existing active label."""

    action: Literal["map_existing"]
    proposal_group_ids: list[str] = Field(min_length=1)
    target_label_id: str = Field(min_length=1)

    @field_validator("proposal_group_ids")
    @classmethod
    def group_ids_must_be_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("proposal_group_ids contains duplicates")
        return value


class CreateResolution(StrictModel):
    """Create one reusable label from one or more semantically equivalent groups."""

    action: Literal["create"]
    proposal_group_ids: list[str] = Field(min_length=1)
    name: str = Field(min_length=1)
    definition: str = Field(min_length=1)

    @field_validator("proposal_group_ids")
    @classmethod
    def group_ids_must_be_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("proposal_group_ids contains duplicates")
        return value


ProposalResolution = Annotated[
    DeferResolution | RejectResolution | MapExistingResolution | CreateResolution,
    Field(discriminator="action"),
]


class MergeDynamicLabel(StrictModel):
    """Retire a dynamic label into an active seed or dynamic label."""

    action: Literal["merge"]
    source_label_id: str = Field(min_length=1)
    target_label_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def source_and_target_must_differ(self) -> "MergeDynamicLabel":
        if self.target_label_id == self.source_label_id:
            raise ValueError("a label cannot be merged into itself")
        return self


class ReviseDynamicLabel(StrictModel):
    """Rewrite the name/definition of a dynamic label without changing its meaning."""

    action: Literal["revise"]
    source_label_id: str = Field(min_length=1)
    # Return the complete replacement label on every revision. This avoids
    # conditional optional fields that structured decoding cannot reliably
    # constrain from a post-validation rule alone.
    name: str = Field(min_length=1)
    definition: str = Field(min_length=1)


DynamicLabelChange = Annotated[
    MergeDynamicLabel | ReviseDynamicLabel,
    Field(discriminator="action"),
]


class ReconciliationOutput(StrictModel):
    proposal_resolutions: list[ProposalResolution]
    dynamic_label_changes: list[DynamicLabelChange]

    @model_validator(mode="after")
    def reconciliation_items_must_be_unique(self) -> "ReconciliationOutput":
        # This does not know which proposal groups were supplied by the caller;
        # validate_reconciliation_semantics() must still enforce exact coverage.
        group_ids = [
            group_id
            for resolution in self.proposal_resolutions
            for group_id in resolution.proposal_group_ids
        ]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError(
                "A proposal group may be resolved at most once per reconciliation"
            )

        sources = [change.source_label_id for change in self.dynamic_label_changes]
        if len(sources) != len(set(sources)):
            raise ValueError(
                "A dynamic label may be changed at most once per reconciliation"
            )
        return self


class DiscoverySettings(StrictModel):
    input_path: str
    input_size_bytes: int = Field(ge=0)
    input_mtime_ns: int = Field(ge=0)
    model_name: str
    vllm_version: str
    batch_size: int = Field(ge=1)
    max_model_len: int = Field(ge=1)
    max_document_tokens: int = Field(ge=1)
    classification_max_tokens: int = Field(ge=1)
    reconcile_max_tokens: int = Field(ge=1)
    tensor_parallel_size: int = Field(ge=1)
    gpu_memory_utilization: float = Field(gt=0.0, le=1.0)
    dtype: str
    seed: int
    thinking_mode: Literal["disabled", "enabled", "template-default"]
    reconcile_every: int = Field(ge=1)
    max_reconcile_groups: int = Field(ge=1)
    min_create_support: int = Field(ge=1)
    max_proposals_per_doc: int = Field(ge=1)
    max_total_labels: int = Field(ge=1)
    expected_seed_label_count: int = Field(ge=1)


class SchemaHistoryEntry(StrictModel):
    timestamp_unix: int = Field(ge=0)
    maintenance_kind: Literal["periodic", "final"]
    discovery_seq_end: int = Field(ge=0)
    version_before: int = Field(ge=1)
    version_after: int = Field(ge=1)
    proposal_groups: list[ProposalGroup]
    reconciliation: ReconciliationOutput
    created_labels: list[DynamicLabelDef]


class TaxonomyState(StrictModel):
    format_version: int = 1
    aspect: str = Field(min_length=1)
    schema_version: int = Field(default=1, ge=1)
    frozen: bool = False
    frozen_at_unix: int | None = Field(default=None, ge=0)
    frozen_taxonomy_hash: str | None = None

    seed_labels_sha256: str = Field(min_length=1)
    seed_labels: list[LabelDef]
    dynamic_labels: list[DynamicLabelDef] = Field(default_factory=list)
    alias_map: dict[str, str] = Field(default_factory=dict)
    deferred_proposals: list[ProposalGroup] = Field(default_factory=list)
    next_dynamic_label_num: int = Field(default=25, ge=1)

    last_reconciled_discovery_seq: int = Field(default=0, ge=0)
    discovery_settings: DiscoverySettings
    history: list[SchemaHistoryEntry] = Field(default_factory=list)

    created_at_unix: int = Field(ge=0)
    updated_at_unix: int = Field(ge=0)


class ClassificationRunMetadata(StrictModel):
    mode: Literal["classify"] = "classify"
    input_path: str
    input_size_bytes: int = Field(ge=0)
    input_mtime_ns: int = Field(ge=0)
    taxonomy_path: str
    taxonomy_version: int = Field(ge=1)
    taxonomy_hash: str = Field(min_length=1)
    model_name: str
    vllm_version: str
    batch_size: int = Field(ge=1)
    max_model_len: int = Field(ge=1)
    max_document_tokens: int = Field(ge=1)
    classification_max_tokens: int = Field(ge=1)
    tensor_parallel_size: int = Field(ge=1)
    gpu_memory_utilization: float = Field(gt=0.0, le=1.0)
    dtype: str
    seed: int
    thinking_mode: Literal["disabled", "enabled", "template-default"]
    created_at_unix: int = Field(ge=0)


class InputDocument(BaseModel):
    """
    Input JSONL record.

    Extra metadata fields are allowed, but doc_id and text are mandatory and
    strictly typed. This lets the corpus carry metadata without the labelling
    pipeline silently accepting malformed core fields.
    """

    model_config = ConfigDict(
        extra="allow",
        strict=True,
        str_strip_whitespace=False,
    )

    doc_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


FinalProposalResolution = Annotated[
    RejectResolution | MapExistingResolution | CreateResolution,
    Field(discriminator="action"),
]


class FinalReconciliationOutput(StrictModel):
    proposal_resolutions: list[FinalProposalResolution]
    dynamic_label_changes: list[DynamicLabelChange]

    @model_validator(mode="after")
    def reconciliation_items_must_be_unique(
        self,
    ) -> "FinalReconciliationOutput":

        group_ids = [
            group_id
            for resolution in self.proposal_resolutions
            for group_id in resolution.proposal_group_ids
        ]

        if len(group_ids) != len(set(group_ids)):
            raise ValueError(
                "A proposal group may be resolved at most once " "per reconciliation"
            )

        sources = [change.source_label_id for change in self.dynamic_label_changes]

        if len(sources) != len(set(sources)):
            raise ValueError(
                "A dynamic label may be changed at most once " "per reconciliation"
            )

        return self
