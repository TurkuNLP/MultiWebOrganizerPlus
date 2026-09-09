from __future__ import annotations

from typing import Annotated, Literal, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator  # type: ignore[import]


class StrictModel(BaseModel):
    """Base class for data that must validate without silent coercion."""

    # Constrain the name and definition fields to start with a letter
    # and contain only letters, digits, spaces, and a limited set of punctuation.
    # This should limit the risk of malformed or nonsensical labels.
    NAME_AND_DEFINITION_PATTERN: ClassVar[str] = r"^[A-Za-z][A-Za-z0-9 .,()'&/+\-:;]*$"
    LABEL_NAME_MAX_LENGTH: ClassVar[int] = 80
    LABEL_DEFINITION_MAX_LENGTH: ClassVar[int] = 500

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _proposal_signature(proposal: "ProposedLabel") -> tuple[str, str]:
    return (
        _normalized_text(proposal.name),
        _normalized_text(proposal.definition),
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
    name: str = Field(
        min_length=1,
        max_length=StrictModel.LABEL_NAME_MAX_LENGTH,
        pattern=StrictModel.NAME_AND_DEFINITION_PATTERN,
    )
    definition: str = Field(
        min_length=1,
        max_length=StrictModel.LABEL_DEFINITION_MAX_LENGTH,
        pattern=StrictModel.NAME_AND_DEFINITION_PATTERN,
    )


class ProposedLabelRecord(ProposedLabel):
    proposal_id: str = Field(min_length=1)


class LabellingOutputSchema(StrictModel):
    """Output used during taxonomy discovery."""

    assigned_label_ids: list[str]
    proposed_labels: list[ProposedLabel]

    @field_validator("assigned_label_ids")
    @classmethod
    def assigned_ids_must_be_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("assigned_label_ids contains duplicates")
        return value

    @model_validator(mode="after")
    def validate_output(self) -> "LabellingOutputSchema":
        if not self.assigned_label_ids and not self.proposed_labels:
            raise ValueError(
                "At least one assigned label or proposed label is required"
            )
        signatures = [_proposal_signature(label) for label in self.proposed_labels]
        if len(signatures) != len(set(signatures)):
            raise ValueError(
                "proposed_labels contains duplicate normalized name/definition pairs"
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

    @model_validator(mode="after")
    def record_items_must_be_unique(self) -> "DiscoveryRecord":
        if len(self.assigned_label_ids) != len(set(self.assigned_label_ids)):
            raise ValueError("assigned_label_ids contains duplicates")
        proposal_ids = [proposal.proposal_id for proposal in self.proposed_labels]
        if len(proposal_ids) != len(set(proposal_ids)):
            raise ValueError("proposed_labels contains duplicate proposal_id values")
        signatures = [_proposal_signature(label) for label in self.proposed_labels]
        if len(signatures) != len(set(signatures)):
            raise ValueError(
                "proposed_labels contains duplicate normalized name/definition pairs"
            )
        if not self.assigned_label_ids and not self.proposed_labels:
            raise ValueError(
                "DiscoveryRecord must contain at least one assigned or proposed label"
            )
        return self


class FinalClassificationRecord(StrictModel):
    seq: int = Field(ge=1)
    doc_id: str = Field(min_length=1)
    taxonomy_version: int = Field(ge=1)
    taxonomy_hash: str = Field(min_length=1)
    assigned_label_ids: list[str] = Field(min_length=1)

    @field_validator("assigned_label_ids")
    @classmethod
    def assigned_ids_must_be_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("assigned_label_ids contains duplicates")
        return value


class ProposalGroup(StrictModel):
    """Deterministically grouped proposal evidence used during discovery."""

    group_id: str = Field(min_length=1)
    name: str = Field(
        min_length=1,
        max_length=StrictModel.LABEL_NAME_MAX_LENGTH,
        pattern=StrictModel.NAME_AND_DEFINITION_PATTERN,
    )
    definition: str = Field(
        min_length=1,
        max_length=StrictModel.LABEL_DEFINITION_MAX_LENGTH,
        pattern=StrictModel.NAME_AND_DEFINITION_PATTERN,
    )
    support_count: int = Field(ge=1)


# ---------------------------------------------------------------------------
# 1. Proposal screening
# ---------------------------------------------------------------------------


class KeepCandidateDecision(StrictModel):
    """Keep a proposal group as evidence for a potentially missing category."""

    action: Literal["keep_candidate"]
    proposal_group_id: str = Field(min_length=1)


class DiscardCandidateDecision(StrictModel):
    """Discard a proposal group as unsuitable for the taxonomy."""

    action: Literal["discard_candidate"]
    proposal_group_id: str = Field(min_length=1)


class MapExistingDecision(StrictModel):
    """Resolve a proposal group to an already-active taxonomy label."""

    action: Literal["map_existing"]
    proposal_group_id: str = Field(min_length=1)
    target_label_id: str = Field(min_length=1)


ProposalScreeningDecision = Annotated[
    KeepCandidateDecision | DiscardCandidateDecision | MapExistingDecision,
    Field(discriminator="action"),
]


class ProposalScreeningOutput(StrictModel):
    decisions: list[ProposalScreeningDecision]

    @model_validator(mode="after")
    def proposal_groups_must_be_unique(self) -> "ProposalScreeningOutput":
        group_ids = [decision.proposal_group_id for decision in self.decisions]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError(
                "A proposal group may be screened at most once per screening pass"
            )
        return self


# ---------------------------------------------------------------------------
# 2. Candidate promotion
# ---------------------------------------------------------------------------


class PromotedCandidateLabel(StrictModel):
    """One new label defined from one or more semantically equivalent candidates."""

    candidate_group_ids: list[str] = Field(min_length=1)
    name: str = Field(
        min_length=1,
        max_length=StrictModel.LABEL_NAME_MAX_LENGTH,
        pattern=StrictModel.NAME_AND_DEFINITION_PATTERN,
    )
    definition: str = Field(
        min_length=1,
        max_length=StrictModel.LABEL_DEFINITION_MAX_LENGTH,
        pattern=StrictModel.NAME_AND_DEFINITION_PATTERN,
    )

    @field_validator("candidate_group_ids")
    @classmethod
    def candidate_ids_must_be_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("candidate_group_ids contains duplicates")
        return value


class CandidatePromotionOutput(StrictModel):
    new_labels: list[PromotedCandidateLabel] = Field(min_length=1)

    @model_validator(mode="after")
    def candidate_groups_must_be_unique(self) -> "CandidatePromotionOutput":
        group_ids = [
            group_id
            for label in self.new_labels
            for group_id in label.candidate_group_ids
        ]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError(
                "A candidate group may contribute to at most one promoted label"
            )
        return self


# ---------------------------------------------------------------------------
# 3. Final merge pass
# ---------------------------------------------------------------------------


class FinalMergeDecision(StrictModel):
    """Retire one dynamic label into another active seed or dynamic label."""

    source_label_id: str = Field(min_length=1)
    target_label_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def source_and_target_must_differ(self) -> "FinalMergeDecision":
        if self.source_label_id == self.target_label_id:
            raise ValueError("a label cannot be merged into itself")
        return self


class FinalMergeOutput(StrictModel):
    merges: list[FinalMergeDecision]

    @model_validator(mode="after")
    def merge_sources_must_be_unique(self) -> "FinalMergeOutput":
        sources = [merge.source_label_id for merge in self.merges]
        if len(sources) != len(set(sources)):
            raise ValueError("A dynamic label may be merged at most once")
        return self


# ---------------------------------------------------------------------------
# 4. Final revision pass
# ---------------------------------------------------------------------------


class FinalRevisionDecision(StrictModel):
    """Clarify one surviving dynamic label without changing its semantic category."""

    label_id: str = Field(min_length=1)
    name: str = Field(
        min_length=1,
        max_length=StrictModel.LABEL_NAME_MAX_LENGTH,
        pattern=StrictModel.NAME_AND_DEFINITION_PATTERN,
    )
    definition: str = Field(
        min_length=1,
        max_length=StrictModel.LABEL_DEFINITION_MAX_LENGTH,
        pattern=StrictModel.NAME_AND_DEFINITION_PATTERN,
    )


class FinalRevisionOutput(StrictModel):
    revisions: list[FinalRevisionDecision]

    @model_validator(mode="after")
    def revised_labels_must_be_unique(self) -> "FinalRevisionOutput":
        label_ids = [revision.label_id for revision in self.revisions]
        if len(label_ids) != len(set(label_ids)):
            raise ValueError("A dynamic label may be revised at most once")
        return self


# ---------------------------------------------------------------------------
# Run configuration and persistent state
# ---------------------------------------------------------------------------


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
    max_assigned_labels_per_doc: int = Field(ge=1)
    maintenance_max_tokens: int = Field(ge=1)
    tensor_parallel_size: int = Field(ge=1)
    gpu_memory_utilization: float = Field(gt=0.0, le=1.0)
    dtype: str
    seed: int
    thinking_mode: Literal["disabled", "enabled", "template-default"]
    screen_every: int = Field(ge=1)
    max_screening_groups: int = Field(ge=1)
    max_promotion_groups: int = Field(ge=1)
    max_revision_labels: int = Field(ge=1)
    min_promotion_support: int = Field(ge=1)
    max_proposals_per_doc: int = Field(ge=1)
    max_total_labels: int = Field(ge=1)
    expected_seed_label_count: int = Field(ge=1)
    creativity: Literal["low", "high"] = "low"


class ProposalScreeningHistoryEntry(StrictModel):
    kind: Literal["proposal_screening"] = "proposal_screening"
    timestamp_unix: int = Field(ge=0)
    discovery_seq_end: int = Field(ge=0)
    taxonomy_version: int = Field(ge=1)
    proposal_groups: list[ProposalGroup]
    screening: ProposalScreeningOutput


class CandidatePromotionHistoryEntry(StrictModel):
    kind: Literal["candidate_promotion"] = "candidate_promotion"
    timestamp_unix: int = Field(ge=0)
    discovery_seq_end: int = Field(ge=0)
    version_before: int = Field(ge=1)
    version_after: int = Field(ge=1)
    candidate_groups: list[ProposalGroup]
    promotion: CandidatePromotionOutput
    created_labels: list[DynamicLabelDef]


class FinalMergeHistoryEntry(StrictModel):
    kind: Literal["final_merge"] = "final_merge"
    timestamp_unix: int = Field(ge=0)
    version_before: int = Field(ge=1)
    version_after: int = Field(ge=1)
    result: FinalMergeOutput


class FinalRevisionHistoryEntry(StrictModel):
    kind: Literal["final_revision"] = "final_revision"
    timestamp_unix: int = Field(ge=0)
    version_before: int = Field(ge=1)
    version_after: int = Field(ge=1)
    result: FinalRevisionOutput


MaintenanceHistoryEntry = Annotated[
    ProposalScreeningHistoryEntry
    | CandidatePromotionHistoryEntry
    | FinalMergeHistoryEntry
    | FinalRevisionHistoryEntry,
    Field(discriminator="kind"),
]


class TaxonomyState(StrictModel):
    # Version 3 records all model-affecting maintenance bounds and uses the
    # audited split-maintenance state semantics.
    format_version: Literal[3] = 3
    aspect: str = Field(min_length=1)
    schema_version: int = Field(default=1, ge=1)
    frozen: bool = False
    frozen_at_unix: int | None = Field(default=None, ge=0)
    frozen_taxonomy_hash: str | None = None

    seed_labels_sha256: str = Field(min_length=1)
    seed_labels: list[LabelDef]
    dynamic_labels: list[DynamicLabelDef] = Field(default_factory=list)
    alias_map: dict[str, str] = Field(default_factory=dict)

    # Proposal groups that passed screening but have not yet been promoted.
    candidate_proposals: list[ProposalGroup] = Field(default_factory=list)
    next_dynamic_label_num: int = Field(default=25, ge=1)

    last_screened_discovery_seq: int = Field(default=0, ge=0)
    discovery_settings: DiscoverySettings
    history: list[MaintenanceHistoryEntry] = Field(default_factory=list)

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
    max_assigned_labels_per_doc: int = Field(ge=1)
    tensor_parallel_size: int = Field(ge=1)
    gpu_memory_utilization: float = Field(gt=0.0, le=1.0)
    dtype: str
    seed: int
    thinking_mode: Literal["disabled", "enabled", "template-default"]
    created_at_unix: int = Field(ge=0)


class InputDocument(BaseModel):
    """Input JSONL record with strict core fields and arbitrary metadata."""

    model_config = ConfigDict(
        extra="allow",
        strict=True,
        str_strip_whitespace=False,
    )

    doc_id: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @field_validator("doc_id")
    @classmethod
    def doc_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("doc_id must not be blank")
        return value

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value
