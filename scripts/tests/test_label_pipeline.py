from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from label_pipeline import (  # noqa: E402
    apply_reconciliation,
    combine_proposal_groups,
    seed_labels_hash,
    validate_reconciliation_semantics,
    validate_taxonomy_state,
)
from schemas import (  # noqa: E402
    DiscoverySettings,
    CreateResolution,
    DeferResolution,
    MergeDynamicLabel,
    FrozenClassificationOutputSchema,
    LabelDef,
    LabellingOutputSchema,
    ProposalGroup,
    ReconciliationOutput,
    TaxonomyState,
    FinalReconciliationOutput
)


def make_state() -> TaxonomyState:
    seed_labels = [
        LabelDef(id=f"L{i:03d}", name=f"Seed {i}", definition=f"Definition {i}")
        for i in range(1, 25)
    ]
    settings = DiscoverySettings(
        input_path="/tmp/input.jsonl",
        input_size_bytes=1,
        input_mtime_ns=1,
        model_name="test-model",
        vllm_version="0.22.1",
        batch_size=4,
        max_model_len=32768,
        max_document_tokens=8192,
        classification_max_tokens=512,
        reconcile_max_tokens=4096,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype="auto",
        seed=0,
        thinking_mode="disabled",
        reconcile_every=64,
        max_reconcile_groups=128,
        min_create_support=2,
        max_proposals_per_doc=3,
        max_total_labels=100,
        expected_seed_label_count=24,
    )
    now = int(time.time())
    return TaxonomyState(
        aspect="topics",
        seed_labels_sha256=seed_labels_hash(seed_labels),
        seed_labels=seed_labels,
        discovery_settings=settings,
        created_at_unix=now,
        updated_at_unix=now,
    )


def test_discovery_output_missing_field_fails() -> None:
    with pytest.raises(ValidationError):
        LabellingOutputSchema.model_validate_json('{"assigned_label_ids":["L001"]}')


def test_discovery_output_duplicates_fail() -> None:
    with pytest.raises(ValidationError):
        LabellingOutputSchema.model_validate_json(
            '{"assigned_label_ids":["L001","L001"],"proposed_labels":[]}'
        )


def test_frozen_output_requires_at_least_one_label() -> None:
    with pytest.raises(ValidationError):
        FrozenClassificationOutputSchema.model_validate_json(
            '{"assigned_label_ids":[]}'
        )


def test_deferred_proposal_support_accumulates_then_creates() -> None:
    state = make_state()
    group = ProposalGroup(
        group_id="G1",
        name="Quantum computing",
        definition="Documents centrally about quantum computing.",
        support_count=1,
    )
    deferred = ReconciliationOutput(
        proposal_resolutions=[
            DeferResolution(action="defer", proposal_group_ids=["G1"])
        ],
        dynamic_label_changes=[],
    )
    state = apply_reconciliation(
        state,
        [group],
        deferred,
        discovery_seq_end=64,
        max_total_labels=100,
        min_create_support=2,
        maintenance_kind="periodic",
    )
    assert state.deferred_proposals[0].support_count == 1

    groups = combine_proposal_groups(
        state.deferred_proposals,
        [group],
    )
    assert groups[0].support_count == 2

    create = ReconciliationOutput(
        proposal_resolutions=[
            CreateResolution(
                action="create",
                proposal_group_ids=["G1"],
                name="Quantum computing",
                definition="Documents centrally about quantum computing.",
            )
        ],
        dynamic_label_changes=[],
    )
    state = apply_reconciliation(
        state,
        groups,
        create,
        discovery_seq_end=128,
        max_total_labels=100,
        min_create_support=2,
        maintenance_kind="periodic",
    )
    assert state.dynamic_labels[0].id == "L025"
    assert state.dynamic_labels[0].proposal_support_count == 2
    assert state.deferred_proposals == []


def test_seed_label_cannot_be_merge_source() -> None:
    state = make_state()
    result = ReconciliationOutput(
        proposal_resolutions=[],
        dynamic_label_changes=[
            MergeDynamicLabel(
                action="merge",
                source_label_id="L001",
                target_label_id="L002",
            )
        ],
    )
    with pytest.raises(ValueError):
        validate_reconciliation_semantics(
            state,
            [],
            result,
            min_create_support=2,
            allow_defer=True,
        )


def test_final_maintenance_cannot_defer() -> None:
    state = make_state()
    group = ProposalGroup(
        group_id="G1",
        name="Candidate",
        definition="Candidate definition",
        support_count=1,
    )
    result = ReconciliationOutput(
        proposal_resolutions=[
            DeferResolution(action="defer", proposal_group_ids=["G1"])
        ],
        dynamic_label_changes=[],
    )
    with pytest.raises(ValueError):
        validate_reconciliation_semantics(
            state,
            [group],
            result,
            min_create_support=2,
            allow_defer=False,
        )


def test_initial_state_invariants() -> None:
    state = make_state()
    validate_taxonomy_state(state)
