from __future__ import annotations

import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

# The orchestration module imports vLLM at module import time. These tests exercise
# deterministic Python state transitions only, so provide a minimal import stub on
# development machines that do not have vLLM installed. On the HPC environment,
# the real vLLM package is used.
if importlib.util.find_spec("vllm") is None:
    vllm_stub = types.ModuleType("vllm")
    vllm_stub.__version__ = "test-stub"
    vllm_stub.LLM = object
    vllm_stub.SamplingParams = object
    sys.modules["vllm"] = vllm_stub

    sampling_params_stub = types.ModuleType("vllm.sampling_params")
    sampling_params_stub.StructuredOutputsParams = object
    sys.modules["vllm.sampling_params"] = sampling_params_stub

from label_pipeline_lib.cli import build_arg_parser, validate_cli_args  # noqa: E402
import label_pipeline_lib.classification as classification_module  # noqa: E402
import label_pipeline_lib.discovery as discovery_module  # noqa: E402
from label_pipeline_lib.common import (  # noqa: E402
    append_models_jsonl,
    atomic_write_model,
    configure_logging,
    load_model_file,
    seed_labels_hash,
    validate_taxonomy_state,
)
from label_pipeline_lib.compute_logger import detect_num_gpus  # noqa: E402
from label_pipeline_lib.discovery import (  # noqa: E402
    accumulate_existing_candidate_support,
    apply_candidate_promotion,
    apply_final_merge,
    apply_final_revision,
    apply_proposal_screening,
    run_final_maintenance,
    screen_new_proposals,
    validate_candidate_promotion_semantics,
    validate_final_merge_semantics,
    validate_final_revision_semantics,
    validate_proposal_screening_semantics,
)
from label_pipeline_lib.schemas import (  # noqa: E402
    CandidatePromotionOutput,
    DiscardCandidateDecision,
    DiscoveryRecord,
    DiscoverySettings,
    DynamicLabelDef,
    FinalMergeDecision,
    FinalMergeOutput,
    FinalRevisionDecision,
    FinalRevisionOutput,
    FinalClassificationRecord,
    FrozenClassificationOutputSchema,
    InputDocument,
    KeepCandidateDecision,
    LabelDef,
    LabellingOutputSchema,
    MapExistingDecision,
    PromotedCandidateLabel,
    ProposalGroup,
    ProposedLabelRecord,
    ProposalScreeningOutput,
    TaxonomyState,
)
from label_pipeline_lib.structured_schemas import (  # noqa: E402
    build_proposal_screening_output_schema,
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
        max_assigned_labels_per_doc=5,
        maintenance_max_tokens=4096,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype="auto",
        seed=0,
        thinking_mode="disabled",
        screen_every=64,
        max_screening_groups=128,
        max_promotion_groups=128,
        max_revision_labels=16,
        min_promotion_support=2,
        max_proposals_per_doc=3,
        max_total_labels=100,
        expected_seed_label_count=24,
    )
    now = int(time.time())
    return TaxonomyState(
        format_version=3,
        aspect="topics",
        seed_labels_sha256=seed_labels_hash(seed_labels),
        seed_labels=seed_labels,
        discovery_settings=settings,
        created_at_unix=now,
        updated_at_unix=now,
    )


def make_group(*, support_count: int = 1) -> ProposalGroup:
    return ProposalGroup(
        group_id="G1",
        name="Quantum computing",
        definition="Documents centrally about quantum computing.",
        support_count=support_count,
    )


def make_dynamic_label(
    *,
    label_id: str = "L025",
    name: str = "Quantum computing",
) -> DynamicLabelDef:
    return DynamicLabelDef(
        id=label_id,
        name=name,
        definition=f"Documents centrally about {name.lower()}.",
        created_version=2,
        last_modified_version=2,
        usage_count=3,
        proposal_support_count=2,
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


def test_screening_requires_each_group_at_most_once() -> None:
    with pytest.raises(ValidationError):
        ProposalScreeningOutput(
            decisions=[
                KeepCandidateDecision(action="keep_candidate", proposal_group_id="G1"),
                DiscardCandidateDecision(
                    action="discard_candidate", proposal_group_id="G1"
                ),
            ]
        )


def test_screening_semantics_require_exact_coverage() -> None:
    state = make_state()
    groups = [
        make_group(),
        ProposalGroup(
            group_id="G2",
            name="Robotics",
            definition="Documents centrally about robotics.",
            support_count=1,
        ),
    ]
    result = ProposalScreeningOutput(
        decisions=[
            KeepCandidateDecision(action="keep_candidate", proposal_group_id="G1")
        ]
    )
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_proposal_screening_semantics(state, groups, result)


def test_keep_candidate_support_accumulates_then_promotes() -> None:
    state = make_state()
    group = make_group(support_count=1)

    screening = ProposalScreeningOutput(
        decisions=[
            KeepCandidateDecision(action="keep_candidate", proposal_group_id="G1")
        ]
    )
    state = apply_proposal_screening(
        state,
        [group],
        screening,
        discovery_seq_end=64,
        max_total_labels=100,
    )
    assert state.candidate_proposals == [group]
    assert state.last_screened_discovery_seq == 64
    assert state.schema_version == 1

    state, groups_to_screen = accumulate_existing_candidate_support(state, [group])
    assert groups_to_screen == []
    assert state.candidate_proposals[0].support_count == 2

    promotion = CandidatePromotionOutput(
        new_labels=[
            PromotedCandidateLabel(
                candidate_group_ids=["G1"],
                name="Quantum computing",
                definition="Documents centrally about quantum computing.",
            )
        ]
    )
    state = apply_candidate_promotion(
        state,
        state.candidate_proposals,
        promotion,
        discovery_seq_end=128,
        min_promotion_support=2,
        max_new_labels=1,
        max_total_labels=100,
    )

    assert state.dynamic_labels[0].id == "L025"
    assert state.dynamic_labels[0].proposal_support_count == 2
    assert state.candidate_proposals == []
    assert state.schema_version == 2


def test_map_existing_removes_previously_kept_candidate() -> None:
    state = make_state().model_copy(update={"candidate_proposals": [make_group()]})
    result = ProposalScreeningOutput(
        decisions=[
            MapExistingDecision(
                action="map_existing",
                proposal_group_id="G1",
                target_label_id="L001",
            )
        ]
    )
    state = apply_proposal_screening(
        state,
        [make_group()],
        result,
        discovery_seq_end=64,
        max_total_labels=100,
    )
    assert state.candidate_proposals == []


def test_promotion_below_support_threshold_fails() -> None:
    state = make_state().model_copy(update={"candidate_proposals": [make_group()]})
    result = CandidatePromotionOutput(
        new_labels=[
            PromotedCandidateLabel(
                candidate_group_ids=["G1"],
                name="Quantum computing",
                definition="Documents centrally about quantum computing.",
            )
        ]
    )
    with pytest.raises(ValueError, match="below minimum"):
        validate_candidate_promotion_semantics(
            state,
            [make_group()],
            result,
            min_promotion_support=2,
            max_new_labels=1,
            max_total_labels=100,
        )


def test_seed_label_cannot_be_final_merge_source() -> None:
    state = make_state()
    result = FinalMergeOutput(
        merges=[FinalMergeDecision(source_label_id="L001", target_label_id="L002")]
    )
    with pytest.raises(ValueError, match="not a dynamic label"):
        validate_final_merge_semantics(state, result)


def test_final_merge_retires_dynamic_label_into_seed() -> None:
    state = make_state().model_copy(
        update={
            "schema_version": 2,
            "dynamic_labels": [make_dynamic_label()],
            "next_dynamic_label_num": 26,
        }
    )
    result = FinalMergeOutput(
        merges=[FinalMergeDecision(source_label_id="L025", target_label_id="L001")]
    )
    state = apply_final_merge(state, result, max_total_labels=100)

    assert state.dynamic_labels == []
    assert state.alias_map["L025"] == "L001"
    assert state.schema_version == 3
    validate_taxonomy_state(state)


def test_final_revision_can_only_change_dynamic_labels() -> None:
    state = make_state().model_copy(
        update={
            "schema_version": 2,
            "dynamic_labels": [make_dynamic_label()],
            "next_dynamic_label_num": 26,
        }
    )

    invalid = FinalRevisionOutput(
        revisions=[
            FinalRevisionDecision(
                label_id="L001",
                name="Changed seed",
                definition="This must not be allowed.",
            )
        ]
    )
    with pytest.raises(ValueError, match="non-dynamic"):
        validate_final_revision_semantics(state, invalid)

    valid = FinalRevisionOutput(
        revisions=[
            FinalRevisionDecision(
                label_id="L025",
                name="Quantum Computing",
                definition="Documents centrally concerned with quantum computing.",
            )
        ]
    )
    state = apply_final_revision(state, valid, max_total_labels=100)
    assert state.dynamic_labels[0].name == "Quantum Computing"
    assert state.dynamic_labels[0].last_modified_version == 3


def test_candidate_state_rejects_duplicate_group_ids() -> None:
    state = make_state().model_copy(
        update={"candidate_proposals": [make_group(), make_group()]}
    )
    with pytest.raises(ValueError, match="Duplicate candidate proposal group IDs"):
        validate_taxonomy_state(state)


def test_old_taxonomy_format_version_fails_validation() -> None:
    payload = make_state().model_dump(mode="json")
    payload["format_version"] = 2
    with pytest.raises(ValidationError):
        TaxonomyState.model_validate(payload)


def test_screening_json_schema_has_exact_decision_count_and_enums() -> None:
    spec = build_proposal_screening_output_schema(
        valid_proposal_group_ids=["P001", "P002"],
        valid_target_label_ids=["L001", "L002"],
    )
    decisions = spec.json_schema["properties"]["decisions"]
    assert decisions["minItems"] == 2
    assert decisions["maxItems"] == 2

    keep_id_schema = spec.json_schema["$defs"]["KeepCandidateDecision"]["properties"][
        "proposal_group_id"
    ]
    assert keep_id_schema["enum"] == ["P001", "P002"]


def test_configured_gpu_count_is_used_for_compute_accounting() -> None:
    assert detect_num_gpus(preferred=4) == 4


def test_classification_rejects_zero_max_assigned_labels() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--mode",
            "classify",
            "--input",
            "/tmp/input.jsonl",
            "--max-assigned-labels-per-doc",
            "0",
        ]
    )
    with pytest.raises(ValueError, match="max-assigned-labels-per-doc"):
        validate_cli_args(args)


def test_input_document_rejects_blank_core_fields() -> None:
    with pytest.raises(ValidationError, match="doc_id must not be blank"):
        InputDocument(doc_id="   ", text="valid")
    with pytest.raises(ValidationError, match="text must not be blank"):
        InputDocument(doc_id="doc-1", text="\n\t  ")


class _KeepAllScreeningModel:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.call_sizes: list[int] = []
        self.fail_on_call = fail_on_call

    def screen_proposals(
        self,
        state: TaxonomyState,
        proposal_groups: list[ProposalGroup],
        aspect: str,
        *,
        max_tokens: int,
    ) -> ProposalScreeningOutput:
        self.call_sizes.append(len(proposal_groups))
        if self.fail_on_call is not None and len(self.call_sizes) == self.fail_on_call:
            raise RuntimeError("synthetic screening failure")
        return ProposalScreeningOutput(
            decisions=[
                KeepCandidateDecision(
                    action="keep_candidate", proposal_group_id=group.group_id
                )
                for group in proposal_groups
            ]
        )


def _write_discovery_records(path: Path, count: int) -> None:
    records = [
        DiscoveryRecord(
            seq=index,
            doc_id=f"doc-{index}",
            taxonomy_version=1,
            assigned_label_ids=["L001"],
            proposed_labels=[
                ProposedLabelRecord(
                    proposal_id=f"P_{index}",
                    name=f"Candidate {index}",
                    definition=f"Documents centrally about candidate {index}.",
                )
            ],
        )
        for index in range(1, count + 1)
    ]
    append_models_jsonl(path, records)


def test_screening_groups_are_chunked_per_model_call(tmp_path: Path) -> None:
    state = make_state()
    discovery_path = tmp_path / "discovery.jsonl"
    taxonomy_path = tmp_path / "taxonomy.json"
    _write_discovery_records(discovery_path, 5)
    atomic_write_model(taxonomy_path, state)

    model = _KeepAllScreeningModel()
    updated = screen_new_proposals(
        model=model,  # type: ignore[arg-type]
        state=state,
        aspect="topics",
        discovery_output=discovery_path,
        taxonomy_path=taxonomy_path,
        max_total_labels=100,
        max_screening_groups=2,
        maintenance_max_tokens=1024,
    )

    assert model.call_sizes == [2, 2, 1]
    assert updated.last_screened_discovery_seq == 5
    assert len(updated.candidate_proposals) == 5
    persisted = load_model_file(taxonomy_path, TaxonomyState)
    assert persisted.last_screened_discovery_seq == 5
    assert len(persisted.candidate_proposals) == 5


def test_screening_chunk_failure_does_not_advance_persisted_cursor(
    tmp_path: Path,
) -> None:
    state = make_state()
    discovery_path = tmp_path / "discovery.jsonl"
    taxonomy_path = tmp_path / "taxonomy.json"
    _write_discovery_records(discovery_path, 5)
    atomic_write_model(taxonomy_path, state)

    model = _KeepAllScreeningModel(fail_on_call=2)
    with pytest.raises(RuntimeError, match="synthetic screening failure"):
        screen_new_proposals(
            model=model,  # type: ignore[arg-type]
            state=state,
            aspect="topics",
            discovery_output=discovery_path,
            taxonomy_path=taxonomy_path,
            max_total_labels=100,
            max_screening_groups=2,
            maintenance_max_tokens=1024,
        )

    persisted = load_model_file(taxonomy_path, TaxonomyState)
    assert persisted.last_screened_discovery_seq == 0
    assert persisted.candidate_proposals == []
    assert persisted.history == []


class _FinalMaintenanceModel:
    def final_merge_pass(
        self, state: TaxonomyState, aspect: str, *, max_tokens: int
    ) -> FinalMergeOutput:
        return FinalMergeOutput(merges=[])

    def final_revision_pass(
        self,
        state: TaxonomyState,
        review_label_ids: list[str],
        aspect: str,
        *,
        max_tokens: int,
    ) -> FinalRevisionOutput:
        return FinalRevisionOutput(
            revisions=[
                FinalRevisionDecision(
                    label_id="L025",
                    name="Quantum Computing",
                    definition="Documents centrally concerned with quantum computing.",
                )
            ]
        )


def test_final_maintenance_is_not_persisted_before_freeze(tmp_path: Path) -> None:
    state = make_state().model_copy(
        update={
            "schema_version": 2,
            "dynamic_labels": [make_dynamic_label()],
            "next_dynamic_label_num": 26,
        }
    )
    taxonomy_path = tmp_path / "taxonomy.json"
    atomic_write_model(taxonomy_path, state)

    updated = run_final_maintenance(
        model=_FinalMaintenanceModel(),  # type: ignore[arg-type]
        state=state,
        aspect="topics",
        taxonomy_path=taxonomy_path,
        max_total_labels=100,
        min_promotion_support=2,
        max_promotion_groups=16,
        max_revision_labels=16,
        maintenance_max_tokens=1024,
    )

    assert updated.dynamic_labels[0].name == "Quantum Computing"
    persisted = load_model_file(taxonomy_path, TaxonomyState)
    assert persisted.dynamic_labels[0].name == "Quantum computing"
    assert persisted.schema_version == 2


def test_discovery_settings_record_assignment_cap() -> None:
    state = make_state()
    assert state.discovery_settings.max_assigned_labels_per_doc == 5
    changed = state.discovery_settings.model_copy(
        update={"max_assigned_labels_per_doc": 6}
    )
    assert changed != state.discovery_settings


def test_candidate_promotion_rejects_duplicate_active_name() -> None:
    state = make_state().model_copy(
        update={"candidate_proposals": [make_group(support_count=2)]}
    )
    result = CandidatePromotionOutput(
        new_labels=[
            PromotedCandidateLabel(
                candidate_group_ids=["G1"],
                name="Seed 1",
                definition="A duplicate active label name.",
            )
        ]
    )
    with pytest.raises(ValueError, match="already present"):
        validate_candidate_promotion_semantics(
            state,
            state.candidate_proposals,
            result,
            min_promotion_support=2,
            max_new_labels=1,
            max_total_labels=100,
        )


def test_final_revision_rejects_noop() -> None:
    dynamic = make_dynamic_label()
    state = make_state().model_copy(
        update={
            "schema_version": 2,
            "dynamic_labels": [dynamic],
            "next_dynamic_label_num": 26,
        }
    )
    result = FinalRevisionOutput(
        revisions=[
            FinalRevisionDecision(
                label_id=dynamic.id,
                name=dynamic.name,
                definition=dynamic.definition,
            )
        ]
    )
    with pytest.raises(ValueError, match="no-op"):
        validate_final_revision_semantics(state, result)


def test_taxonomy_rejects_invalid_dynamic_version_metadata() -> None:
    state = make_state().model_copy(
        update={
            "schema_version": 2,
            "dynamic_labels": [
                make_dynamic_label().model_copy(
                    update={"created_version": 3, "last_modified_version": 2}
                )
            ],
        }
    )
    with pytest.raises(ValueError, match="created_version"):
        validate_taxonomy_state(state)

    state = make_state().model_copy(
        update={
            "schema_version": 2,
            "dynamic_labels": [
                make_dynamic_label().model_copy(update={"last_modified_version": 3})
            ],
        }
    )
    with pytest.raises(ValueError, match="newer than taxonomy schema_version"):
        validate_taxonomy_state(state)


def test_cli_rejects_negative_seed_for_reproducibility() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        ["--mode", "classify", "--input", "/tmp/input.jsonl", "--seed", "-1"]
    )
    with pytest.raises(ValueError, match="seed"):
        validate_cli_args(args)


def test_cli_rejects_non_boolean_config_style_runtime_values() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["--mode", "classify", "--input", "/tmp/input.jsonl"])
    args.compute_logging = "false"
    with pytest.raises(ValueError, match="compute-logging"):
        validate_cli_args(args)


def test_discovery_output_rejects_duplicate_proposal_evidence() -> None:
    payload = (
        '{"assigned_label_ids":[],"proposed_labels":['
        '{"name":"Quantum Computing","definition":"Documents about quantum computing."},'
        '{"name":" quantum   computing ","definition":"Documents about quantum computing."}'
        "]}"
    )
    with pytest.raises(ValidationError, match="duplicate normalized"):
        LabellingOutputSchema.model_validate_json(payload)


def test_persisted_records_reject_duplicate_assignments_and_proposals() -> None:
    with pytest.raises(ValidationError, match="assigned_label_ids contains duplicates"):
        DiscoveryRecord(
            seq=1,
            doc_id="doc-1",
            taxonomy_version=1,
            assigned_label_ids=["L001", "L001"],
            proposed_labels=[],
        )

    proposal = ProposedLabelRecord(
        proposal_id="P_1",
        name="Quantum computing",
        definition="Documents about quantum computing.",
    )
    with pytest.raises(ValidationError, match="duplicate proposal_id"):
        DiscoveryRecord(
            seq=1,
            doc_id="doc-1",
            taxonomy_version=1,
            assigned_label_ids=[],
            proposed_labels=[proposal, proposal],
        )


class _RevisionBatchModel:
    def __init__(self) -> None:
        self.review_batches: list[list[str]] = []

    def final_merge_pass(
        self, state: TaxonomyState, aspect: str, *, max_tokens: int
    ) -> FinalMergeOutput:
        return FinalMergeOutput(merges=[])

    def final_revision_pass(
        self,
        state: TaxonomyState,
        review_label_ids: list[str],
        aspect: str,
        *,
        max_tokens: int,
    ) -> FinalRevisionOutput:
        self.review_batches.append(list(review_label_ids))
        return FinalRevisionOutput(revisions=[])


def test_final_revision_is_batched(tmp_path: Path) -> None:
    dynamics = [
        make_dynamic_label(label_id=f"L{i:03d}", name=f"Dynamic {i}")
        for i in range(25, 30)
    ]
    state = make_state().model_copy(
        update={
            "schema_version": 2,
            "dynamic_labels": dynamics,
            "next_dynamic_label_num": 30,
        }
    )
    model = _RevisionBatchModel()
    updated = run_final_maintenance(
        model=model,  # type: ignore[arg-type]
        state=state,
        aspect="topics",
        taxonomy_path=tmp_path / "taxonomy.json",
        max_total_labels=100,
        min_promotion_support=2,
        max_promotion_groups=16,
        max_revision_labels=2,
        maintenance_max_tokens=1024,
    )
    assert model.review_batches == [
        ["L025", "L026"],
        ["L027", "L028"],
        ["L029"],
    ]
    assert updated.dynamic_labels == dynamics


def test_invalid_log_level_fails_instead_of_falling_back() -> None:
    with pytest.raises(ValueError, match="Unknown logging level"):
        configure_logging("VERBOSE")


def test_taxonomy_rejects_rolled_back_dynamic_id_counter() -> None:
    state = make_state().model_copy(
        update={
            "schema_version": 2,
            "dynamic_labels": [make_dynamic_label()],
            "next_dynamic_label_num": 25,
        }
    )
    with pytest.raises(ValueError, match="next_dynamic_label_num"):
        validate_taxonomy_state(state)


class _EndToEndDiscoveryModel:
    def __init__(self, **kwargs: object) -> None:
        pass

    def discover_batch(
        self,
        docs: list[InputDocument],
        labels: list[LabelDef],
        **kwargs: object,
    ) -> list[LabellingOutputSchema]:
        outputs: list[LabellingOutputSchema] = []
        for doc in docs:
            if doc.doc_id in {"doc-1", "doc-2"}:
                outputs.append(
                    LabellingOutputSchema(
                        assigned_label_ids=[],
                        proposed_labels=[
                            {
                                "name": "Quantum computing",
                                "definition": "Documents centrally about quantum computing.",
                            }
                        ],
                    )
                )
            elif doc.doc_id == "doc-4":
                outputs.append(
                    LabellingOutputSchema(
                        assigned_label_ids=[],
                        proposed_labels=[
                            {
                                "name": "Robotics",
                                "definition": "Documents centrally about robotics.",
                            }
                        ],
                    )
                )
            else:
                outputs.append(
                    LabellingOutputSchema(
                        assigned_label_ids=["L001"],
                        proposed_labels=[],
                    )
                )
        return outputs

    def screen_proposals(
        self,
        state: TaxonomyState,
        proposal_groups: list[ProposalGroup],
        aspect: str,
        *,
        max_tokens: int,
    ) -> ProposalScreeningOutput:
        return ProposalScreeningOutput(
            decisions=[
                KeepCandidateDecision(
                    action="keep_candidate",
                    proposal_group_id=group.group_id,
                )
                for group in proposal_groups
            ]
        )

    def promote_candidates(
        self,
        state: TaxonomyState,
        candidate_groups: list[ProposalGroup],
        aspect: str,
        *,
        max_new_labels: int,
        max_tokens: int,
    ) -> CandidatePromotionOutput:
        return CandidatePromotionOutput(
            new_labels=[
                PromotedCandidateLabel(
                    candidate_group_ids=[group.group_id],
                    name=group.name,
                    definition=group.definition,
                )
                for group in candidate_groups
            ]
        )

    def final_merge_pass(
        self, state: TaxonomyState, aspect: str, *, max_tokens: int
    ) -> FinalMergeOutput:
        return FinalMergeOutput(merges=[])

    def final_revision_pass(
        self,
        state: TaxonomyState,
        review_label_ids: list[str],
        aspect: str,
        *,
        max_tokens: int,
    ) -> FinalRevisionOutput:
        return FinalRevisionOutput(revisions=[])


class _EndToEndClassificationModel:
    def __init__(self, **kwargs: object) -> None:
        pass

    def classify_batch(
        self,
        docs: list[InputDocument],
        labels: list[LabelDef],
        **kwargs: object,
    ) -> list[FrozenClassificationOutputSchema]:
        return [
            FrozenClassificationOutputSchema(assigned_label_ids=["L001"]) for _ in docs
        ]


def _write_seed_yaml(path: Path) -> None:
    seed_labels = [
        {
            "id": f"L{i:03d}",
            "name": f"Seed {i}",
            "definition": f"Definition {i}",
        }
        for i in range(1, 25)
    ]
    path.write_text(yaml.safe_dump(seed_labels, sort_keys=False), encoding="utf-8")


def _write_input_jsonl(path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index in range(1, 5):
            handle.write(
                json.dumps(
                    {"doc_id": f"doc-{index}", "text": f"Document {index} text."}
                )
                + "\n"
            )


def test_end_to_end_discovery_freeze_and_classification_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input.jsonl"
    seed_path = tmp_path / "seed.yaml"
    taxonomy_path = tmp_path / "taxonomy.json"
    discovery_path = tmp_path / "discovery.jsonl"
    final_path = tmp_path / "final.jsonl"
    _write_input_jsonl(input_path)
    _write_seed_yaml(seed_path)

    parser = build_arg_parser()
    discover_args = parser.parse_args(
        [
            "--mode",
            "discover",
            "--input",
            str(input_path),
            "--seed-labels",
            str(seed_path),
            "--taxonomy",
            str(taxonomy_path),
            "--discovery-output",
            str(discovery_path),
            "--batch-size",
            "2",
            "--screen-every",
            "2",
            "--max-screening-groups",
            "2",
            "--max-promotion-groups",
            "2",
            "--max-revision-labels",
            "2",
            "--min-promotion-support",
            "2",
            "--classification-max-tokens",
            "512",
            "--maintenance-max-tokens",
            "1024",
        ]
    )
    validate_cli_args(discover_args)
    monkeypatch.setattr(discovery_module, "ModelClient", _EndToEndDiscoveryModel)
    discovery_module.run_discovery(discover_args)

    state = load_model_file(taxonomy_path, TaxonomyState)
    assert state.frozen is True
    assert state.format_version == 3
    assert [label.id for label in state.dynamic_labels] == ["L025"]
    assert state.dynamic_labels[0].name == "Quantum computing"
    assert state.dynamic_labels[0].proposal_support_count == 2
    assert len(state.candidate_proposals) == 1
    assert state.candidate_proposals[0].name == "Robotics"
    assert state.candidate_proposals[0].support_count == 1

    records = list(discovery_module.iter_jsonl_models(discovery_path, DiscoveryRecord))
    assert len(records) == 4
    assert [record.seq for record in records] == [1, 2, 3, 4]

    classify_args = parser.parse_args(
        [
            "--mode",
            "classify",
            "--input",
            str(input_path),
            "--taxonomy",
            str(taxonomy_path),
            "--output",
            str(final_path),
            "--batch-size",
            "2",
            "--classification-max-tokens",
            "512",
        ]
    )
    validate_cli_args(classify_args)
    monkeypatch.setattr(
        classification_module, "ModelClient", _EndToEndClassificationModel
    )
    classification_module.run_classification(classify_args)

    final_records = list(
        classification_module.iter_jsonl_models(final_path, FinalClassificationRecord)
    )
    assert len(final_records) == 4
    assert all(
        record.taxonomy_hash == state.frozen_taxonomy_hash for record in final_records
    )

    # Running the same frozen classification again must resume cleanly without
    # duplicating already-persisted records.
    classification_module.run_classification(classify_args)
    final_records_after_resume = list(
        classification_module.iter_jsonl_models(final_path, FinalClassificationRecord)
    )
    assert final_records_after_resume == final_records
