from __future__ import annotations

import argparse
import hashlib
import time
from collections import Counter
from pathlib import Path
from typing import Iterator, Sequence

import vllm  # type: ignore

from label_pipeline_lib.common import (
    LOGGER,
    active_labels,
    append_models_jsonl,
    atomic_write_model,
    batch_iter,
    count_jsonl_records,
    file_fingerprint,
    iter_jsonl_models,
    load_model_file,
    load_seed_labels,
    next_dynamic_id,
    normalize_text,
    resolve_alias,
    seed_labels_hash,
    stream_documents,
    taxonomy_hash,
    validate_taxonomy_state,
    validate_unique_labels,
)
from label_pipeline_lib.compute_logger import detect_num_gpus, start_compute_logger
from label_pipeline_lib.model import ModelClient
from label_pipeline_lib.schemas import (
    CandidatePromotionHistoryEntry,
    CandidatePromotionOutput,
    DiscoveryRecord,
    DiscoverySettings,
    DynamicLabelDef,
    FinalMergeHistoryEntry,
    FinalMergeOutput,
    FinalRevisionHistoryEntry,
    FinalRevisionOutput,
    InputDocument,
    LabelDef,
    ProposalGroup,
    ProposalScreeningHistoryEntry,
    ProposalScreeningOutput,
    ProposedLabelRecord,
    TaxonomyState,
)


def proposal_id(doc_id: str, index: int, name: str, definition: str) -> str:
    payload = f"{doc_id}\0{index}\0{name}\0{definition}".encode("utf-8")
    return "P_" + hashlib.sha256(payload).hexdigest()[:20]


def proposal_group_id(name: str, definition: str) -> str:
    payload = f"{normalize_text(name)}\0{normalize_text(definition)}".encode("utf-8")
    return "G_" + hashlib.sha256(payload).hexdigest()[:20]


def load_result_index(path: Path, model_type: type) -> tuple[set[str], int]:
    ids: set[str] = set()
    expected_seq = 1
    for record in iter_jsonl_models(path, model_type):
        seq = getattr(record, "seq")
        doc_id = getattr(record, "doc_id")
        if seq != expected_seq:
            raise ValueError(
                f"Non-contiguous sequence in {path}: expected {expected_seq}, got {seq}"
            )
        expected_seq += 1
        if doc_id in ids:
            raise ValueError(f"Duplicate doc_id {doc_id!r} in {path}")
        ids.add(doc_id)
    return ids, expected_seq - 1


def scan_discovery_results(
    path: Path,
    state: TaxonomyState,
    *,
    after_seq: int,
) -> tuple[int, Counter[str], list[ProposalGroup]]:
    """Scan persisted discovery results and group proposals after ``after_seq``.

    Usage counts are recomputed from the complete persisted discovery file so they
    remain deterministic after resume. Proposal evidence is returned only for the
    yet-unscreened suffix of the file.
    """

    usage: Counter[str] = Counter()
    grouped: dict[str, ProposalGroup] = {}
    latest_seq = 0
    expected_seq = 1
    seen_docs: set[str] = set()
    seen_proposal_ids: set[str] = set()

    active_ids = {label.id for label in active_labels(state)}
    historically_known_ids = active_ids | set(state.alias_map)
    active_dynamic_ids = {label.id for label in state.dynamic_labels}

    for record in iter_jsonl_models(path, DiscoveryRecord):
        if record.seq != expected_seq:
            raise ValueError(
                f"Non-contiguous sequence in {path}: expected {expected_seq}, got {record.seq}"
            )
        expected_seq += 1
        latest_seq = record.seq

        if record.doc_id in seen_docs:
            raise ValueError(f"Duplicate doc_id {record.doc_id!r} in {path}")
        seen_docs.add(record.doc_id)

        if record.taxonomy_version > state.schema_version:
            raise ValueError(
                f"Discovery record {record.doc_id} uses taxonomy version "
                f"{record.taxonomy_version}, newer than state version {state.schema_version}"
            )

        for label_id in record.assigned_label_ids:
            if label_id not in historically_known_ids:
                raise ValueError(
                    f"Discovery record {record.doc_id} references label {label_id!r} "
                    "which is absent from the active taxonomy and alias history"
                )
            resolved = resolve_alias(label_id, state.alias_map)
            if resolved in active_dynamic_ids:
                usage[resolved] += 1

        for proposed in record.proposed_labels:
            if proposed.proposal_id in seen_proposal_ids:
                raise ValueError(
                    f"Duplicate proposal_id {proposed.proposal_id!r} in {path}"
                )
            seen_proposal_ids.add(proposed.proposal_id)

            if record.seq <= after_seq:
                continue

            group_id = proposal_group_id(proposed.name, proposed.definition)
            existing = grouped.get(group_id)
            if existing is None:
                grouped[group_id] = ProposalGroup(
                    group_id=group_id,
                    name=proposed.name,
                    definition=proposed.definition,
                    support_count=1,
                )
                continue

            if normalize_text(existing.name) != normalize_text(
                proposed.name
            ) or normalize_text(existing.definition) != normalize_text(
                proposed.definition
            ):
                raise RuntimeError("Proposal-group hash collision")

            grouped[group_id] = existing.model_copy(
                update={"support_count": existing.support_count + 1}
            )

    return latest_seq, usage, list(grouped.values())


def update_usage_counts(state: TaxonomyState, usage: Counter[str]) -> TaxonomyState:
    updated = [
        label.model_copy(update={"usage_count": usage[label.id]})
        for label in state.dynamic_labels
    ]
    return state.model_copy(update={"dynamic_labels": updated})


def accumulate_existing_candidate_support(
    state: TaxonomyState,
    new_groups: Sequence[ProposalGroup],
) -> tuple[TaxonomyState, list[ProposalGroup]]:
    """Accumulate exact repeats of already-kept candidates.

    A group that is already in ``candidate_proposals`` has already passed semantic
    screening, so a later exact normalized recurrence only increments support. New
    group IDs are returned for screening.
    """

    candidate_by_id: dict[str, ProposalGroup] = {}
    for candidate in state.candidate_proposals:
        if candidate.group_id in candidate_by_id:
            raise ValueError(
                f"Duplicate candidate proposal group {candidate.group_id!r} in taxonomy state"
            )
        candidate_by_id[candidate.group_id] = candidate

    groups_to_screen: list[ProposalGroup] = []

    for group in new_groups:
        existing = candidate_by_id.get(group.group_id)
        if existing is None:
            groups_to_screen.append(group)
            continue

        if normalize_text(existing.name) != normalize_text(
            group.name
        ) or normalize_text(existing.definition) != normalize_text(group.definition):
            raise RuntimeError("Proposal-group hash collision")

        candidate_by_id[group.group_id] = existing.model_copy(
            update={"support_count": existing.support_count + group.support_count}
        )

    return (
        state.model_copy(
            update={"candidate_proposals": list(candidate_by_id.values())}
        ),
        groups_to_screen,
    )


def validate_proposal_screening_semantics(
    state: TaxonomyState,
    proposal_groups: Sequence[ProposalGroup],
    result: ProposalScreeningOutput,
) -> None:
    expected_ids = {group.group_id for group in proposal_groups}
    returned_ids = [decision.proposal_group_id for decision in result.decisions]

    if len(returned_ids) != len(set(returned_ids)):
        raise ValueError("Proposal screener returned a proposal group more than once")

    returned_id_set = set(returned_ids)
    if returned_id_set != expected_ids:
        raise ValueError(
            "Proposal screening coverage mismatch; "
            f"missing={sorted(expected_ids - returned_id_set)}, "
            f"unknown={sorted(returned_id_set - expected_ids)}"
        )

    active_ids = {label.id for label in active_labels(state)}
    for decision in result.decisions:
        if decision.action == "map_existing":
            if decision.target_label_id not in active_ids:
                raise ValueError(
                    "Proposal screener mapped to unknown active label "
                    f"{decision.target_label_id!r}"
                )
        elif decision.action not in {"keep_candidate", "discard_candidate"}:
            raise ValueError(f"Unknown screening action {decision.action!r}")


def apply_proposal_screening(
    state: TaxonomyState,
    proposal_groups: Sequence[ProposalGroup],
    result: ProposalScreeningOutput,
    *,
    discovery_seq_end: int,
    max_total_labels: int,
) -> TaxonomyState:
    """Apply screening without changing the active taxonomy version."""

    validate_proposal_screening_semantics(state, proposal_groups, result)
    group_by_id = {group.group_id: group for group in proposal_groups}
    candidate_by_id = {group.group_id: group for group in state.candidate_proposals}

    if len(candidate_by_id) != len(state.candidate_proposals):
        raise ValueError("Taxonomy state contains duplicate candidate proposal groups")

    for decision in result.decisions:
        group_id = decision.proposal_group_id

        if decision.action == "keep_candidate":
            incoming = group_by_id[group_id]
            existing = candidate_by_id.get(group_id)
            if existing is not None:
                # Re-screening an existing candidate is allowed, but the evidence
                # object itself must be unchanged by the model boundary.
                if existing != incoming:
                    raise ValueError(
                        f"Candidate {group_id!r} changed unexpectedly during re-screening"
                    )
            else:
                candidate_by_id[group_id] = incoming

        elif decision.action in {"map_existing", "discard_candidate"}:
            # This also handles a re-screened existing candidate that has become
            # covered by a newly created active label.
            candidate_by_id.pop(group_id, None)

        else:  # pragma: no cover - discriminated Pydantic schema prevents this.
            raise ValueError(f"Unknown screening action {decision.action!r}")

    now = int(time.time())
    history_entry = ProposalScreeningHistoryEntry(
        timestamp_unix=now,
        discovery_seq_end=discovery_seq_end,
        taxonomy_version=state.schema_version,
        proposal_groups=list(proposal_groups),
        screening=result,
    )

    candidate = state.model_copy(
        update={
            "candidate_proposals": list(candidate_by_id.values()),
            "last_screened_discovery_seq": discovery_seq_end,
            "history": [*state.history, history_entry],
            "updated_at_unix": now,
        }
    )
    validate_taxonomy_state(candidate, max_total_labels=max_total_labels)
    return candidate


def screen_new_proposals(
    *,
    model: ModelClient,
    state: TaxonomyState,
    aspect: str,
    discovery_output: Path,
    taxonomy_path: Path,
    max_total_labels: int,
    max_screening_groups: int,
    maintenance_max_tokens: int,
) -> TaxonomyState:
    latest_seq, usage, new_groups = scan_discovery_results(
        discovery_output,
        state,
        after_seq=state.last_screened_discovery_seq,
    )
    state = update_usage_counts(state, usage)
    state, groups_to_screen = accumulate_existing_candidate_support(state, new_groups)

    if not groups_to_screen:
        state = state.model_copy(
            update={
                "last_screened_discovery_seq": latest_seq,
                "updated_at_unix": int(time.time()),
            }
        )
        validate_taxonomy_state(state, max_total_labels=max_total_labels)
        atomic_write_model(taxonomy_path, state)
        return state

    # max_screening_groups remains a per-prompt bound. The chunks are
    # semantically independent because they all screen against the same taxonomy
    # snapshot, so submit every chunk to vLLM in one scheduler batch. State is
    # still applied sequentially only after the entire model batch succeeds,
    # preserving the transactional screening cursor.
    group_batches = [
        list(group_batch)
        for group_batch in batch_iter(groups_to_screen, max_screening_groups)
    ]

    LOGGER.info(
        "Maintenance: Screening %d proposal groups in %d concurrent chunks "
        "against taxonomy v%d",
        len(groups_to_screen),
        len(group_batches),
        state.schema_version,
    )

    results = model.screen_proposal_batches(
        state,
        group_batches,
        aspect,
        max_tokens=maintenance_max_tokens,
    )
    if len(results) != len(group_batches):
        raise RuntimeError(
            f"Expected {len(group_batches)} screening results, got {len(results)}"
        )

    for group_batch, result in zip(group_batches, results, strict=True):
        state = apply_proposal_screening(
            state,
            group_batch,
            result,
            discovery_seq_end=latest_seq,
            max_total_labels=max_total_labels,
        )

    atomic_write_model(taxonomy_path, state)
    return state


def validate_candidate_promotion_semantics(
    state: TaxonomyState,
    candidate_groups: Sequence[ProposalGroup],
    result: CandidatePromotionOutput,
    *,
    min_promotion_support: int,
    max_new_labels: int,
    max_total_labels: int,
) -> None:
    expected_ids = {group.group_id for group in candidate_groups}
    returned_ids = [
        group_id
        for new_label in result.new_labels
        for group_id in new_label.candidate_group_ids
    ]

    if len(returned_ids) != len(set(returned_ids)):
        raise ValueError("Candidate promoter used a candidate group more than once")

    returned_id_set = set(returned_ids)
    if returned_id_set != expected_ids:
        raise ValueError(
            "Candidate promotion coverage mismatch; "
            f"missing={sorted(expected_ids - returned_id_set)}, "
            f"unknown={sorted(returned_id_set - expected_ids)}"
        )

    if len(result.new_labels) > max_new_labels:
        raise ValueError(
            f"Candidate promoter created {len(result.new_labels)} labels, exceeding "
            f"the task limit {max_new_labels}"
        )

    new_names = [normalize_text(label.name) for label in result.new_labels]
    if len(new_names) != len(set(new_names)):
        raise ValueError("Candidate promoter produced duplicate new-label names")

    active_names = {normalize_text(label.name) for label in active_labels(state)}
    overlapping_names = sorted(set(new_names) & active_names)
    if overlapping_names:
        raise ValueError(
            "Candidate promoter produced a label name already present in the active "
            f"taxonomy: {overlapping_names}"
        )

    state_candidates = {group.group_id: group for group in state.candidate_proposals}
    if len(state_candidates) != len(state.candidate_proposals):
        raise ValueError("Taxonomy state contains duplicate candidate proposal groups")

    for group in candidate_groups:
        persisted = state_candidates.get(group.group_id)
        if persisted is None:
            raise ValueError(
                f"Promotion candidate {group.group_id!r} is absent from taxonomy state"
            )
        if persisted != group:
            raise ValueError(
                f"Promotion candidate {group.group_id!r} does not match persisted evidence"
            )
        if group.support_count < min_promotion_support:
            raise ValueError(
                f"Promotion candidate {group.group_id!r} has support "
                f"{group.support_count}, below minimum {min_promotion_support}"
            )

    resulting_count = len(active_labels(state)) + len(result.new_labels)
    if resulting_count > max_total_labels:
        raise ValueError(
            f"Candidate promotion would create {resulting_count} active labels, "
            f"exceeding maximum {max_total_labels}"
        )


def apply_candidate_promotion(
    state: TaxonomyState,
    candidate_groups: Sequence[ProposalGroup],
    result: CandidatePromotionOutput,
    *,
    discovery_seq_end: int,
    min_promotion_support: int,
    max_new_labels: int,
    max_total_labels: int,
) -> TaxonomyState:
    validate_candidate_promotion_semantics(
        state,
        candidate_groups,
        result,
        min_promotion_support=min_promotion_support,
        max_new_labels=max_new_labels,
        max_total_labels=max_total_labels,
    )

    before_version = state.schema_version
    after_version = before_version + 1
    group_by_id = {group.group_id: group for group in candidate_groups}
    promoted_group_ids = {
        group_id
        for new_label in result.new_labels
        for group_id in new_label.candidate_group_ids
    }

    dynamic_by_id = {label.id: label for label in state.dynamic_labels}
    state_for_id_allocation = state.model_copy(
        update={"dynamic_labels": list(dynamic_by_id.values())}
    )
    created_labels: list[DynamicLabelDef] = []

    for promoted in result.new_labels:
        support = sum(
            group_by_id[group_id].support_count
            for group_id in promoted.candidate_group_ids
        )
        new_id = next_dynamic_id(state_for_id_allocation)
        created = DynamicLabelDef(
            id=new_id,
            name=promoted.name,
            definition=promoted.definition,
            created_version=after_version,
            last_modified_version=after_version,
            usage_count=0,
            proposal_support_count=support,
        )
        dynamic_by_id[new_id] = created
        state_for_id_allocation.dynamic_labels.append(created)
        created_labels.append(created)

    remaining_candidates = [
        group
        for group in state.candidate_proposals
        if group.group_id not in promoted_group_ids
    ]

    now = int(time.time())
    candidate = state.model_copy(
        update={
            "schema_version": after_version,
            "dynamic_labels": list(dynamic_by_id.values()),
            "candidate_proposals": remaining_candidates,
            "next_dynamic_label_num": state_for_id_allocation.next_dynamic_label_num,
            "updated_at_unix": now,
        }
    )

    validate_unique_labels(active_labels(candidate), context="promoted taxonomy")
    if len(active_labels(candidate)) > max_total_labels:
        raise ValueError(
            f"Promotion produced {len(active_labels(candidate))} active labels, "
            f"exceeding cap {max_total_labels}"
        )

    history_entry = CandidatePromotionHistoryEntry(
        timestamp_unix=now,
        discovery_seq_end=discovery_seq_end,
        version_before=before_version,
        version_after=after_version,
        candidate_groups=list(candidate_groups),
        promotion=result,
        created_labels=created_labels,
    )
    candidate = candidate.model_copy(
        update={"history": [*candidate.history, history_entry]}
    )
    validate_taxonomy_state(candidate, max_total_labels=max_total_labels)
    return candidate


def promote_eligible_candidates(
    *,
    model: ModelClient,
    state: TaxonomyState,
    aspect: str,
    taxonomy_path: Path,
    min_promotion_support: int,
    max_promotion_groups: int,
    max_total_labels: int,
    maintenance_max_tokens: int,
    persist: bool = True,
) -> TaxonomyState:
    """Promote supported candidates in bounded batches.

    Before each promotion batch, the selected candidates are screened again against
    the *current* taxonomy. This matters because a candidate kept at taxonomy v2 may
    become adequately covered by a label created at v3.

    The selected batch never contains more candidate groups than available label
    slots. Therefore the promoter can always keep every candidate distinct if needed;
    it is never forced to merge unrelated concepts merely to satisfy the taxonomy cap.
    """

    while True:
        available_slots = max_total_labels - len(active_labels(state))
        if available_slots <= 0:
            return state

        eligible = sorted(
            (
                group
                for group in state.candidate_proposals
                if group.support_count >= min_promotion_support
            ),
            key=lambda group: (-group.support_count, group.group_id),
        )
        if not eligible:
            return state

        batch_size = min(max_promotion_groups, available_slots, len(eligible))
        selected = eligible[:batch_size]

        # Re-screen against the current taxonomy immediately before promotion.
        screening = model.screen_proposals(
            state,
            selected,
            aspect=aspect,
            max_tokens=maintenance_max_tokens,
        )
        state = apply_proposal_screening(
            state,
            selected,
            screening,
            discovery_seq_end=state.last_screened_discovery_seq,
            max_total_labels=max_total_labels,
        )
        if persist:
            atomic_write_model(taxonomy_path, state)

        kept_ids = {
            decision.proposal_group_id
            for decision in screening.decisions
            if decision.action == "keep_candidate"
        }
        if not kept_ids:
            # All selected candidates were mapped/discarded, which is still
            # deterministic progress because they were removed from the pool.
            continue

        candidate_by_id = {group.group_id: group for group in state.candidate_proposals}
        to_promote = [
            candidate_by_id[group.group_id]
            for group in selected
            if group.group_id in kept_ids
        ]

        max_new_labels = len(to_promote)
        promotion = model.promote_candidates(
            state,
            to_promote,
            aspect=aspect,
            max_new_labels=max_new_labels,
            max_tokens=maintenance_max_tokens,
        )
        state = apply_candidate_promotion(
            state,
            to_promote,
            promotion,
            discovery_seq_end=state.last_screened_discovery_seq,
            min_promotion_support=min_promotion_support,
            max_new_labels=max_new_labels,
            max_total_labels=max_total_labels,
        )
        if persist:
            atomic_write_model(taxonomy_path, state)


def run_periodic_maintenance(
    *,
    model: ModelClient,
    state: TaxonomyState,
    aspect: str,
    discovery_output: Path,
    taxonomy_path: Path,
    max_total_labels: int,
    min_promotion_support: int,
    max_screening_groups: int,
    max_promotion_groups: int,
    maintenance_max_tokens: int,
) -> TaxonomyState:
    state = screen_new_proposals(
        model=model,
        state=state,
        aspect=aspect,
        discovery_output=discovery_output,
        taxonomy_path=taxonomy_path,
        max_total_labels=max_total_labels,
        max_screening_groups=max_screening_groups,
        maintenance_max_tokens=maintenance_max_tokens,
    )
    state = promote_eligible_candidates(
        model=model,
        state=state,
        aspect=aspect,
        taxonomy_path=taxonomy_path,
        min_promotion_support=min_promotion_support,
        max_promotion_groups=max_promotion_groups,
        max_total_labels=max_total_labels,
        maintenance_max_tokens=maintenance_max_tokens,
    )
    return state


def validate_final_merge_semantics(
    state: TaxonomyState,
    result: FinalMergeOutput,
) -> None:
    active_ids = {label.id for label in active_labels(state)}
    dynamic_ids = {label.id for label in state.dynamic_labels}
    sources = [merge.source_label_id for merge in result.merges]

    if len(sources) != len(set(sources)):
        raise ValueError("Final merge pass returned a dynamic source more than once")

    source_set = set(sources)
    for merge in result.merges:
        if merge.source_label_id not in dynamic_ids:
            raise ValueError(
                f"Final merge source {merge.source_label_id!r} is not a dynamic label"
            )
        if merge.target_label_id not in active_ids:
            raise ValueError(
                f"Final merge target {merge.target_label_id!r} is not active"
            )
        if merge.source_label_id == merge.target_label_id:
            raise ValueError("A label cannot be merged into itself")
        if merge.target_label_id in source_set:
            raise ValueError(
                "Final merge chains are forbidden; target "
                f"{merge.target_label_id!r} is also a merge source"
            )


def apply_final_merge(
    state: TaxonomyState,
    result: FinalMergeOutput,
    *,
    max_total_labels: int,
) -> TaxonomyState:
    validate_final_merge_semantics(state, result)

    before_version = state.schema_version
    after_version = before_version + 1 if result.merges else before_version
    dynamic_by_id = {label.id: label for label in state.dynamic_labels}
    alias_map = dict(state.alias_map)

    for merge in result.merges:
        source = dynamic_by_id.pop(merge.source_label_id)
        target_id = resolve_alias(merge.target_label_id, alias_map)

        if target_id in dynamic_by_id:
            target = dynamic_by_id[target_id]
            dynamic_by_id[target_id] = target.model_copy(
                update={
                    "usage_count": target.usage_count + source.usage_count,
                    "proposal_support_count": (
                        target.proposal_support_count + source.proposal_support_count
                    ),
                    "last_modified_version": after_version,
                }
            )

        alias_map[source.id] = target_id
        for old_source, old_target in list(alias_map.items()):
            if old_target == source.id:
                alias_map[old_source] = target_id

    now = int(time.time())
    candidate = state.model_copy(
        update={
            "schema_version": after_version,
            "dynamic_labels": list(dynamic_by_id.values()),
            "alias_map": alias_map,
            "updated_at_unix": now,
        }
    )
    validate_unique_labels(active_labels(candidate), context="post-merge taxonomy")

    history_entry = FinalMergeHistoryEntry(
        timestamp_unix=now,
        version_before=before_version,
        version_after=after_version,
        result=result,
    )
    candidate = candidate.model_copy(
        update={"history": [*candidate.history, history_entry]}
    )
    validate_taxonomy_state(candidate, max_total_labels=max_total_labels)
    return candidate


def validate_final_revision_semantics(
    state: TaxonomyState,
    result: FinalRevisionOutput,
) -> None:
    dynamic_ids = {label.id for label in state.dynamic_labels}
    returned_ids = [revision.label_id for revision in result.revisions]

    if len(returned_ids) != len(set(returned_ids)):
        raise ValueError("Final revision pass returned a dynamic label more than once")

    unknown = set(returned_ids) - dynamic_ids
    if unknown:
        raise ValueError(
            f"Final revision pass referenced non-dynamic labels {sorted(unknown)}"
        )

    revisions_by_id = {revision.label_id: revision for revision in result.revisions}
    for label in state.dynamic_labels:
        revision = revisions_by_id.get(label.id)
        if revision is not None and (
            revision.name == label.name and revision.definition == label.definition
        ):
            raise ValueError(
                f"Final revision for {label.id!r} is a no-op; unchanged labels must "
                "be omitted from revisions"
            )

    resulting_names = [normalize_text(label.name) for label in state.seed_labels]
    resulting_names.extend(
        (
            normalize_text(revisions_by_id[label.id].name)
            if label.id in revisions_by_id
            else normalize_text(label.name)
        )
        for label in state.dynamic_labels
    )
    if len(resulting_names) != len(set(resulting_names)):
        raise ValueError("Final revisions would create duplicate active label names")


def apply_final_revision(
    state: TaxonomyState,
    result: FinalRevisionOutput,
    *,
    max_total_labels: int,
) -> TaxonomyState:
    validate_final_revision_semantics(state, result)

    before_version = state.schema_version
    after_version = before_version + 1 if result.revisions else before_version
    dynamic_by_id = {label.id: label for label in state.dynamic_labels}

    for revision in result.revisions:
        label = dynamic_by_id[revision.label_id]
        dynamic_by_id[label.id] = label.model_copy(
            update={
                "name": revision.name,
                "definition": revision.definition,
                "last_modified_version": after_version,
            }
        )

    now = int(time.time())
    candidate = state.model_copy(
        update={
            "schema_version": after_version,
            "dynamic_labels": list(dynamic_by_id.values()),
            "updated_at_unix": now,
        }
    )
    validate_unique_labels(active_labels(candidate), context="post-revision taxonomy")

    history_entry = FinalRevisionHistoryEntry(
        timestamp_unix=now,
        version_before=before_version,
        version_after=after_version,
        result=result,
    )
    candidate = candidate.model_copy(
        update={"history": [*candidate.history, history_entry]}
    )
    validate_taxonomy_state(candidate, max_total_labels=max_total_labels)
    return candidate


def run_final_maintenance(
    *,
    model: ModelClient,
    state: TaxonomyState,
    aspect: str,
    taxonomy_path: Path,
    max_total_labels: int,
    min_promotion_support: int,
    max_promotion_groups: int,
    max_revision_labels: int,
    maintenance_max_tokens: int,
) -> TaxonomyState:
    """Finish taxonomy construction with isolated merge/revision tasks.

    A merge can free taxonomy slots. Therefore final maintenance alternates a
    merge pass with deterministic promotion of any still-supported candidates.
    If promotion creates labels, another merge pass is run so newly created
    labels also receive final redundancy review. The loop stops as soon as a
    merge pass is followed by no promotion. Revision then runs exactly once on
    the surviving dynamic labels.

    This function intentionally does not persist intermediate final-maintenance
    states. The caller atomically writes the taxonomy only after setting the
    frozen metadata, so a crash cannot leave a half-finalized state that would
    receive a second final pass on resume.
    """

    while True:
        merge_result = model.final_merge_pass(
            state,
            aspect=aspect,
            max_tokens=maintenance_max_tokens,
        )
        state = apply_final_merge(
            state,
            merge_result,
            max_total_labels=max_total_labels,
        )

        version_before_promotion = state.schema_version
        state = promote_eligible_candidates(
            model=model,
            state=state,
            aspect=aspect,
            taxonomy_path=taxonomy_path,
            min_promotion_support=min_promotion_support,
            max_promotion_groups=max_promotion_groups,
            max_total_labels=max_total_labels,
            maintenance_max_tokens=maintenance_max_tokens,
            persist=False,
        )

        if state.schema_version == version_before_promotion:
            break

    if max_revision_labels < 1:
        raise ValueError("max_revision_labels must be >= 1")

    # Revision output includes complete names and definitions, so bound the number
    # of labels reviewed per call. Every call still sees the complete surviving
    # taxonomy for global name/style consistency.
    revision_ids = [label.id for label in state.dynamic_labels]
    for review_batch in batch_iter(revision_ids, max_revision_labels):
        revision_result = model.final_revision_pass(
            state,
            review_batch,
            aspect=aspect,
            max_tokens=maintenance_max_tokens,
        )
        state = apply_final_revision(
            state,
            revision_result,
            max_total_labels=max_total_labels,
        )
    return state


def build_discovery_settings(
    args: argparse.Namespace,
    input_path: Path,
) -> DiscoverySettings:
    resolved, size, mtime_ns = file_fingerprint(input_path)
    return DiscoverySettings(
        input_path=resolved,
        input_size_bytes=size,
        input_mtime_ns=mtime_ns,
        model_name=args.model,
        vllm_version=vllm.__version__,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        max_document_tokens=args.max_document_tokens,
        classification_max_tokens=args.classification_max_tokens,
        max_assigned_labels_per_doc=args.max_assigned_labels_per_doc,
        maintenance_max_tokens=args.maintenance_max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        seed=args.seed,
        thinking_mode=args.thinking_mode,
        screen_every=args.screen_every,
        max_screening_groups=args.max_screening_groups,
        max_promotion_groups=args.max_promotion_groups,
        max_revision_labels=args.max_revision_labels,
        min_promotion_support=args.min_promotion_support,
        max_proposals_per_doc=args.max_proposals_per_doc,
        max_total_labels=args.max_total_labels,
        expected_seed_label_count=args.expected_seed_label_count,
        creativity=args.creativity,
    )


def initialize_or_load_taxonomy(
    *,
    taxonomy_path: Path,
    seed_labels: Sequence[LabelDef],
    aspect: str,
    settings: DiscoverySettings,
) -> TaxonomyState:
    seed_hash = seed_labels_hash(seed_labels)

    if taxonomy_path.exists():
        state = load_model_file(taxonomy_path, TaxonomyState)
        validate_taxonomy_state(
            state,
            max_total_labels=settings.max_total_labels,
            expected_seed_count=settings.expected_seed_label_count,
        )
        if state.frozen:
            raise ValueError(
                f"Taxonomy {taxonomy_path} is already frozen; use --mode classify "
                "or start a fresh discovery run"
            )
        if state.aspect != aspect:
            raise ValueError(
                f"Resume aspect mismatch: taxonomy={state.aspect!r}, requested={aspect!r}"
            )
        if state.seed_labels_sha256 != seed_hash:
            raise ValueError("Seed labels differ from the taxonomy being resumed")
        if state.discovery_settings != settings:
            raise ValueError(
                "Discovery settings differ from the existing taxonomy state. "
                "For research reproducibility, resume with identical settings or "
                "use --reset-discovery."
            )
        return state

    now = int(time.time())
    state = TaxonomyState(
        format_version=3,
        aspect=aspect,
        schema_version=1,
        frozen=False,
        seed_labels_sha256=seed_hash,
        seed_labels=list(seed_labels),
        dynamic_labels=[],
        alias_map={},
        candidate_proposals=[],
        next_dynamic_label_num=25,
        last_screened_discovery_seq=0,
        discovery_settings=settings,
        history=[],
        created_at_unix=now,
        updated_at_unix=now,
    )
    validate_taxonomy_state(
        state,
        max_total_labels=settings.max_total_labels,
        expected_seed_count=settings.expected_seed_label_count,
    )
    atomic_write_model(taxonomy_path, state)
    return state


def reset_discovery_files(taxonomy_path: Path, discovery_output: Path) -> None:
    for path in (taxonomy_path, discovery_output):
        if path.exists():
            path.unlink()


def run_discovery(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    seed_path = Path(args.seed_labels)
    taxonomy_path = Path(args.taxonomy)
    discovery_output = Path(args.discovery_output)

    if args.reset_discovery:
        reset_discovery_files(taxonomy_path, discovery_output)

    if discovery_output.exists() and not taxonomy_path.exists():
        raise ValueError(
            f"Discovery output {discovery_output} exists but taxonomy state "
            f"{taxonomy_path} is missing. Use --reset-discovery or restore the state file."
        )

    seed_labels = load_seed_labels(seed_path, args.expected_seed_label_count)
    settings = build_discovery_settings(args, input_path)
    state = initialize_or_load_taxonomy(
        taxonomy_path=taxonomy_path,
        seed_labels=seed_labels,
        aspect=args.aspect,
        settings=settings,
    )

    if not discovery_output.exists() and (
        state.last_screened_discovery_seq != 0
        or state.schema_version != 1
        or state.dynamic_labels
        or state.candidate_proposals
        or state.history
    ):
        raise ValueError(
            "Taxonomy state contains discovery progress but discovery output is missing"
        )

    completed_ids, last_seq = load_result_index(discovery_output, DiscoveryRecord)
    if last_seq < state.last_screened_discovery_seq:
        raise ValueError(
            "Taxonomy screening cursor is ahead of persisted discovery output"
        )

    LOGGER.info(
        "Discovery resume state: %d completed documents, taxonomy version %d, "
        "%d kept candidates",
        len(completed_ids),
        state.schema_version,
        len(state.candidate_proposals),
    )

    model = ModelClient(
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        seed=args.seed,
        thinking_mode=args.thinking_mode,
    )

    seen_input_ids: set[str] = set()
    next_seq = last_seq + 1
    processed_this_run = 0
    total_input = count_jsonl_records(input_path)
    if len(completed_ids) > total_input:
        raise ValueError(
            f"Discovery output contains {len(completed_ids)} records but input "
            f"contains only {total_input} documents"
        )
    total_pending = total_input - len(completed_ids)

    stop_event = None
    if getattr(args, "compute_logging", False) and total_pending > 0:

        def _get_progress() -> tuple[int, int]:
            return processed_this_run, total_pending

        stop_event = start_compute_logger(
            interval=getattr(args, "compute_logging_interval", 300.0),
            num_gpus=detect_num_gpus(getattr(args, "tensor_parallel_size", 1)),
            get_progress=_get_progress,
        )

    try:

        def pending_docs() -> Iterator[InputDocument]:
            for doc in stream_documents(input_path):
                if doc.doc_id in seen_input_ids:
                    raise ValueError(f"Duplicate doc_id {doc.doc_id!r} in input corpus")
                seen_input_ids.add(doc.doc_id)
                if doc.doc_id in completed_ids:
                    continue
                yield doc

        for batch in batch_iter(pending_docs(), args.batch_size):
            labels_snapshot = active_labels(state)
            outputs = model.discover_batch(
                batch,
                labels_snapshot,
                aspect=state.aspect,
                creativity=args.creativity,
                max_document_tokens=args.max_document_tokens,
                output_max_tokens=args.classification_max_tokens,
                max_assigned_labels_per_doc=args.max_assigned_labels_per_doc,
                max_proposals_per_doc=args.max_proposals_per_doc,
            )
            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"Expected {len(batch)} validated outputs, got {len(outputs)}"
                )

            records: list[DiscoveryRecord] = []
            for doc, output in zip(batch, outputs, strict=True):
                proposals = [
                    ProposedLabelRecord(
                        proposal_id=proposal_id(
                            doc.doc_id,
                            index,
                            proposed.name,
                            proposed.definition,
                        ),
                        name=proposed.name,
                        definition=proposed.definition,
                    )
                    for index, proposed in enumerate(output.proposed_labels)
                ]
                records.append(
                    DiscoveryRecord(
                        seq=next_seq,
                        doc_id=doc.doc_id,
                        taxonomy_version=state.schema_version,
                        assigned_label_ids=output.assigned_label_ids,
                        proposed_labels=proposals,
                    )
                )
                next_seq += 1

            append_models_jsonl(discovery_output, records)
            for record in records:
                completed_ids.add(record.doc_id)
            processed_this_run += len(records)
            last_seq = records[-1].seq

            LOGGER.info(
                "Discovery: processed %d documents this run; latest seq=%d; taxonomy v%d",
                processed_this_run,
                last_seq,
                state.schema_version,
            )

            if last_seq - state.last_screened_discovery_seq >= args.screen_every:
                state = run_periodic_maintenance(
                    model=model,
                    state=state,
                    aspect=state.aspect,
                    discovery_output=discovery_output,
                    taxonomy_path=taxonomy_path,
                    max_total_labels=args.max_total_labels,
                    min_promotion_support=args.min_promotion_support,
                    max_screening_groups=args.max_screening_groups,
                    max_promotion_groups=args.max_promotion_groups,
                    maintenance_max_tokens=args.maintenance_max_tokens,
                )
                LOGGER.info(
                    "Maintenance complete: taxonomy v%d, %d active labels, %d kept candidates",
                    state.schema_version,
                    len(active_labels(state)),
                    len(state.candidate_proposals),
                )

    finally:
        if stop_event is not None:
            stop_event.set()

    if total_input == 0:
        raise ValueError("Input corpus contains no documents")

    missing_completed = completed_ids - seen_input_ids
    if missing_completed:
        raise ValueError(
            "Discovery output contains document IDs absent from the input corpus: "
            f"{sorted(missing_completed)[:10]}"
        )

    # Screen the final suffix and promote every supported candidate for which the
    # taxonomy cap leaves room.
    state = run_periodic_maintenance(
        model=model,
        state=state,
        aspect=state.aspect,
        discovery_output=discovery_output,
        taxonomy_path=taxonomy_path,
        max_total_labels=args.max_total_labels,
        min_promotion_support=args.min_promotion_support,
        max_screening_groups=args.max_screening_groups,
        max_promotion_groups=args.max_promotion_groups,
        maintenance_max_tokens=args.maintenance_max_tokens,
    )

    # Existing dynamic-label cleanup is deliberately isolated from proposal work.
    state = run_final_maintenance(
        model=model,
        state=state,
        aspect=state.aspect,
        taxonomy_path=taxonomy_path,
        max_total_labels=args.max_total_labels,
        min_promotion_support=args.min_promotion_support,
        max_promotion_groups=args.max_promotion_groups,
        max_revision_labels=args.max_revision_labels,
        maintenance_max_tokens=args.maintenance_max_tokens,
    )

    eligible_unpromoted = [
        group
        for group in state.candidate_proposals
        if group.support_count >= args.min_promotion_support
    ]
    if eligible_unpromoted:
        if len(active_labels(state)) < args.max_total_labels:
            raise RuntimeError(
                "Supported candidates remain after final promotion even though taxonomy "
                "capacity is available; this indicates an orchestration bug"
            )
        LOGGER.warning(
            "%d supported candidate groups remain unpromoted because the taxonomy "
            "has reached --max-total-labels=%d",
            len(eligible_unpromoted),
            args.max_total_labels,
        )

    if state.candidate_proposals:
        LOGGER.info(
            "Freezing taxonomy with %d unpromoted candidate groups retained as "
            "discovery audit evidence",
            len(state.candidate_proposals),
        )

    frozen_hash = taxonomy_hash(state)
    state = state.model_copy(
        update={
            "frozen": True,
            "frozen_at_unix": int(time.time()),
            "frozen_taxonomy_hash": frozen_hash,
            "updated_at_unix": int(time.time()),
        }
    )
    validate_taxonomy_state(
        state,
        max_total_labels=args.max_total_labels,
        expected_seed_count=args.expected_seed_label_count,
    )
    atomic_write_model(taxonomy_path, state)

    LOGGER.info(
        "Discovery complete. Frozen taxonomy v%d with %d active labels. Hash=%s",
        state.schema_version,
        len(active_labels(state)),
        frozen_hash,
    )
