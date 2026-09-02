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
from label_pipeline_lib.model import ModelClient
from schemas import (
    DiscoveryRecord,
    DiscoverySettings,
    DynamicLabelDef,
    DynamicLabelChange,
    InputDocument,
    LabelDef,
    ProposalGroup,
    ProposalResolution,
    ProposedLabelRecord,
    ReconciliationOutput,
    SchemaHistoryEntry,
    TaxonomyState,
    FinalReconciliationOutput,
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
            else:
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
    updated = []
    for label in state.dynamic_labels:
        updated.append(label.model_copy(update={"usage_count": usage[label.id]}))
    return state.model_copy(update={"dynamic_labels": updated})


def combine_proposal_groups(
    deferred: Sequence[ProposalGroup],
    new_groups: Sequence[ProposalGroup],
) -> list[ProposalGroup]:
    combined: dict[str, ProposalGroup] = {group.group_id: group for group in deferred}
    for group in new_groups:
        existing = combined.get(group.group_id)
        if existing is None:
            combined[group.group_id] = group
            continue
        if normalize_text(existing.name) != normalize_text(
            group.name
        ) or normalize_text(existing.definition) != normalize_text(group.definition):
            raise RuntimeError("Proposal-group hash collision")
        combined[group.group_id] = existing.model_copy(
            update={"support_count": existing.support_count + group.support_count}
        )
    return list(combined.values())


def validate_reconciliation_semantics(
    state: TaxonomyState,
    proposal_groups: Sequence[ProposalGroup],
    result: ReconciliationOutput | FinalReconciliationOutput,
    *,
    min_create_support: int,
    allow_defer: bool,
    max_total_labels: int,
) -> None:
    expected_group_ids = {group.group_id for group in proposal_groups}

    seen_group_ids = [
        group_id
        for resolution in result.proposal_resolutions
        for group_id in resolution.proposal_group_ids
    ]

    # Defensive check. The Pydantic output model should already catch this,
    # but retaining it here is appropriate for a fail-fast research pipeline.
    if len(seen_group_ids) != len(set(seen_group_ids)):
        raise ValueError("Reconciler resolved a proposal group more than once")

    seen_group_id_set = set(seen_group_ids)

    if seen_group_id_set != expected_group_ids:
        missing = expected_group_ids - seen_group_id_set
        unknown = seen_group_id_set - expected_group_ids

        raise ValueError(
            "Reconciler proposal coverage mismatch; "
            f"missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )

    group_by_id = {group.group_id: group for group in proposal_groups}

    active = active_labels(state)
    active_ids = {label.id for label in active}
    dynamic_ids = {label.id for label in state.dynamic_labels}

    # Work this out before validating proposal resolutions, because
    # map_existing must not target a label that disappears in this same
    # reconciliation.
    merge_sources = {
        change.source_label_id
        for change in result.dynamic_label_changes
        if change.action == "merge"
    }

    # ------------------------------------------------------------------
    # Proposal resolutions
    # ------------------------------------------------------------------

    num_creates = 0

    for resolution in result.proposal_resolutions:

        if resolution.action == "defer":
            if not allow_defer:
                raise ValueError(
                    "Reconciler deferred a proposal during final maintenance"
                )

            # This is already structurally enforced by DeferResolution,
            # but keep the semantic assertion as defense in depth.
            if len(resolution.proposal_group_ids) != 1:
                raise ValueError(
                    "A defer resolution must contain exactly one proposal "
                    "group so candidate evidence remains attributable"
                )

        elif resolution.action == "reject":
            # Nothing additional to validate against taxonomy state.
            pass

        elif resolution.action == "map_existing":
            target_id = resolution.target_label_id

            if target_id not in active_ids:
                raise ValueError(f"Reconciler mapped to unknown label {target_id}")

            if target_id in merge_sources:
                raise ValueError(
                    f"Proposal mapped to {target_id}, but that label is "
                    "merged away in the same reconciliation"
                )

        elif resolution.action == "create":
            support = sum(
                group_by_id[group_id].support_count
                for group_id in resolution.proposal_group_ids
            )

            if support < min_create_support:
                raise ValueError(
                    f"Create action has support {support}, below minimum "
                    f"{min_create_support}"
                )

            num_creates += 1

        else:
            # In principle unreachable because of the Pydantic schema.
            raise ValueError(
                f"Unknown proposal resolution action: " f"{resolution.action!r}"
            )

    # ------------------------------------------------------------------
    # Dynamic-label changes
    # ------------------------------------------------------------------

    num_merges = 0

    for change in result.dynamic_label_changes:

        if change.source_label_id not in dynamic_ids:
            raise ValueError(
                "Reconciler attempted to modify non-dynamic label "
                f"{change.source_label_id}"
            )

        if change.action == "merge":
            target_id = change.target_label_id

            if target_id not in active_ids:
                raise ValueError(f"Merge target {target_id} is not an active label")

            if target_id in merge_sources:
                raise ValueError(
                    "Merge chains within one reconciliation are forbidden; "
                    f"target {target_id} is also a merge source"
                )

            num_merges += 1

        elif change.action == "revise":
            # name and definition are structurally required by
            # ReviseDynamicLabel, so there is no additional state-dependent
            # validation needed here.
            pass

        else:
            raise ValueError(f"Unknown dynamic-label action: {change.action!r}")

    # ------------------------------------------------------------------
    # Resulting taxonomy size
    # ------------------------------------------------------------------

    resulting_active_count = len(active) - num_merges + num_creates

    if resulting_active_count > max_total_labels:
        raise ValueError(
            "Reconciliation exceeds taxonomy-size limit: "
            f"{len(active)} current "
            f"- {num_merges} merges "
            f"+ {num_creates} creations "
            f"= {resulting_active_count}, "
            f"maximum is {max_total_labels}"
        )


def apply_reconciliation(
    state: TaxonomyState,
    proposal_groups: Sequence[ProposalGroup],
    result: ReconciliationOutput | FinalReconciliationOutput,
    *,
    discovery_seq_end: int,
    max_total_labels: int,
    min_create_support: int,
    maintenance_kind: str,
) -> TaxonomyState:
    validate_reconciliation_semantics(
        state,
        proposal_groups,
        result,
        min_create_support=min_create_support,
        allow_defer=(maintenance_kind != "final"),
        max_total_labels=max_total_labels,
    )

    before_version = state.schema_version
    taxonomy_changes_requested = bool(result.dynamic_label_changes) or any(
        resolution.action == "create" for resolution in result.proposal_resolutions
    )
    after_version = before_version + 1 if taxonomy_changes_requested else before_version

    dynamic_by_id = {label.id: label for label in state.dynamic_labels}
    alias_map = dict(state.alias_map)

    for change in result.dynamic_label_changes:
        if change.action != "merge":
            continue
        source = dynamic_by_id.pop(change.source_label_id)
        assert change.target_label_id is not None
        target_id = resolve_alias(change.target_label_id, alias_map)

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

    for change in result.dynamic_label_changes:
        if change.action != "revise":
            continue
        label = dynamic_by_id[change.source_label_id]
        dynamic_by_id[label.id] = label.model_copy(
            update={
                "name": change.name if change.name is not None else label.name,
                "definition": (
                    change.definition
                    if change.definition is not None
                    else label.definition
                ),
                "last_modified_version": after_version,
            }
        )

    group_by_id = {group.group_id: group for group in proposal_groups}
    created_labels: list[DynamicLabelDef] = []

    state_for_id_allocation = state.model_copy(
        update={
            "dynamic_labels": list(dynamic_by_id.values()),
            "alias_map": alias_map,
        }
    )
    for resolution in result.proposal_resolutions:
        if resolution.action != "create":
            continue
        assert resolution.name is not None
        assert resolution.definition is not None
        support = sum(
            group_by_id[group_id].support_count
            for group_id in resolution.proposal_group_ids
        )
        new_id = next_dynamic_id(state_for_id_allocation)
        created = DynamicLabelDef(
            id=new_id,
            name=resolution.name,
            definition=resolution.definition,
            created_version=after_version,
            last_modified_version=after_version,
            usage_count=0,
            proposal_support_count=support,
        )
        dynamic_by_id[new_id] = created
        state_for_id_allocation.dynamic_labels.append(created)
        created_labels.append(created)

    deferred_by_id: dict[str, ProposalGroup] = {}
    for resolution in result.proposal_resolutions:
        if resolution.action != "defer":
            continue
        group_id = resolution.proposal_group_ids[0]
        deferred_by_id[group_id] = group_by_id[group_id]

    dynamic_labels = list(dynamic_by_id.values())
    candidate = state.model_copy(
        update={
            "schema_version": after_version,
            "dynamic_labels": dynamic_labels,
            "alias_map": alias_map,
            "deferred_proposals": list(deferred_by_id.values()),
            "next_dynamic_label_num": state_for_id_allocation.next_dynamic_label_num,
            "last_reconciled_discovery_seq": discovery_seq_end,
            "updated_at_unix": int(time.time()),
        }
    )

    validate_unique_labels(active_labels(candidate), context="reconciled taxonomy")
    if len(active_labels(candidate)) > max_total_labels:
        raise ValueError(
            f"Reconciliation would produce {len(active_labels(candidate))} active "
            f"labels, exceeding cap {max_total_labels}"
        )

    history_entry = SchemaHistoryEntry(
        timestamp_unix=int(time.time()),
        maintenance_kind=maintenance_kind,
        discovery_seq_end=discovery_seq_end,
        version_before=before_version,
        version_after=after_version,
        proposal_groups=list(proposal_groups),
        reconciliation=result,
        created_labels=created_labels,
    )
    candidate = candidate.model_copy(
        update={"history": [*candidate.history, history_entry]}
    )
    validate_taxonomy_state(candidate, max_total_labels=max_total_labels)
    return candidate


def reconcile_pending(
    *,
    model: ModelClient,
    state: TaxonomyState,
    discovery_output: Path,
    taxonomy_path: Path,
    max_total_labels: int,
    min_create_support: int,
    max_reconcile_groups: int,
    reconcile_max_tokens: int,
    final_maintenance: bool,
) -> TaxonomyState:
    latest_seq, usage, new_proposal_groups = scan_discovery_results(
        discovery_output,
        state,
        after_seq=state.last_reconciled_discovery_seq,
    )
    state = update_usage_counts(state, usage)

    proposal_groups = combine_proposal_groups(
        state.deferred_proposals,
        new_proposal_groups,
    )

    if len(proposal_groups) > max_reconcile_groups:
        raise ValueError(
            f"Found {len(proposal_groups)} candidate proposal groups, exceeding "
            f"--max-reconcile-groups={max_reconcile_groups}. Increase that limit "
            "or reconcile more frequently so the candidate pool stays manageable."
        )

    if not new_proposal_groups and not final_maintenance:
        state = state.model_copy(
            update={
                "last_reconciled_discovery_seq": latest_seq,
                "updated_at_unix": int(time.time()),
            }
        )
        validate_taxonomy_state(state, max_total_labels=max_total_labels)
        atomic_write_model(taxonomy_path, state)
        return state

    if final_maintenance and not proposal_groups and not state.dynamic_labels:
        state = state.model_copy(
            update={
                "last_reconciled_discovery_seq": latest_seq,
                "updated_at_unix": int(time.time()),
            }
        )
        validate_taxonomy_state(state, max_total_labels=max_total_labels)
        atomic_write_model(taxonomy_path, state)
        return state

    result = model.reconcile(
        state,
        proposal_groups,
        max_total_labels=max_total_labels,
        min_create_support=min_create_support,
        max_tokens=reconcile_max_tokens,
        final_maintenance=final_maintenance,
    )
    state = apply_reconciliation(
        state,
        proposal_groups,
        result,
        discovery_seq_end=latest_seq,
        max_total_labels=max_total_labels,
        min_create_support=min_create_support,
        maintenance_kind="final" if final_maintenance else "periodic",
    )
    atomic_write_model(taxonomy_path, state)
    return state


def build_discovery_settings(
    args: argparse.Namespace, input_path: Path
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
        reconcile_max_tokens=args.reconcile_max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        seed=args.seed,
        thinking_mode=args.thinking_mode,
        reconcile_every=args.reconcile_every,
        max_reconcile_groups=args.max_reconcile_groups,
        min_create_support=args.min_create_support,
        max_proposals_per_doc=args.max_proposals_per_doc,
        max_total_labels=args.max_total_labels,
        expected_seed_label_count=args.expected_seed_label_count,
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
        aspect=aspect,
        schema_version=1,
        frozen=False,
        seed_labels_sha256=seed_hash,
        seed_labels=list(seed_labels),
        dynamic_labels=[],
        alias_map={},
        deferred_proposals=[],
        next_dynamic_label_num=25,
        last_reconciled_discovery_seq=0,
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
        state.last_reconciled_discovery_seq != 0
        or state.schema_version != 1
        or state.dynamic_labels
        or state.deferred_proposals
        or state.history
    ):
        raise ValueError(
            "Taxonomy state contains discovery progress but discovery output is missing"
        )

    completed_ids, last_seq = load_result_index(discovery_output, DiscoveryRecord)
    if last_seq < state.last_reconciled_discovery_seq:
        raise ValueError(
            "Taxonomy reconciliation cursor is ahead of persisted discovery output"
        )

    LOGGER.info(
        "Discovery resume state: %d completed documents, taxonomy version %d",
        len(completed_ids),
        state.schema_version,
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
    total_input = 0

    def pending_docs() -> Iterator[InputDocument]:
        nonlocal total_input
        for doc in stream_documents(input_path):
            total_input += 1
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
            "Discovery: persisted %d documents this run; latest seq=%d; taxonomy v%d",
            processed_this_run,
            last_seq,
            state.schema_version,
        )

        if last_seq - state.last_reconciled_discovery_seq >= args.reconcile_every:
            state = reconcile_pending(
                model=model,
                state=state,
                discovery_output=discovery_output,
                taxonomy_path=taxonomy_path,
                max_total_labels=args.max_total_labels,
                min_create_support=args.min_create_support,
                max_reconcile_groups=args.max_reconcile_groups,
                reconcile_max_tokens=args.reconcile_max_tokens,
                final_maintenance=False,
            )
            LOGGER.info(
                "Reconciled taxonomy: version=%d active_labels=%d",
                state.schema_version,
                len(active_labels(state)),
            )

    if total_input == 0:
        raise ValueError("Input corpus contains no documents")

    state = reconcile_pending(
        model=model,
        state=state,
        discovery_output=discovery_output,
        taxonomy_path=taxonomy_path,
        max_total_labels=args.max_total_labels,
        min_create_support=args.min_create_support,
        max_reconcile_groups=args.max_reconcile_groups,
        reconcile_max_tokens=args.reconcile_max_tokens,
        final_maintenance=False,
    )

    # Final maintenance reconciliation to ensure that all deferred proposals are resolved and the taxonomy is frozen.
    state = reconcile_pending(
        model=model,
        state=state,
        discovery_output=discovery_output,
        taxonomy_path=taxonomy_path,
        max_total_labels=args.max_total_labels,
        min_create_support=args.min_create_support,
        max_reconcile_groups=args.max_reconcile_groups,
        reconcile_max_tokens=args.reconcile_max_tokens,
        final_maintenance=True,
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
