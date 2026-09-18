#!/usr/bin/env python3
"""Sweep lexical candidate-consolidation thresholds for an existing taxonomy state.

This is a standalone diagnostic script. It does NOT import vLLM, Pydantic, or the
label pipeline package, and it never modifies the input taxonomy_state.json.

It reproduces the cheap lexical blocking/neighborhood construction used by the
candidate-consolidation stage:

- normalize candidate names;
- remove a small stopword set;
- score names with max(SequenceMatcher(sorted tokens), token overlap coefficient);
- retrieve plausible neighbors through a shared-token inverted index;
- build disjoint neighborhoods greedily in group_id order;
- require every pair inside a neighborhood to meet the selected threshold
  (i.e. no naive connected-component chaining).

The LLM semantic consolidation step is intentionally NOT run here. The output shows
which candidates would be *presented together to the LLM* at each lexical threshold.

Outputs:
  consolidation_threshold_summary.csv
  consolidation_threshold_groups.csv
  consolidation_pair_scores.csv
  consolidation_threshold_sweep.json
  consolidation_threshold_sweep.html

Example:
  python candidate_consolidation_threshold_sweep.py taxonomy_state.json \
      --thresholds 0.70,0.75,0.80,0.85,0.90,0.95 \
      --output-dir threshold_sweep
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_THRESHOLDS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
DEFAULT_MAX_GROUP_SIZE = 12
DEFAULT_PROMOTION_SUPPORT = 2

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "for",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)


@dataclass(frozen=True)
class Candidate:
    group_id: str
    name: str
    definition: str
    support_count: int


@dataclass(frozen=True)
class PairScore:
    left_id: str
    right_id: str
    score: float
    sequence_score: float
    overlap_score: float
    shared_tokens: tuple[str, ...]


@dataclass(frozen=True)
class SweepGroup:
    threshold: float
    sweep_group_id: str
    members: tuple[Candidate, ...]
    min_pair_similarity: float
    mean_pair_similarity: float
    combined_support: int


@dataclass(frozen=True)
class SweepResult:
    threshold: float
    groups: tuple[SweepGroup, ...]
    singleton_ids: tuple[str, ...]


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def candidate_name_tokens(value: str) -> tuple[str, ...]:
    normalized = value.casefold().replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return tuple(
        sorted(
            {
                token
                for token in normalized.split()
                if token and token not in STOPWORDS
            }
        )
    )


def similarity_components(left: Candidate, right: Candidate) -> PairScore:
    left_name = normalize_text(left.name).replace("&", " and ")
    right_name = normalize_text(right.name).replace("&", " and ")

    left_tokens = candidate_name_tokens(left.name)
    right_tokens = candidate_name_tokens(right.name)
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    shared = tuple(sorted(left_set & right_set))

    if left_name == right_name:
        sequence_score = 1.0
        overlap_score = 1.0 if left_set and right_set else 0.0
        score = 1.0
    elif not left_tokens or not right_tokens:
        sequence_score = SequenceMatcher(None, left_name, right_name).ratio()
        overlap_score = 0.0
        score = sequence_score
    else:
        sorted_left = " ".join(left_tokens)
        sorted_right = " ".join(right_tokens)
        sequence_score = SequenceMatcher(None, sorted_left, sorted_right).ratio()
        overlap_score = 0.0
        if len(shared) >= 2:
            overlap_score = len(shared) / min(len(left_set), len(right_set))
        score = max(sequence_score, overlap_score)

    left_id, right_id = sorted((left.group_id, right.group_id))
    return PairScore(
        left_id=left_id,
        right_id=right_id,
        score=score,
        sequence_score=sequence_score,
        overlap_score=overlap_score,
        shared_tokens=shared,
    )


def parse_thresholds(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value > 1.0:
            value /= 100.0
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Threshold {part!r} is outside [0, 1] / [0, 100]")
        values.append(value)
    if not values:
        raise ValueError("At least one threshold is required")
    return sorted(set(values))


def load_taxonomy(path: Path) -> tuple[dict[str, Any], list[Candidate]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("taxonomy_state.json must contain a JSON object")

    raw_candidates = payload.get("candidate_proposals")
    if not isinstance(raw_candidates, list):
        raise ValueError("taxonomy state has no candidate_proposals list")

    candidates: list[Candidate] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_candidates, start=1):
        if not isinstance(raw, dict):
            raise TypeError(f"candidate_proposals[{index - 1}] must be an object")
        try:
            group_id = raw["group_id"]
            name = raw["name"]
            definition = raw["definition"]
            support_count = raw["support_count"]
        except KeyError as exc:
            raise ValueError(
                f"candidate_proposals[{index - 1}] is missing {exc.args[0]!r}"
            ) from exc

        if not isinstance(group_id, str) or not group_id:
            raise TypeError(f"candidate_proposals[{index - 1}].group_id must be non-empty")
        if group_id in seen_ids:
            raise ValueError(f"Duplicate candidate group_id {group_id!r}")
        seen_ids.add(group_id)
        if not isinstance(name, str) or not name:
            raise TypeError(f"candidate {group_id}: name must be a non-empty string")
        if not isinstance(definition, str) or not definition:
            raise TypeError(f"candidate {group_id}: definition must be a non-empty string")
        if isinstance(support_count, bool) or not isinstance(support_count, int) or support_count < 1:
            raise TypeError(f"candidate {group_id}: support_count must be an integer >= 1")

        candidates.append(
            Candidate(
                group_id=group_id,
                name=name,
                definition=definition,
                support_count=support_count,
            )
        )

    return payload, candidates


def reviewed_non_equivalence_pairs(
    taxonomy: dict[str, Any], current_ids: set[str]
) -> set[frozenset[str]]:
    """Extract non-equivalent candidate pairs previously reviewed by the LLM.

    This mirrors reviewed_candidate_non_equivalence_pairs() in the pipeline. It is
    optional in this diagnostic because a pure threshold sweep is usually easier to
    interpret when historical exclusions are ignored.
    """

    excluded: set[frozenset[str]] = set()
    history = taxonomy.get("history", [])
    if not isinstance(history, list):
        return excluded

    for entry in history:
        if not isinstance(entry, dict) or entry.get("kind") != "candidate_consolidation":
            continue
        consolidation = entry.get("consolidation")
        if not isinstance(consolidation, dict):
            continue
        assignments = consolidation.get("assignments")
        if not isinstance(assignments, dict):
            continue

        reviewed_ids = sorted(set(assignments) & current_ids)
        for index, left_id in enumerate(reviewed_ids):
            for right_id in reviewed_ids[index + 1 :]:
                if assignments.get(left_id) != assignments.get(right_id):
                    excluded.add(frozenset((left_id, right_id)))
    return excluded


def build_token_index(candidates: Sequence[Candidate]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    tokens_by_id = {
        candidate.group_id: set(candidate_name_tokens(candidate.name))
        for candidate in candidates
    }
    token_index: dict[str, set[str]] = {}
    for group_id, tokens in tokens_by_id.items():
        for token in tokens:
            token_index.setdefault(token, set()).add(group_id)
    return tokens_by_id, token_index


def build_pair_cache(
    candidates: Sequence[Candidate],
    tokens_by_id: dict[str, set[str]],
    token_index: dict[str, set[str]],
) -> dict[tuple[str, str], PairScore]:
    """Score only pairs reachable through the same shared-token blocker as runtime."""

    by_id = {candidate.group_id: candidate for candidate in candidates}
    candidate_pairs: set[tuple[str, str]] = set()
    for candidate in candidates:
        neighbor_ids: set[str] = set()
        for token in tokens_by_id[candidate.group_id]:
            neighbor_ids.update(token_index.get(token, set()))
        neighbor_ids.discard(candidate.group_id)
        for other_id in neighbor_ids:
            candidate_pairs.add(tuple(sorted((candidate.group_id, other_id))))

    cache: dict[tuple[str, str], PairScore] = {}
    for left_id, right_id in sorted(candidate_pairs):
        cache[(left_id, right_id)] = similarity_components(
            by_id[left_id], by_id[right_id]
        )
    return cache


def pair_score(
    left: Candidate,
    right: Candidate,
    cache: dict[tuple[str, str], PairScore],
) -> PairScore:
    key = tuple(sorted((left.group_id, right.group_id)))
    result = cache.get(key)
    if result is None:
        # Runtime's clique check can compare a candidate against a member reached
        # through another shared-token edge, so compute lazily when necessary.
        result = similarity_components(left, right)
        cache[key] = result
    return result


def build_groups_for_threshold(
    candidates: Sequence[Candidate],
    *,
    threshold: float,
    max_group_size: int,
    excluded_pairs: set[frozenset[str]],
    tokens_by_id: dict[str, set[str]],
    token_index: dict[str, set[str]],
    score_cache: dict[tuple[str, str], PairScore],
) -> SweepResult:
    if max_group_size < 2:
        raise ValueError("max_group_size must be >= 2")

    ordered = sorted(candidates, key=lambda candidate: candidate.group_id)
    by_id = {candidate.group_id: candidate for candidate in ordered}
    remaining = {candidate.group_id for candidate in ordered}
    groups: list[SweepGroup] = []

    def eligible(left: Candidate, right: Candidate) -> bool:
        if frozenset((left.group_id, right.group_id)) in excluded_pairs:
            return False
        return pair_score(left, right, score_cache).score >= threshold

    for seed in ordered:
        if seed.group_id not in remaining:
            continue

        potential_neighbor_ids: set[str] = set()
        for token in tokens_by_id[seed.group_id]:
            potential_neighbor_ids.update(token_index.get(token, set()))
        potential_neighbor_ids &= remaining
        potential_neighbor_ids.discard(seed.group_id)

        neighbors: list[tuple[float, str]] = []
        for candidate_id in potential_neighbor_ids:
            candidate = by_id[candidate_id]
            if eligible(seed, candidate):
                neighbors.append(
                    (pair_score(seed, candidate, score_cache).score, candidate_id)
                )
        neighbors.sort(key=lambda item: (-item[0], item[1]))

        batch = [seed]
        for _, candidate_id in neighbors:
            if len(batch) >= max_group_size:
                break
            candidate = by_id[candidate_id]
            if all(eligible(candidate, member) for member in batch):
                batch.append(candidate)

        for member in batch:
            remaining.remove(member.group_id)

        if len(batch) >= 2:
            similarities = [
                pair_score(batch[i], batch[j], score_cache).score
                for i in range(len(batch))
                for j in range(i + 1, len(batch))
            ]
            groups.append(
                SweepGroup(
                    threshold=threshold,
                    sweep_group_id=f"T{threshold:.3f}_G{len(groups) + 1:04d}",
                    members=tuple(batch),
                    min_pair_similarity=min(similarities),
                    mean_pair_similarity=sum(similarities) / len(similarities),
                    combined_support=sum(member.support_count for member in batch),
                )
            )

    grouped_ids = {member.group_id for group in groups for member in group.members}
    singleton_ids = tuple(
        candidate.group_id for candidate in ordered if candidate.group_id not in grouped_ids
    )
    return SweepResult(
        threshold=threshold,
        groups=tuple(groups),
        singleton_ids=singleton_ids,
    )


def threshold_label(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def write_summary_csv(
    path: Path,
    results: Sequence[SweepResult],
    *,
    total_candidates: int,
    promotion_support: int,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "threshold",
                "total_candidates",
                "num_multi_candidate_groups",
                "grouped_candidates",
                "singleton_candidates",
                "groups_reaching_promotion_support",
                "grouped_support_total",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "threshold": threshold_label(result.threshold),
                    "total_candidates": total_candidates,
                    "num_multi_candidate_groups": len(result.groups),
                    "grouped_candidates": sum(len(group.members) for group in result.groups),
                    "singleton_candidates": len(result.singleton_ids),
                    "groups_reaching_promotion_support": sum(
                        group.combined_support >= promotion_support
                        for group in result.groups
                    ),
                    "grouped_support_total": sum(
                        group.combined_support for group in result.groups
                    ),
                }
            )


def write_groups_csv(
    path: Path,
    results: Sequence[SweepResult],
    candidates: Sequence[Candidate],
    *,
    promotion_support: int,
) -> None:
    by_id = {candidate.group_id: candidate for candidate in candidates}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "threshold",
                "sweep_group_id",
                "member_count",
                "candidate_group_id",
                "candidate_name",
                "candidate_definition",
                "candidate_support",
                "combined_support",
                "reaches_promotion_support",
                "min_pair_similarity",
                "mean_pair_similarity",
                "status",
            ],
        )
        writer.writeheader()
        for result in results:
            grouped_ids: set[str] = set()
            for group in result.groups:
                for member in group.members:
                    grouped_ids.add(member.group_id)
                    writer.writerow(
                        {
                            "threshold": threshold_label(result.threshold),
                            "sweep_group_id": group.sweep_group_id,
                            "member_count": len(group.members),
                            "candidate_group_id": member.group_id,
                            "candidate_name": member.name,
                            "candidate_definition": member.definition,
                            "candidate_support": member.support_count,
                            "combined_support": group.combined_support,
                            "reaches_promotion_support": group.combined_support
                            >= promotion_support,
                            "min_pair_similarity": f"{group.min_pair_similarity:.6f}",
                            "mean_pair_similarity": f"{group.mean_pair_similarity:.6f}",
                            "status": "grouped",
                        }
                    )
            for candidate_id in result.singleton_ids:
                candidate = by_id[candidate_id]
                writer.writerow(
                    {
                        "threshold": threshold_label(result.threshold),
                        "sweep_group_id": "",
                        "member_count": 1,
                        "candidate_group_id": candidate.group_id,
                        "candidate_name": candidate.name,
                        "candidate_definition": candidate.definition,
                        "candidate_support": candidate.support_count,
                        "combined_support": candidate.support_count,
                        "reaches_promotion_support": candidate.support_count
                        >= promotion_support,
                        "min_pair_similarity": "",
                        "mean_pair_similarity": "",
                        "status": "singleton",
                    }
                )


def write_pair_scores_csv(
    path: Path,
    pair_cache: dict[tuple[str, str], PairScore],
    candidates: Sequence[Candidate],
    thresholds: Sequence[float],
    *,
    excluded_pairs: set[frozenset[str]],
) -> None:
    by_id = {candidate.group_id: candidate for candidate in candidates}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "left_id",
                "left_name",
                "right_id",
                "right_name",
                "score",
                "sequence_score",
                "overlap_score",
                "shared_tokens",
                "historically_excluded",
                "eligible_thresholds",
            ],
        )
        writer.writeheader()
        for key in sorted(pair_cache):
            pair = pair_cache[key]
            writer.writerow(
                {
                    "left_id": pair.left_id,
                    "left_name": by_id[pair.left_id].name,
                    "right_id": pair.right_id,
                    "right_name": by_id[pair.right_id].name,
                    "score": f"{pair.score:.6f}",
                    "sequence_score": f"{pair.sequence_score:.6f}",
                    "overlap_score": f"{pair.overlap_score:.6f}",
                    "shared_tokens": " | ".join(pair.shared_tokens),
                    "historically_excluded": frozenset((pair.left_id, pair.right_id))
                    in excluded_pairs,
                    "eligible_thresholds": ",".join(
                        threshold_label(threshold)
                        for threshold in thresholds
                        if pair.score >= threshold
                    ),
                }
            )


def result_to_json(
    taxonomy_path: Path,
    taxonomy: dict[str, Any],
    candidates: Sequence[Candidate],
    results: Sequence[SweepResult],
    *,
    max_group_size: int,
    promotion_support: int,
    respect_history: bool,
    excluded_pairs: set[frozenset[str]],
) -> dict[str, Any]:
    return {
        "input_taxonomy": str(taxonomy_path.resolve()),
        "taxonomy_format_version": taxonomy.get("format_version"),
        "taxonomy_schema_version": taxonomy.get("schema_version"),
        "candidate_count": len(candidates),
        "total_candidate_support": sum(candidate.support_count for candidate in candidates),
        "max_group_size": max_group_size,
        "promotion_support": promotion_support,
        "respect_history": respect_history,
        "historically_excluded_pair_count": len(excluded_pairs),
        "thresholds": [result.threshold for result in results],
        "results": [
            {
                "threshold": result.threshold,
                "groups": [
                    {
                        "sweep_group_id": group.sweep_group_id,
                        "member_count": len(group.members),
                        "combined_support": group.combined_support,
                        "reaches_promotion_support": group.combined_support
                        >= promotion_support,
                        "min_pair_similarity": group.min_pair_similarity,
                        "mean_pair_similarity": group.mean_pair_similarity,
                        "members": [
                            {
                                "group_id": member.group_id,
                                "name": member.name,
                                "definition": member.definition,
                                "support_count": member.support_count,
                            }
                            for member in group.members
                        ],
                    }
                    for group in result.groups
                ],
                "singleton_ids": list(result.singleton_ids),
            }
            for result in results
        ],
    }


def write_html_report(
    path: Path,
    report: dict[str, Any],
    candidates: Sequence[Candidate],
) -> None:
    by_id = {candidate.group_id: candidate for candidate in candidates}
    rows = []
    for result in report["results"]:
        threshold = result["threshold"]
        groups = result["groups"]
        grouped_count = sum(group["member_count"] for group in groups)
        rows.append(
            f"<tr><td>{threshold:.2f}</td><td>{len(groups)}</td>"
            f"<td>{grouped_count}</td><td>{len(result['singleton_ids'])}</td>"
            f"<td>{sum(1 for g in groups if g['reaches_promotion_support'])}</td></tr>"
        )

    sections = []
    for result in report["results"]:
        threshold = result["threshold"]
        group_blocks = []
        for group in result["groups"]:
            member_rows = "".join(
                "<tr>"
                f"<td><code>{html.escape(member['group_id'])}</code></td>"
                f"<td>{html.escape(member['name'])}</td>"
                f"<td>{member['support_count']}</td>"
                f"<td>{html.escape(member['definition'])}</td>"
                "</tr>"
                for member in group["members"]
            )
            status = "YES" if group["reaches_promotion_support"] else "no"
            group_blocks.append(
                "<div class='group'>"
                f"<h3>{html.escape(group['sweep_group_id'])} &mdash; "
                f"{group['member_count']} candidates, support={group['combined_support']}, "
                f"promotion support reached: {status}</h3>"
                f"<p>min pair similarity={group['min_pair_similarity']:.3f}; "
                f"mean pair similarity={group['mean_pair_similarity']:.3f}</p>"
                "<table><thead><tr><th>ID</th><th>Name</th><th>Support</th><th>Definition</th>"
                f"</tr></thead><tbody>{member_rows}</tbody></table></div>"
            )

        singleton_rows = "".join(
            "<tr>"
            f"<td><code>{html.escape(candidate_id)}</code></td>"
            f"<td>{html.escape(by_id[candidate_id].name)}</td>"
            f"<td>{by_id[candidate_id].support_count}</td>"
            "</tr>"
            for candidate_id in result["singleton_ids"]
        )
        sections.append(
            f"<details {'open' if math.isclose(threshold, 0.80) else ''}>"
            f"<summary>Threshold {threshold:.2f}: {len(result['groups'])} groups</summary>"
            + "".join(group_blocks)
            + ("<h3>Singletons</h3><table><thead><tr><th>ID</th><th>Name</th><th>Support</th>"
               f"</tr></thead><tbody>{singleton_rows}</tbody></table>" if singleton_rows else "")
            + "</details>"
        )

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Candidate consolidation threshold sweep</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1500px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
table {{ border-collapse: collapse; width: 100%; margin: .75rem 0 1.5rem; }}
th, td {{ border: 1px solid #ccc; padding: .45rem .55rem; text-align: left; vertical-align: top; }}
th {{ background: #f3f3f3; }}
code {{ white-space: nowrap; }}
details {{ margin: 1.2rem 0; border: 1px solid #ddd; padding: .75rem; border-radius: 6px; }}
summary {{ font-size: 1.15rem; font-weight: 650; cursor: pointer; }}
.group {{ border-top: 1px solid #ddd; margin-top: 1rem; padding-top: .5rem; }}
.small {{ color: #555; }}
</style></head><body>
<h1>Candidate consolidation threshold sweep</h1>
<p class="small">Input: <code>{html.escape(report['input_taxonomy'])}</code></p>
<p>{report['candidate_count']} candidates; total support={report['total_candidate_support']}; "
max group size={report['max_group_size']}; promotion support={report['promotion_support']}; "
respect historical non-equivalence={str(report['respect_history']).lower()}.</p>
<h2>Summary</h2>
<table><thead><tr><th>Threshold</th><th>Multi-candidate groups</th><th>Grouped candidates</th><th>Singletons</th><th>Groups reaching promotion support</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Groups by threshold</h2>
{''.join(sections)}
</body></html>"""
    path.write_text(document, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the lexical candidate-consolidation threshold over an existing "
            "taxonomy_state.json without loading vLLM or modifying the taxonomy."
        )
    )
    parser.add_argument("taxonomy", type=Path, help="Path to taxonomy_state.json")
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
        help=(
            "Comma-separated similarity thresholds, either 0-1 or percentages. "
            "Default: 0.70,0.75,0.80,0.85,0.90,0.95"
        ),
    )
    parser.add_argument(
        "--max-group-size",
        type=int,
        default=None,
        help=(
            "Maximum candidates per lexical neighborhood. Defaults to "
            "discovery_settings.candidate_consolidation_max_groups or 12."
        ),
    )
    parser.add_argument(
        "--promotion-support",
        type=int,
        default=None,
        help=(
            "Support threshold used only for report annotation. Defaults to "
            "discovery_settings.min_promotion_support or 2."
        ),
    )
    parser.add_argument(
        "--respect-history",
        action="store_true",
        help=(
            "Exclude candidate pairs previously judged non-equivalent in "
            "candidate_consolidation history. Default is a pure lexical threshold sweep."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for reports. Default: <taxonomy stem>_consolidation_threshold_sweep "
            "next to the taxonomy file."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    taxonomy_path = args.taxonomy
    taxonomy, candidates = load_taxonomy(taxonomy_path)
    thresholds = parse_thresholds(args.thresholds)

    settings = taxonomy.get("discovery_settings")
    if not isinstance(settings, dict):
        settings = {}

    max_group_size = args.max_group_size
    if max_group_size is None:
        max_group_size = settings.get(
            "candidate_consolidation_max_groups", DEFAULT_MAX_GROUP_SIZE
        )
    if isinstance(max_group_size, bool) or not isinstance(max_group_size, int) or max_group_size < 2:
        raise ValueError("--max-group-size must be an integer >= 2")

    promotion_support = args.promotion_support
    if promotion_support is None:
        promotion_support = settings.get(
            "min_promotion_support", DEFAULT_PROMOTION_SUPPORT
        )
    if (
        isinstance(promotion_support, bool)
        or not isinstance(promotion_support, int)
        or promotion_support < 1
    ):
        raise ValueError("--promotion-support must be an integer >= 1")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = taxonomy_path.with_name(
            taxonomy_path.stem + "_consolidation_threshold_sweep"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    current_ids = {candidate.group_id for candidate in candidates}
    excluded_pairs = (
        reviewed_non_equivalence_pairs(taxonomy, current_ids)
        if args.respect_history
        else set()
    )

    tokens_by_id, token_index = build_token_index(candidates)
    pair_cache = build_pair_cache(candidates, tokens_by_id, token_index)

    results = [
        build_groups_for_threshold(
            candidates,
            threshold=threshold,
            max_group_size=max_group_size,
            excluded_pairs=excluded_pairs,
            tokens_by_id=tokens_by_id,
            token_index=token_index,
            score_cache=pair_cache,
        )
        for threshold in thresholds
    ]

    summary_csv = output_dir / "consolidation_threshold_summary.csv"
    groups_csv = output_dir / "consolidation_threshold_groups.csv"
    pairs_csv = output_dir / "consolidation_pair_scores.csv"
    json_path = output_dir / "consolidation_threshold_sweep.json"
    html_path = output_dir / "consolidation_threshold_sweep.html"

    write_summary_csv(
        summary_csv,
        results,
        total_candidates=len(candidates),
        promotion_support=promotion_support,
    )
    write_groups_csv(
        groups_csv,
        results,
        candidates,
        promotion_support=promotion_support,
    )
    write_pair_scores_csv(
        pairs_csv,
        pair_cache,
        candidates,
        thresholds,
        excluded_pairs=excluded_pairs,
    )

    report = result_to_json(
        taxonomy_path,
        taxonomy,
        candidates,
        results,
        max_group_size=max_group_size,
        promotion_support=promotion_support,
        respect_history=args.respect_history,
        excluded_pairs=excluded_pairs,
    )
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_html_report(html_path, report, candidates)

    print(f"Loaded {len(candidates)} candidate proposals from {taxonomy_path}")
    print(
        "Thresholds: "
        + ", ".join(threshold_label(threshold) for threshold in thresholds)
    )
    print(f"Wrote reports to {output_dir.resolve()}")
    for result in results:
        grouped = sum(len(group.members) for group in result.groups)
        support_ready = sum(
            group.combined_support >= promotion_support for group in result.groups
        )
        print(
            f"  threshold={result.threshold:.2f}: groups={len(result.groups)}, "
            f"grouped_candidates={grouped}, singletons={len(result.singleton_ids)}, "
            f"groups_support>={promotion_support}={support_ready}"
        )


if __name__ == "__main__":
    main()
