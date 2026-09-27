"""Ranking metrics for binary relevance.

Every function takes the ranked ids a retriever returned and the set of ids a
human labeled as correct. Ids can be anything hashable.
"""
from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Sequence


def first_hit_rank(ranked: Sequence[Hashable], relevant: Iterable[Hashable]) -> int | None:
    """1-based rank of the first relevant id, or None if none was returned."""
    rel = set(relevant)
    for i, rid in enumerate(ranked):
        if rid in rel:
            return i + 1
    return None


def recall_at_k(ranked: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    """1.0 if any relevant id is in the top k, else 0.0 (a.k.a. hit rate@k).

    With one or two labeled answers per query this is the number people mean
    by "recall@k" in RAG evaluation; for graded, many-answer sets use
    `fraction_recalled_at_k` instead.
    """
    rel = set(relevant)
    return 1.0 if any(rid in rel for rid in ranked[:k]) else 0.0


def fraction_recalled_at_k(ranked: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    """Share of the relevant ids that appear in the top k."""
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(rel.intersection(ranked[:k])) / len(rel)


def reciprocal_rank(ranked: Sequence[Hashable], relevant: Iterable[Hashable]) -> float:
    rank = first_hit_rank(ranked, relevant)
    return 1.0 / rank if rank else 0.0


def ndcg_at_k(ranked: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    rel = set(relevant)
    dcg = sum(1.0 / math.log2(i + 2) for i, rid in enumerate(ranked[:k]) if rid in rel)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal if ideal else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile; stable for the small samples evals produce."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(p / 100 * len(ordered)) - 1))
    return ordered[idx]
