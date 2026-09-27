"""Run a labeled query set through one or more retrievers and score them."""
from __future__ import annotations

import asyncio
import inspect
import json
import statistics
import time
from collections.abc import Awaitable, Callable, Hashable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Union

from . import metrics

Retriever = Callable[[str], Union[Sequence[Any], Awaitable[Sequence[Any]]]]


@dataclass(frozen=True)
class Case:
    q: str
    relevant: frozenset
    tags: tuple[str, ...] = ()


@dataclass
class CaseResult:
    q: str
    rank: int | None
    latency_ms: float
    tags: tuple[str, ...] = ()
    ranked: list = field(default_factory=list)
    relevant: list = field(default_factory=list)
    labels: dict = field(default_factory=dict)


@dataclass
class Report:
    name: str
    n: int
    recall: dict[str, float]
    mrr: float
    ndcg: dict[str, float]
    latency_ms: dict[str, float]
    cases: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cases"] = [asdict(c) for c in self.cases]
        return d


def save_cases(path: str | Path, cases: Sequence[Case]) -> None:
    """Write cases back as JSONL, keeping ids' JSON types (ints stay ints)."""
    lines = []
    for c in cases:
        row: dict[str, Any] = {"q": c.q, "relevant": sorted(c.relevant, key=str)}
        if c.tags:
            row["tags"] = list(c.tags)
        lines.append(json.dumps(row, ensure_ascii=False))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_cases(path: str | Path) -> list[Case]:
    """JSONL, one case per line: {"q": "...", "relevant": [id, ...], "tags": [...]}."""
    cases: list[Case] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        if not row.get("q") or not row.get("relevant"):
            raise ValueError(f"{path}:{lineno}: every case needs a non-empty 'q' and 'relevant'")
        cases.append(Case(q=row["q"], relevant=frozenset(row["relevant"]), tags=tuple(row.get("tags", ()))))
    return cases


async def _call(retriever: Retriever, q: str) -> Sequence[Any]:
    out = retriever(q)
    if inspect.isawaitable(out):
        out = await out
    return list(out)


def _split(items: Sequence[Any]) -> tuple[list[Hashable], dict[str, str]]:
    """Retrievers may return bare ids or {"id": ..., "title": ...} items; the
    titles are only for people reading results, never for scoring."""
    ids: list[Hashable] = []
    labels: dict[str, str] = {}
    for item in items:
        if isinstance(item, dict):
            ids.append(item["id"])
            label = item.get("title") or item.get("label")
            if label:
                labels[str(item["id"])] = str(label)
        else:
            ids.append(item)
    return ids, labels


async def evaluate(
    name: str,
    retriever: Retriever,
    cases: Sequence[Case],
    ks: Sequence[int] = (1, 3, 5),
    warmup: bool = True,
) -> Report:
    """Score one retriever. A warm-up pass runs first so caches (query
    embeddings, connection pools) don't make the first configuration look slow."""
    if warmup:
        for c in cases:
            await _call(retriever, c.q)
    max_k = max(ks)
    recall = {k: 0.0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    rr = 0.0
    results: list[CaseResult] = []
    for c in cases:
        t0 = time.perf_counter()
        ranked, labels = _split(list(await _call(retriever, c.q))[:max_k])
        latency = (time.perf_counter() - t0) * 1000
        for k in ks:
            recall[k] += metrics.recall_at_k(ranked, c.relevant, k)
            ndcg[k] += metrics.ndcg_at_k(ranked, c.relevant, k)
        rr += metrics.reciprocal_rank(ranked, c.relevant)
        results.append(
            CaseResult(
                c.q,
                metrics.first_hit_rank(ranked, c.relevant),
                latency,
                c.tags,
                ranked=list(ranked),
                relevant=sorted(c.relevant, key=str),
                labels=labels,
            )
        )
    n = len(cases) or 1
    lat = [r.latency_ms for r in results]
    return Report(
        name=name,
        n=len(cases),
        recall={f"@{k}": recall[k] / n for k in ks},
        mrr=rr / n,
        ndcg={f"@{k}": ndcg[k] / n for k in ks},
        latency_ms={"p50": statistics.median(lat) if lat else 0.0, "p95": metrics.percentile(lat, 95)},
        cases=results,
    )


async def compare(
    retrievers: dict[str, Retriever], cases: Sequence[Case], ks: Sequence[int] = (1, 3, 5)
) -> list[Report]:
    """Ablation: the same cases through every retriever, in order."""
    return [await evaluate(name, r, cases, ks) for name, r in retrievers.items()]


def run(retrievers: dict[str, Retriever], cases: Sequence[Case], ks: Sequence[int] = (1, 3, 5)) -> list[Report]:
    return asyncio.run(compare(retrievers, cases, ks))
