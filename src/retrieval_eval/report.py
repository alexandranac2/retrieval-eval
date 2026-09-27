"""Human-readable output and the regression gate."""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .runner import Report


def table(reports: Sequence[Report]) -> str:
    """Markdown table: one row per configuration."""
    if not reports:
        return ""
    ks = list(reports[0].recall)
    head = ["config", *[f"R{k}" for k in ks], "MRR", f"nDCG{ks[-1]}", "p50 ms", "p95 ms"]
    rows = [
        [
            r.name,
            *[f"{r.recall[k]:.0%}" for k in ks],
            f"{r.mrr:.3f}",
            f"{r.ndcg[ks[-1]]:.3f}",
            f"{r.latency_ms['p50']:.1f}",
            f"{r.latency_ms['p95']:.1f}",
        ]
        for r in reports
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def disagreements(reports: Sequence[Report]) -> str:
    """The queries where configurations rank differently — where the real story is."""
    if len(reports) < 2:
        return ""
    out = ["| " + " | ".join([*(r.name for r in reports), "query"]) + " |", "|" + "---|" * (len(reports) + 1)]
    for i, case in enumerate(reports[0].cases):
        ranks = [r.cases[i].rank for r in reports]
        if len(set(ranks)) > 1:
            out.append("| " + " | ".join([*(str(x) if x else "–" for x in ranks), case.q]) + " |")
    return "\n".join(out) if len(out) > 2 else "(all configurations agree on every query)"


@dataclass
class Regression:
    metric: str
    baseline: float
    current: float

    def __str__(self) -> str:
        return f"{self.metric}: {self.baseline:.3f} → {self.current:.3f}"


def save_baseline(report: Report, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def check_regression(report: Report, baseline_path: str | Path, tolerance: float = 0.0) -> list[Regression]:
    """Metrics that dropped more than `tolerance` below the saved baseline.

    Latency is deliberately not gated: it depends on the machine. Quality
    metrics don't, so a drop there means the change broke something.
    """
    base = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    found: list[Regression] = []
    pairs = [("mrr", base["mrr"], report.mrr)]
    pairs += [(f"recall{k}", v, report.recall.get(k, 0.0)) for k, v in base["recall"].items()]
    pairs += [(f"ndcg{k}", v, report.ndcg.get(k, 0.0)) for k, v in base["ndcg"].items()]
    for metric, before, now in pairs:
        if now < before - tolerance:
            found.append(Regression(metric, before, now))
    return found
