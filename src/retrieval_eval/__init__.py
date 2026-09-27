"""Small, honest retrieval evaluation: labeled queries in, recall@k / MRR / nDCG,
latency, per-query disagreements and a regression gate out."""
from .metrics import first_hit_rank, fraction_recalled_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from .report import check_regression, disagreements, save_baseline, table
from .runner import Case, Report, compare, evaluate, load_cases, run

__all__ = [
    "Case",
    "Report",
    "check_regression",
    "compare",
    "disagreements",
    "evaluate",
    "first_hit_rank",
    "fraction_recalled_at_k",
    "load_cases",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "run",
    "save_baseline",
    "table",
]
__version__ = "0.1.0"
