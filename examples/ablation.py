"""Ablation with in-process retrievers: the pattern behind the Alex OS report.

Swap the toy retrievers for your own (they just take a query string and return
ranked ids, sync or async), point `load_cases` at your labeled JSONL, run:

    python examples/ablation.py
"""
from pathlib import Path

from retrieval_eval import disagreements, load_cases, run, table

DOCS = {
    1: "alex os reasoning service retrieval assistant",
    2: "theme dna design tokens templates",
    3: "wordpress platform engineering plugins themes",
}


def keyword(q: str):
    words = set(q.lower().split())
    scored = sorted(DOCS, key=lambda d: -len(words & set(DOCS[d].split())))
    return scored


def first_doc_always(q: str):
    return [1, 2, 3]


if __name__ == "__main__":
    cases = load_cases(Path(__file__).with_name("cases.jsonl"))
    reports = run({"keyword": keyword, "constant": first_doc_always}, cases, ks=(1, 3))
    print(table(reports))
    print()
    print(disagreements(reports))
