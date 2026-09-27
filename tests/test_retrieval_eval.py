import json
import math

import pytest

from retrieval_eval import (
    Case,
    check_regression,
    disagreements,
    fraction_recalled_at_k,
    load_cases,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    run,
    save_baseline,
    table,
)
from retrieval_eval.cli import main
from retrieval_eval.metrics import first_hit_rank, percentile


def test_first_hit_and_reciprocal_rank():
    assert first_hit_rank([5, 3, 9], {9}) == 3
    assert first_hit_rank([5, 3], {9}) is None
    assert reciprocal_rank([5, 3, 9], {9}) == pytest.approx(1 / 3)
    assert reciprocal_rank([], {1}) == 0.0


def test_recall_is_hit_rate_and_fraction_is_share():
    assert recall_at_k([1, 2, 3], {3}, 2) == 0.0
    assert recall_at_k([1, 2, 3], {3}, 3) == 1.0
    assert fraction_recalled_at_k([1, 2, 3], {1, 3, 7}, 3) == pytest.approx(2 / 3)
    assert fraction_recalled_at_k([1], set(), 3) == 0.0


def test_ndcg_rewards_rank_position():
    assert ndcg_at_k([1, 2], {1}, 5) == 1.0
    assert ndcg_at_k([2, 1], {1}, 5) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k([2, 3], {1}, 5) == 0.0


def test_percentile_nearest_rank():
    assert percentile([], 95) == 0.0
    assert percentile([1, 2, 3, 4, 100], 95) == 100
    assert percentile([1, 2, 3, 4], 50) == 2


def test_load_cases_skips_comments_and_validates(tmp_path):
    f = tmp_path / "cases.jsonl"
    f.write_text('# comment\n{"q": "a", "relevant": [1], "tags": ["t"]}\n\n{"q": "b", "relevant": ["x"]}\n')
    cases = load_cases(f)
    assert [c.q for c in cases] == ["a", "b"]
    assert cases[0].relevant == frozenset({1}) and cases[0].tags == ("t",)
    f.write_text('{"q": "a", "relevant": []}\n')
    with pytest.raises(ValueError):
        load_cases(f)


CASES = [Case("alpha", frozenset({1})), Case("beta", frozenset({2})), Case("gamma", frozenset({3}))]


def perfect(q):
    return [{"alpha": 1, "beta": 2, "gamma": 3}[q], 99]


async def async_off_by_one(q):
    # Right answer at rank 2 for alpha, missing for gamma.
    return {"alpha": [9, 1], "beta": [2], "gamma": [8, 7]}[q]


def test_run_compares_sync_and_async_retrievers():
    good, bad = run({"good": perfect, "bad": async_off_by_one}, CASES, ks=(1, 3))
    assert good.recall == {"@1": 1.0, "@3": 1.0} and good.mrr == 1.0
    assert bad.recall["@1"] == pytest.approx(1 / 3)
    assert bad.recall["@3"] == pytest.approx(2 / 3)
    assert bad.mrr == pytest.approx((0.5 + 1 + 0) / 3)
    assert [c.rank for c in bad.cases] == [2, 1, None]
    assert "| good |" in table([good, bad])
    diff = disagreements([good, bad])
    assert "alpha" in diff and "gamma" in diff and "beta" not in diff


def test_regression_gate(tmp_path):
    good, bad = run({"good": perfect, "bad": async_off_by_one}, CASES, ks=(1, 3))
    path = tmp_path / "baseline.json"
    save_baseline(good, path)
    assert check_regression(good, path) == []
    found = {r.metric for r in check_regression(bad, path)}
    assert {"mrr", "recall@1", "recall@3"} <= found
    # A generous tolerance lets small drops through.
    assert check_regression(bad, path, tolerance=1.0) == []


def test_cli_against_http_endpoint(tmp_path, monkeypatch, capsys):
    cases = tmp_path / "cases.jsonl"
    cases.write_text("\n".join(json.dumps({"q": c.q, "relevant": sorted(c.relevant)}) for c in CASES))
    monkeypatch.setattr("retrieval_eval.cli.resolve", lambda spec: perfect)
    baseline = tmp_path / "b.json"
    assert main([str(cases), "--endpoint", "a=http://x", "--save-baseline", str(baseline)]) == 0
    assert main([str(cases), "--endpoint", "a=http://x", "--baseline", str(baseline)]) == 0
    monkeypatch.setattr("retrieval_eval.cli.resolve", lambda spec: async_off_by_one)
    assert main([str(cases), "--endpoint", "a=http://x", "--baseline", str(baseline)]) == 1
    assert "REGRESSION" in capsys.readouterr().out
