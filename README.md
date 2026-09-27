# retrieval-eval

Small, honest retrieval evaluation for RAG systems. Give it hand-labeled
queries and one or more retrievers; get recall@k, MRR, nDCG, latency, the
queries where configurations disagree, and a regression gate for CI.

No dependencies. No LLM judge. Nothing to configure beyond "here are my
questions, here is the right answer to each".

## Why it exists

I built a hybrid retriever (semantic search fused with keyword search, plus a
reranker) and shipped it on by default without ever measuring whether it
helped. When I finally ran an ablation on 31 hand-labeled questions, the
default was the weakest of five configurations:

| config | R@1 | R@3 | MRR | nDCG@5 |
|---|---|---|---|---|
| keyword | 87% | 97% | 0.922 | 0.942 |
| semantic | 90% | 97% | 0.925 | 0.935 |
| hybrid (RRF) | 87% | 94% | 0.898 | 0.907 |
| semantic + rerank | 90% | 94% | 0.927 | 0.937 |
| hybrid + rerank (shipped default) | 84% | 94% | 0.882 | 0.896 |

The full write-up, including why these numbers are not enough to crown a
winner, is at <https://alexandranace.com/alex-os-evaluation>. This package is
that harness pulled out of the project so it can be pointed at any retriever.

## Install

```bash
pip install -e .            # from a checkout
```

## Labeled cases

JSONL, one question per line. `relevant` holds the id(s) that are genuinely
the right answer, checked against your corpus rather than guessed. `tags` are
optional and carried into the results so you can slice by query type.

```json
{"q": "insurance policy renewal assistant", "relevant": [79], "tags": ["title"]}
{"q": "is pgvector required for this system to work?", "relevant": [98], "tags": ["behaviour"]}
```

## The UI

```bash
retrieval-eval ui cases.jsonl \
  --target semantic=mypkg.search:semantic \
  --target hybrid=http://localhost:8000/retrieve?mode=hybrid
```

Opens a local page (127.0.0.1 only, standard library, no build step) where you can:

- **label questions**: edit the question, the correct ids and tags, then save back to the JSONL
- **run** any mix of targets and keep every run in `.retrieval-eval/`
- **compare** configurations in a scoreboard, with the best value per metric marked
- **set a baseline** and see every later run as a delta against it
- **read the rank grid**: every question against every configuration, showing where the right answer landed, filtered to the questions where configurations disagree
- **open a question** to see exactly what each configuration returned, with the right answer highlighted
- **download a report**: the same view as one self-contained HTML file you can send to someone

Retrievers can return bare ids or `{"id": ..., "title": ...}` items. Titles only
make the drilldown readable; scoring uses ids.

Add `--html report.html` to a terminal run to write the same static report without opening the UI.

## Python API: ablations over in-process retrievers

A retriever is any callable, sync or async, that takes a query string and
returns ranked ids.

```python
from retrieval_eval import load_cases, run, table, disagreements

cases = load_cases("cases.jsonl")
reports = run({"semantic": semantic_search, "hybrid": hybrid_search}, cases, ks=(1, 3, 5))
print(table(reports))
print(disagreements(reports))   # the queries where the configurations rank differently
```

A warm-up pass runs before timing, so cached embeddings don't flatter whichever
configuration happens to run second.

## CLI: any retriever that speaks HTTP

Each endpoint receives `POST {"query": "..."}` and returns a JSON list of ids
(or `{"ids": [...]}`).

```bash
retrieval-eval cases.jsonl \
  --endpoint semantic=http://localhost:8000/retrieve?mode=semantic \
  --endpoint hybrid=http://localhost:8000/retrieve?mode=hybrid
```

## Regression gate

Save a baseline once, then fail CI when a change makes retrieval worse:

```bash
retrieval-eval cases.jsonl --endpoint live=http://localhost:8000/retrieve --save-baseline baseline.json
retrieval-eval cases.jsonl --endpoint live=http://localhost:8000/retrieve --baseline baseline.json --tolerance 0.02
```

Exit code 1 means recall@k, MRR or nDCG dropped more than the tolerance.
Latency is reported but not gated, because it depends on the machine.

## What it doesn't do

- **Graded relevance.** Relevance is binary. For "partly right" answers you
  need a graded set and a different nDCG.
- **Answer quality.** This scores retrieval only. A right document can still
  produce a wrong answer; test that separately.
- **Tell you your set is big enough.** With 30 questions, one question is
  3 percentage points. Look at the disagreement table before believing a
  one-point difference.

## Metrics

- **recall@k**: share of queries with at least one relevant id in the top k
  (hit rate). `fraction_recalled_at_k` is the many-answers variant.
- **MRR**: mean of 1 / rank of the first relevant id (0 when missing).
- **nDCG@k**: binary-relevance discounted gain, normalised by the ideal ranking.
- **p50 / p95**: nearest-rank percentiles of per-query retrieval time.

## License

MIT
