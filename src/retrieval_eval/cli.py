"""retrieval-eval: score retrievers against hand-labeled queries.

Run from the terminal:
    retrieval-eval cases.jsonl --target semantic=http://localhost:8000/retrieve?mode=semantic \\
                               --target hybrid=mypkg.search:hybrid

Or open the local UI (label cases, run, compare, set a baseline):
    retrieval-eval ui cases.jsonl --target semantic=mypkg.search:semantic

A target is NAME=SPEC, where SPEC is an HTTP(S) URL (POST {"query": ...} →
JSON list of ids, or {"ids": [...]}) or an importable "module:function" that
takes a query string and returns ranked ids (sync or async).
"""
from __future__ import annotations

import argparse
import json
import sys

from . import report as rpt
from .runner import load_cases, run
from .targets import parse_named, resolve


def _targets(specs: list[str] | None) -> list[dict[str, str]]:
    out = []
    for spec in specs or []:
        name, target = parse_named(spec)
        out.append({"name": name, "spec": target})
    return out


def _ui(argv: list[str]) -> int:
    from .server import serve

    ap = argparse.ArgumentParser(prog="retrieval-eval ui", description="Open the local evaluation UI.")
    ap.add_argument("cases", help="JSONL file of labeled cases (created if missing)")
    ap.add_argument("--target", "--endpoint", action="append", metavar="NAME=SPEC", help="retriever to offer in the UI (repeatable)")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--workspace", default=".retrieval-eval", help="folder for run history and the baseline")
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = ap.parse_args(argv)
    serve(args.cases, args.workspace, args.port, _targets(args.target), args.k, open_browser=not args.no_browser)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "ui":
        return _ui(argv[1:])

    ap = argparse.ArgumentParser(prog="retrieval-eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", help="JSONL file of labeled cases")
    ap.add_argument("--target", "--endpoint", dest="target", action="append", required=True, metavar="NAME=SPEC", help="retriever to score (repeatable)")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--json", metavar="PATH", help="write the full reports as JSON")
    ap.add_argument("--html", metavar="PATH", help="write a self-contained HTML report (the UI, read-only)")
    ap.add_argument("--save-baseline", metavar="PATH", help="save the first target's report as the baseline")
    ap.add_argument("--baseline", metavar="PATH", help="fail (exit 1) if the first target regressed against this")
    ap.add_argument("--tolerance", type=float, default=0.0, help="allowed drop before it counts as a regression")
    args = ap.parse_args(argv)

    targets = _targets(args.target)
    retrievers = {t["name"]: resolve(t["spec"]) for t in targets}
    cases = load_cases(args.cases)
    reports = run(retrievers, cases, args.k)
    print(f"{len(cases)} labeled queries\n")
    print(rpt.table(reports))
    if len(reports) > 1:
        print("\nWhere configurations disagree (rank of first relevant hit):\n")
        print(rpt.disagreements(reports))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump([r.to_dict() for r in reports], fh, indent=2)
    if args.html:
        import time

        from .server import static_report_html

        run_ = {
            "id": time.strftime("%Y%m%d-%H%M%S"),
            "label": "",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ks": args.k,
            "targets": targets,
            "reports": [r.to_dict() for r in reports],
        }
        with open(args.html, "w", encoding="utf-8") as fh:
            fh.write(static_report_html(run_, args.cases))
        print(f"\nHTML report written to {args.html}")
    if args.save_baseline:
        rpt.save_baseline(reports[0], args.save_baseline)
        print(f"\nBaseline saved to {args.save_baseline}")
    if args.baseline:
        regressions = rpt.check_regression(reports[0], args.baseline, args.tolerance)
        if regressions:
            print("\nREGRESSION against baseline:")
            for r in regressions:
                print(f"  {r}")
            return 1
        print("\nNo regression against baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
