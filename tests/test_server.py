import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from retrieval_eval.server import App, make_handler, static_report_html
from retrieval_eval.store import Workspace

# Module-level retrievers so they're importable as "test_server:perfect".
ANSWERS = {"alpha": 1, "beta": 2}


def perfect(q):
    return [ANSWERS[q], 99]


async def wrong(q):
    return [99, 98]


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path.parent))
    monkeypatch.chdir(__import__("pathlib").Path(__file__).parent)
    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"q": "alpha", "relevant": [1], "tags": ["t"]}\n{"q": "beta", "relevant": [2]}\n')
    ws = Workspace(tmp_path / "ws")
    ws.set_targets([{"name": "good", "spec": "test_server:perfect"}, {"name": "bad", "spec": "test_server:wrong"}])
    app = App(cases, ws, [1, 3])
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base, app
    httpd.shutdown()


def call(base, method, path, body=None, host=None):
    req = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body is not None else None)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    if host:
        req.add_header("Host", host)
    try:
        with urllib.request.urlopen(req) as res:
            ctype = res.headers.get("Content-Type", "")
            raw = res.read()
            return res.status, (json.loads(raw) if "json" in ctype else raw.decode())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_ui_page_and_state(server):
    base, _ = server
    status, html = call(base, "GET", "/")
    assert status == 200 and "retrieval-eval" in html
    status, st = call(base, "GET", "/api/state")
    assert st["cases"][0] == {"q": "alpha", "relevant": [1], "tags": ["t"]}
    assert [t["name"] for t in st["targets"]] == ["good", "bad"]


def test_run_baseline_and_delete(server):
    base, _ = server
    st = call(base, "GET", "/api/state")[1]
    status, run = call(base, "POST", "/api/run", {"targets": st["targets"], "ks": [1, 3], "label": "first"})
    assert status == 200
    good, bad = run["reports"]
    assert good["mrr"] == 1.0 and bad["mrr"] == 0.0
    assert good["cases"][0]["ranked"] == [1, 99] and good["cases"][0]["relevant"] == [1]
    call(base, "POST", "/api/baseline", {"run": run["id"], "config": "good"})
    st = call(base, "GET", "/api/state")[1]
    assert st["baseline"] == {"run": run["id"], "config": "good"}
    assert st["runs"][0]["label"] == "first"
    status, exported = call(base, "GET", f"/export/{run['id']}")
    assert status == 200 and "window.__STATIC__" in exported
    call(base, "DELETE", f"/api/runs/{run['id']}")
    st = call(base, "GET", "/api/state")[1]
    assert st["runs"] == [] and st["baseline"] is None


def test_bad_target_reports_a_readable_error(server):
    base, _ = server
    status, err = call(base, "POST", "/api/run", {"targets": [{"name": "x", "spec": "no_such_module:fn"}]})
    assert status == 400 and "no_such_module" in err["error"]


def test_save_cases_validates_and_keeps_id_types(server):
    base, app = server
    status, err = call(base, "PUT", "/api/cases", {"cases": [{"q": "", "relevant": [1]}]})
    assert status == 400 and "row 1" in err["error"]
    status, st = call(base, "PUT", "/api/cases", {"cases": [{"q": "gamma", "relevant": [3, "doc-x"], "tags": ["new"]}]})
    assert status == 200
    assert json.loads(app.cases_path.read_text().splitlines()[0]) == {"q": "gamma", "relevant": [3, "doc-x"], "tags": ["new"]}


def test_rejects_non_local_host(server):
    base, _ = server
    status, err = call(base, "GET", "/api/state", host="evil.example")
    assert status == 403


def test_static_report_escapes_script_close():
    run = {"id": "r", "reports": [], "label": "</script><b>x"}
    html = static_report_html(run)
    assert "</script><b>x" not in html
