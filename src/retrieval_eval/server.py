"""Local web UI: `retrieval-eval ui cases.jsonl`.

Standard library only. Binds to 127.0.0.1 and rejects requests whose Host
header isn't local, because targets can be Python import specs and a page on
another site must not be able to trigger runs (DNS rebinding).
"""
from __future__ import annotations

import asyncio
import json
import threading
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .runner import Case, compare, load_cases, save_cases
from .store import Workspace
from .targets import resolve

LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


def ui_html() -> str:
    return resources.files("retrieval_eval").joinpath("ui/index.html").read_text(encoding="utf-8")


def static_report_html(run: dict[str, Any], cases_path: str = "") -> str:
    """The same UI, read-only, with one run baked in — a shareable single file."""
    payload = json.dumps({"run": run, "cases_path": cases_path}).replace("</", "<\\/")
    return ui_html().replace("<!--STATIC_RUN-->", f"<script>window.__STATIC__ = {payload};</script>")


class App:
    def __init__(self, cases_path: str | Path, workspace: Workspace, default_ks: list[int]):
        self.cases_path = Path(cases_path)
        self.ws = workspace
        self.default_ks = default_ks
        self.run_lock = threading.Lock()
        # One long-lived event loop for every run. Async retrievers often cache
        # clients (DB pools, HTTP sessions) bound to the loop that created them;
        # a fresh asyncio.run() per run would break them on the second run.
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, name="retrieval-eval-runs", daemon=True).start()

    def cases_json(self) -> list[dict[str, Any]]:
        if not self.cases_path.exists():
            return []
        return [
            {"q": c.q, "relevant": sorted(c.relevant, key=str), "tags": list(c.tags)}
            for c in load_cases(self.cases_path)
        ]

    def state(self) -> dict[str, Any]:
        st = self.ws.state()
        try:
            cases, cases_error = self.cases_json(), None
        except (ValueError, json.JSONDecodeError) as exc:
            cases, cases_error = [], str(exc)
        return {
            "cases_path": str(self.cases_path),
            "cases": cases,
            "cases_error": cases_error,
            "targets": st.get("targets", []),
            "baseline": st.get("baseline"),
            "runs": self.ws.list_runs(),
            "default_ks": self.default_ks,
        }

    def save_cases(self, rows: list[dict[str, Any]]) -> None:
        cases = []
        for i, row in enumerate(rows, 1):
            q = str(row.get("q", "")).strip()
            relevant = row.get("relevant") or []
            if not q or not relevant:
                raise ValueError(f"row {i}: a question and at least one correct id are required")
            cases.append(Case(q=q, relevant=frozenset(relevant), tags=tuple(t for t in row.get("tags", []) if t)))
        save_cases(self.cases_path, cases)

    def run(self, targets: list[dict[str, str]], ks: list[int], label: str) -> dict[str, Any]:
        if not targets:
            raise ValueError("pick at least one target to run")
        ks = sorted({int(k) for k in ks if int(k) > 0}) or self.default_ks
        retrievers = {}
        for t in targets:
            try:
                retrievers[t["name"]] = resolve(t["spec"])
            except Exception as exc:  # noqa: BLE001 — surface any import/URL problem to the UI
                raise ValueError(f"target {t['name']!r} ({t['spec']}): {exc}") from exc
        cases = load_cases(self.cases_path)
        if not cases:
            raise ValueError("there are no labeled cases to run yet")
        with self.run_lock:
            reports = asyncio.run_coroutine_threadsafe(compare(retrievers, cases, ks), self.loop).result()
        return self.ws.add_run(label, targets, ks, [r.to_dict() for r in reports])


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "retrieval-eval"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
            pass

        # --- plumbing ------------------------------------------------------
        def _local(self) -> bool:
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
            return host in LOCAL_HOSTS

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: Any, status: int = 200) -> None:
            self._send(status, json.dumps(data).encode(), "application/json")

        def _error(self, message: str, status: int = 400) -> None:
            self._json({"error": message}, status)

        def _body(self) -> Any:
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length) or b"{}")

        def _route(self, method: str) -> None:
            if not self._local():
                return self._error("only local requests are accepted", HTTPStatus.FORBIDDEN)
            path = urlparse(self.path).path.rstrip("/") or "/"
            try:
                if method == "GET" and path == "/":
                    return self._send(200, ui_html().encode(), "text/html; charset=utf-8")
                if method == "GET" and path == "/api/state":
                    return self._json(app.state())
                if method == "PUT" and path == "/api/cases":
                    app.save_cases(self._body().get("cases", []))
                    return self._json(app.state())
                if method == "PUT" and path == "/api/targets":
                    app.ws.set_targets(self._body().get("targets", []))
                    return self._json(app.state())
                if method == "POST" and path == "/api/run":
                    body = self._body()
                    run_ = app.run(body.get("targets", []), body.get("ks", app.default_ks), body.get("label", ""))
                    return self._json(run_)
                if method == "POST" and path == "/api/baseline":
                    body = self._body()
                    app.ws.set_baseline(body.get("run"), body.get("config"))
                    return self._json(app.state())
                if method == "POST" and path == "/api/import":
                    return self._json(app.ws.import_run(self._body()))
                if path.startswith("/api/runs/"):
                    run_id = path.split("/")[-1]
                    if method == "GET":
                        found = app.ws.get_run(run_id)
                        return self._json(found) if found else self._error("no run with that id", 404)
                    if method == "DELETE":
                        return self._json({"deleted": app.ws.delete_run(run_id)})
                if method == "GET" and path.startswith("/export/"):
                    found = app.ws.get_run(path.split("/")[-1])
                    if not found:
                        return self._error("no run with that id", 404)
                    html = static_report_html(found, str(app.cases_path)).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Disposition", f'attachment; filename="retrieval-eval-{found["id"]}.html"')
                    self.send_header("Content-Length", str(len(html)))
                    self.end_headers()
                    self.wfile.write(html)
                    return None
                return self._error("not found", 404)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                return self._error(str(exc))
            except Exception as exc:  # noqa: BLE001 — a crashing retriever shouldn't kill the server
                traceback.print_exc()
                return self._error(f"{type(exc).__name__}: {exc}", 500)

        def do_GET(self) -> None:
            self._route("GET")

        def do_POST(self) -> None:
            self._route("POST")

        def do_PUT(self) -> None:
            self._route("PUT")

        def do_DELETE(self) -> None:
            self._route("DELETE")

    return Handler


def serve(
    cases_path: str | Path,
    workspace: str | Path = ".retrieval-eval",
    port: int = 8765,
    targets: list[dict[str, str]] | None = None,
    ks: list[int] | None = None,
    open_browser: bool = True,
) -> None:
    ws = Workspace(workspace)
    if targets:
        known = {t["name"]: t for t in ws.state().get("targets", [])}
        known.update({t["name"]: t for t in targets})
        ws.set_targets(list(known.values()))
    app = App(cases_path, ws, ks or [1, 3, 5])
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"retrieval-eval UI on {url}  (cases: {cases_path}, workspace: {workspace})  Ctrl+C to stop")
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
