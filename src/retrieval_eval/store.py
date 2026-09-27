"""A workspace folder that remembers runs, targets and the baseline.

    .retrieval-eval/
      runs/<id>.json      one file per run: every configuration's full report
      state.json          saved targets + which run/config is the baseline
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


class Workspace:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"

    # --- state -------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"targets": [], "baseline": None}

    def save_state(self, state: dict[str, Any]) -> None:
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def set_targets(self, targets: list[dict[str, str]]) -> None:
        st = self.state()
        st["targets"] = [{"name": t["name"], "spec": t["spec"]} for t in targets if t.get("name") and t.get("spec")]
        self.save_state(st)

    def set_baseline(self, run_id: str | None, config: str | None) -> None:
        st = self.state()
        st["baseline"] = {"run": run_id, "config": config} if run_id and config else None
        self.save_state(st)

    # --- runs --------------------------------------------------------------
    def add_run(self, label: str, targets: list[dict[str, str]], ks: list[int], reports: list[dict]) -> dict[str, Any]:
        run = {
            "id": time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4],
            "label": label or "",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ks": ks,
            "targets": targets,
            "reports": reports,
        }
        (self.runs_dir / f"{run['id']}.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
        return run

    def import_run(self, run: dict[str, Any]) -> dict[str, Any]:
        if "id" not in run or not isinstance(run.get("reports"), list):
            raise ValueError("not a run file: needs 'id' and 'reports'")
        (self.runs_dir / f"{run['id']}.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
        return run

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        path = self.runs_dir / f"{Path(run_id).name}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def delete_run(self, run_id: str) -> bool:
        path = self.runs_dir / f"{Path(run_id).name}.json"
        if not path.exists():
            return False
        path.unlink()
        st = self.state()
        if st.get("baseline") and st["baseline"].get("run") == run_id:
            st["baseline"] = None
            self.save_state(st)
        return True

    def list_runs(self) -> list[dict[str, Any]]:
        out = []
        for path in sorted(self.runs_dir.glob("*.json"), reverse=True):
            run = json.loads(path.read_text(encoding="utf-8"))
            out.append(
                {
                    "id": run["id"],
                    "label": run.get("label", ""),
                    "created_at": run.get("created_at", ""),
                    "configs": [{"name": r["name"], "mrr": r["mrr"], "n": r["n"]} for r in run["reports"]],
                }
            )
        return out
