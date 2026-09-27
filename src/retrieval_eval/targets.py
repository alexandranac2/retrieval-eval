"""Turn a target spec into a retriever.

A spec is either
  - an HTTP(S) URL: POST {"query": "..."} → a JSON list of ids or {"ids": [...]}
  - "package.module:function": any importable callable, sync or async,
    taking a query string and returning ranked ids.
"""
from __future__ import annotations

import importlib
import json
import sys
import urllib.request
from pathlib import Path

from .runner import Retriever


def http_retriever(url: str, timeout: float = 30.0) -> Retriever:
    def retrieve(q: str):
        req = urllib.request.Request(
            url, data=json.dumps({"query": q}).encode(), headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as res:  # noqa: S310 — user-supplied eval target
            body = json.loads(res.read())
        return body["ids"] if isinstance(body, dict) else body

    return retrieve


def python_retriever(spec: str) -> Retriever:
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError(f"expected 'module:function', got {spec!r}")
    # Let specs like "examples.ablation:keyword" resolve from the working directory.
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    obj = importlib.import_module(module_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise ValueError(f"{spec!r} is not callable")
    return obj


def resolve(spec: str) -> Retriever:
    spec = spec.strip()
    if spec.startswith(("http://", "https://")):
        return http_retriever(spec)
    return python_retriever(spec)


def parse_named(spec: str) -> tuple[str, str]:
    """'NAME=SPEC' → (name, spec). A bare spec is named after its last segment."""
    name, sep, target = spec.partition("=")
    if sep and target and "://" not in name:
        return name.strip(), target.strip()
    target = spec.strip()
    tail = target.rstrip("/").split(":")[-1].split("/")[-1] or target
    return tail, target
