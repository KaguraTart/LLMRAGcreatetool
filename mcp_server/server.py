"""MCP server exposing llmragd operations as tools."""

from __future__ import annotations

import os
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

from .client import LLMRAGDaemonClient

_DAEMON_URL = os.environ.get("LLMRAG_DAEMON_URL", "http://127.0.0.1:7474")
_DAEMON_TIMEOUT = int(os.environ.get("LLMRAG_DAEMON_TIMEOUT", "30"))

mcp = FastMCP(
    "llmrag",
    instructions=(
        "Use these tools to manage LLMRAG workspaces and execute indexing/query/answer "
        "operations through the local llmrag daemon."
    ),
)
_client = LLMRAGDaemonClient(base_url=_DAEMON_URL, timeout=_DAEMON_TIMEOUT)


def _run_safe(fn):
    try:
        return fn()
    except requests.HTTPError as exc:
        response = getattr(exc, "response", None)
        detail = ""
        if response is not None:
            try:
                detail = f" | body={response.text}"
            except Exception:
                detail = ""
        raise RuntimeError(f"Daemon request failed: {exc}{detail}") from exc
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc


@mcp.tool(name="daemon_health")
def daemon_health() -> dict[str, Any]:
    """Check whether llmragd is reachable."""

    return _run_safe(_client.health)


@mcp.tool(name="daemon_status")
def daemon_status() -> dict[str, Any]:
    """Return daemon status and workspace counts."""

    return _run_safe(_client.status)


@mcp.tool(name="workspace_list")
def workspace_list() -> list[dict[str, Any]]:
    """List all available workspaces."""

    return _run_safe(_client.workspace_list)


@mcp.tool(name="workspace_create")
def workspace_create(name: str, config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a workspace."""

    return _run_safe(lambda: _client.workspace_create(name=name, config_overrides=config_overrides))


@mcp.tool(name="workspace_delete")
def workspace_delete(name: str) -> dict[str, Any]:
    """Delete a workspace and persisted data."""

    return _run_safe(lambda: _client.workspace_delete(name=name))


@mcp.tool(name="index_path")
def index_path(
    workspace: str,
    path: str,
    wait: bool = True,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Start indexing and optionally wait for completion."""

    def _run() -> dict[str, Any]:
        job_id = _client.index_start(workspace=workspace, path=path)
        if not wait:
            return {"job_id": job_id, "status": "pending"}
        job = _client.index_wait(
            workspace=workspace,
            job_id=job_id,
            timeout_seconds=timeout_seconds,
        )
        return {"job_id": job_id, "job": job}

    return _run_safe(_run)


@mcp.tool(name="query_workspace")
def query_workspace(workspace: str, text: str, top_k: int = 5) -> dict[str, Any]:
    """Run retrieval-only query."""

    return _run_safe(lambda: _client.query(workspace=workspace, text=text, top_k=top_k))


@mcp.tool(name="answer_workspace")
def answer_workspace(workspace: str, text: str) -> dict[str, Any]:
    """Run full RAG answer generation."""

    return _run_safe(lambda: _client.answer(workspace=workspace, text=text))

