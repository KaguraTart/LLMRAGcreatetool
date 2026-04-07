"""OpenClaw plugin adapter with daemon/MCP request mapping."""

from __future__ import annotations

import os
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Any

from mcp_server.client import LLMRAGDaemonClient

from .base import AdapterAuthError, AdapterConfigError, AdapterRequestError


@dataclass
class OpenClawAdapterConfig:
    daemon_url: str = "http://127.0.0.1:7474"
    timeout: int = 30
    auth_token: str = ""
    auth_token_env: str = "LLMRAG_OPENCLAW_TOKEN"

    def resolved_token(self) -> str:
        return os.environ.get(self.auth_token_env, self.auth_token or "")


class OpenClawAdapter:
    """Bridges OpenClaw tool requests to llmrag daemon APIs."""

    def __init__(self, config: OpenClawAdapterConfig | None = None):
        self.config = config or OpenClawAdapterConfig()
        if self.config.timeout <= 0:
            raise AdapterConfigError("timeout must be > 0")
        parsed = urlparse(self.config.daemon_url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme == "http" and host not in {"127.0.0.1", "localhost", "::1"}:
            raise AdapterConfigError("Non-local daemon_url must use https")
        self._client = LLMRAGDaemonClient(base_url=self.config.daemon_url, timeout=self.config.timeout)

    @staticmethod
    def _normalize_tool_name(name: str) -> str:
        return (name or "").strip().replace("-", "_")

    def authenticate(self, auth_header: str | None = None) -> bool:
        token = self.config.resolved_token()
        if not token:
            return True
        if not auth_header:
            raise AdapterAuthError("Missing Authorization header")
        expected = f"Bearer {token}"
        if auth_header.strip() != expected:
            raise AdapterAuthError("Invalid Authorization token")
        return True

    def describe(self) -> dict[str, Any]:
        return {
            "name": "llmrag-openclaw-adapter",
            "transport": "http",
            "daemon_url": self.config.daemon_url,
            "tools": sorted(list(self._tool_map().keys())),
        }

    def _tool_map(self):
        return {
            "daemon_health": lambda args: self._client.health(),
            "daemon_status": lambda args: self._client.status(),
            "workspace_list": lambda args: self._client.workspace_list(),
            "workspace_create": lambda args: self._client.workspace_create(
                name=str(args.get("name", "")),
                config_overrides=args.get("config_overrides") or {},
            ),
            "workspace_delete": lambda args: self._client.workspace_delete(name=str(args.get("name", ""))),
            "index_path": self._index_path,
            "query_workspace": lambda args: self._client.query(
                workspace=str(args.get("workspace", "")),
                text=str(args.get("text", "")),
                top_k=int(args.get("top_k", 5)),
            ),
            "answer_workspace": lambda args: self._client.answer(
                workspace=str(args.get("workspace", "")),
                text=str(args.get("text", "")),
            ),
        }

    def _index_path(self, args: dict[str, Any]) -> dict[str, Any]:
        workspace = str(args.get("workspace", ""))
        path = str(args.get("path", ""))
        wait = bool(args.get("wait", True))
        timeout_seconds = int(args.get("timeout_seconds", 1800))
        job_id = self._client.index_start(workspace=workspace, path=path)
        if not wait:
            return {"job_id": job_id, "status": "pending"}
        job = self._client.index_wait(workspace=workspace, job_id=job_id, timeout_seconds=timeout_seconds)
        return {"job_id": job_id, "job": job}

    def handle_request(self, request: dict[str, Any], auth_header: str | None = None) -> dict[str, Any]:
        self.authenticate(auth_header=auth_header)

        tool = self._normalize_tool_name(str(request.get("tool", "")))
        if not tool:
            raise AdapterRequestError("Request must include non-empty 'tool'")

        arguments = request.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise AdapterRequestError("'arguments' must be an object")

        tool_map = self._tool_map()
        if tool not in tool_map:
            raise AdapterRequestError(f"Unsupported tool: {tool}")

        try:
            result = tool_map[tool](arguments)
        except Exception as exc:
            raise AdapterRequestError(f"Tool call failed ({tool}): {exc}") from exc

        return {"ok": True, "tool": tool, "result": result}
