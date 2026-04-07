"""HTTP client wrapper for llmragd daemon APIs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class LLMRAGDaemonClient:
    base_url: str = "http://127.0.0.1:7474"
    timeout: int = 30

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    def health(self) -> dict[str, Any]:
        response = requests.get(self._url("/health"), timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def status(self) -> dict[str, Any]:
        response = requests.get(self._url("/api/v1/status"), timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def workspace_list(self) -> list[dict[str, Any]]:
        response = requests.get(self._url("/api/v1/workspaces"), timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else []

    def workspace_create(self, name: str, config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {"name": name, "config_overrides": config_overrides or {}}
        response = requests.post(self._url("/api/v1/workspaces"), json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def workspace_delete(self, name: str) -> dict[str, Any]:
        response = requests.delete(self._url(f"/api/v1/workspaces/{name}"), timeout=self.timeout)
        if response.status_code == 204:
            return {"deleted": True, "workspace": name}
        response.raise_for_status()
        return {"deleted": False, "workspace": name}

    def index_start(self, workspace: str, path: str) -> str:
        payload = {"path": path}
        response = requests.post(
            self._url(f"/api/v1/workspaces/{workspace}/index"),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["job_id"])

    def index_job(self, workspace: str, job_id: str) -> dict[str, Any]:
        response = requests.get(
            self._url(f"/api/v1/workspaces/{workspace}/jobs/{job_id}"),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def index_wait(
        self,
        workspace: str,
        job_id: str,
        timeout_seconds: int = 1800,
        poll_interval: float = 1.0,
    ) -> dict[str, Any]:
        start = time.monotonic()
        while True:
            job = self.index_job(workspace, job_id)
            if job.get("status") in {"done", "failed"}:
                return job
            if time.monotonic() - start > timeout_seconds:
                raise TimeoutError(f"Index job timed out after {timeout_seconds}s: {job_id}")
            time.sleep(poll_interval)

    def query(self, workspace: str, text: str, top_k: int = 5) -> dict[str, Any]:
        payload = {"text": text, "top_k": top_k}
        response = requests.post(
            self._url(f"/api/v1/workspaces/{workspace}/query"),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def answer(self, workspace: str, text: str) -> dict[str, Any]:
        payload = {"text": text}
        response = requests.post(
            self._url(f"/api/v1/workspaces/{workspace}/answer"),
            json=payload,
            timeout=max(self.timeout, 120),
        )
        response.raise_for_status()
        return response.json()

