"""Claude Code subprocess adapter (JSONL over stdio)."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .base import AdapterConfigError, AdapterRequestError


@dataclass
class ClaudeCodeAdapterConfig:
    command: list[str] = field(default_factory=lambda: ["python", "-m", "mcp_server"])
    startup_timeout: float = 10.0
    request_timeout: float = 30.0
    max_retries: int = 2


class ClaudeCodeAdapter:
    """Maintains a long-lived subprocess and exchanges JSON lines."""

    def __init__(self, config: ClaudeCodeAdapterConfig | None = None):
        self.config = config or ClaudeCodeAdapterConfig()
        if not self.config.command:
            raise AdapterConfigError("command cannot be empty")
        if self.config.request_timeout <= 0:
            raise AdapterConfigError("request_timeout must be > 0")
        self._proc: subprocess.Popen | None = None
        self._stdout_queue: queue.Queue[str] = queue.Queue()
        self._stdout_thread: threading.Thread | None = None

    def start(self):
        if self.is_running:
            return

        self._proc = subprocess.Popen(
            self.config.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        def _reader():
            assert self._proc is not None and self._proc.stdout is not None
            for line in self._proc.stdout:
                self._stdout_queue.put(line.rstrip("\n"))

        self._stdout_thread = threading.Thread(target=_reader, daemon=True)
        self._stdout_thread.start()

        deadline = time.monotonic() + self.config.startup_timeout
        while time.monotonic() < deadline:
            if self.is_running:
                return
            time.sleep(0.05)

        raise AdapterRequestError(
            f"subprocess did not start in time (command={self.config.command}, timeout={self.config.startup_timeout}s)"
        )

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self):
        if self._proc is None:
            return
        proc = self._proc
        if self.is_running:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        if proc.stdin:
            proc.stdin.close()
        if proc.stdout:
            proc.stdout.close()
        if proc.stderr:
            proc.stderr.close()
        self._proc = None

    def restart(self):
        self.stop()
        self.start()

    def _write_request(self, payload: dict[str, Any]):
        if not self.is_running:
            self.start()
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

    def _read_response(self, timeout: float) -> dict[str, Any]:
        try:
            line = self._stdout_queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError(f"No response within {timeout}s") from exc

        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise AdapterRequestError(f"Invalid JSON response: {line}") from exc

    def send(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                self._write_request(payload)
                return self._read_response(timeout=self.config.request_timeout)
            except Exception as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    self.restart()
                    continue
                break

        raise AdapterRequestError(f"Subprocess request failed after retries: {last_error}")

    def close(self):
        self.stop()
