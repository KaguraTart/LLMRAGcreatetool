"""
llmrag — CLI adapter for the llmragd daemon.

Commands:
    llmrag daemon start|stop|status
    llmrag workspace create <name>
    llmrag workspace list
    llmrag workspace delete <name>
    llmrag index <path> --workspace <name>
    llmrag query  <text> --workspace <name>
    llmrag answer <text> --workspace <name>
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import click
import requests

_DEFAULT_PORT = int(os.environ.get("LLMRAG_PORT", 7474))
_DEFAULT_HOST = os.environ.get("LLMRAG_HOST", "127.0.0.1")
_PID_FILE = Path.home() / ".llmrag" / "llmragd.pid"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_url() -> str:
    return f"http://{_DEFAULT_HOST}:{_DEFAULT_PORT}"


def _api(path: str) -> str:
    return f"{_base_url()}{path}"


def _is_running() -> bool:
    try:
        r = requests.get(_api("/health"), timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _require_running():
    if not _is_running():
        click.echo("Error: llmragd is not running. Start it with: llmrag daemon start", err=True)
        sys.exit(1)


def _print_json(data):
    click.echo(json.dumps(data, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
def cli():
    """llmrag — LLM RAG toolkit CLI"""


# ---------------------------------------------------------------------------
# daemon
# ---------------------------------------------------------------------------


@cli.group()
def daemon():
    """Manage the llmragd background daemon."""


@daemon.command("start")
@click.option("--port", default=_DEFAULT_PORT, show_default=True, help="Daemon port")
@click.option("--host", default=_DEFAULT_HOST, show_default=True, help="Bind host")
def daemon_start(port: int, host: str):
    """Start the llmragd daemon."""
    if _is_running():
        click.echo(f"llmragd is already running at {_base_url()}")
        return

    repo_root = Path(__file__).parent.parent
    cmd = [sys.executable, "-m", "daemon", "--host", host, "--port", str(port)]
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Wait up to 10s for it to become ready
    for _ in range(20):
        time.sleep(0.5)
        if _is_running():
            click.echo(f"llmragd started (PID {proc.pid}) at http://{host}:{port}")
            return

    click.echo("Warning: llmragd may not have started correctly. Check logs.", err=True)


@daemon.command("stop")
def daemon_stop():
    """Stop the llmragd daemon."""
    if not _PID_FILE.exists():
        click.echo("No PID file found. Is llmragd running?", err=True)
        return

    pid = int(_PID_FILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        click.echo(f"Sent SIGTERM to PID {pid}")
    except ProcessLookupError:
        click.echo(f"Process {pid} not found (already stopped?)")
        _PID_FILE.unlink(missing_ok=True)


@daemon.command("status")
def daemon_status():
    """Show daemon status."""
    if not _is_running():
        click.echo("llmragd: not running")
        return

    r = requests.get(_api("/api/v1/status"), timeout=5)
    r.raise_for_status()
    _print_json(r.json())


# ---------------------------------------------------------------------------
# workspace
# ---------------------------------------------------------------------------


@cli.group()
def workspace():
    """Manage RAG workspaces."""


@workspace.command("create")
@click.argument("name")
def workspace_create(name: str):
    """Create a new workspace."""
    _require_running()
    r = requests.post(_api("/api/v1/workspaces"), json={"name": name}, timeout=10)
    if r.status_code == 409:
        click.echo(f"Workspace '{name}' already exists.", err=True)
        sys.exit(1)
    r.raise_for_status()
    click.echo(f"Workspace '{name}' created.")
    _print_json(r.json())


@workspace.command("list")
def workspace_list():
    """List all workspaces."""
    _require_running()
    r = requests.get(_api("/api/v1/workspaces"), timeout=5)
    r.raise_for_status()
    data = r.json()
    if not data:
        click.echo("No workspaces found.")
        return
    for ws in data:
        click.echo(f"  {ws['name']}  (created: {ws.get('created_at', '?')})")


@workspace.command("delete")
@click.argument("name")
@click.confirmation_option(prompt="Delete workspace and all indexed data?")
def workspace_delete(name: str):
    """Delete a workspace."""
    _require_running()
    r = requests.delete(_api(f"/api/v1/workspaces/{name}"), timeout=10)
    if r.status_code == 404:
        click.echo(f"Workspace '{name}' not found.", err=True)
        sys.exit(1)
    r.raise_for_status()
    click.echo(f"Workspace '{name}' deleted.")


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


@cli.command("index")
@click.argument("path")
@click.option("--workspace", "-w", required=True, help="Target workspace name")
@click.option("--wait/--no-wait", default=True, show_default=True, help="Wait for indexing to complete")
def index_cmd(path: str, workspace: str, wait: bool):
    """Index a file or directory into a workspace."""
    _require_running()
    r = requests.post(
        _api(f"/api/v1/workspaces/{workspace}/index"),
        json={"path": path},
        timeout=10,
    )
    if r.status_code == 404:
        click.echo(f"Workspace '{workspace}' not found.", err=True)
        sys.exit(1)
    r.raise_for_status()
    job_id = r.json()["job_id"]
    click.echo(f"Indexing job started: {job_id}")

    if not wait:
        return

    with click.progressbar(length=100, label="Indexing") as bar:
        last = 0
        while True:
            jr = requests.get(_api(f"/api/v1/workspaces/{workspace}/jobs/{job_id}"), timeout=5)
            jr.raise_for_status()
            job = jr.json()
            delta = job["progress"] - last
            if delta > 0:
                bar.update(delta)
                last = job["progress"]
            if job["status"] in ("done", "failed"):
                break
            time.sleep(1)

    if job["status"] == "done":
        click.echo(f"\nDone. {job.get('message', '')}")
    else:
        click.echo(f"\nFailed: {job.get('message', '')}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


@cli.command("query")
@click.argument("text")
@click.option("--workspace", "-w", required=True, help="Workspace to search")
@click.option("--top-k", default=5, show_default=True, help="Number of results")
def query_cmd(text: str, workspace: str, top_k: int):
    """Vector search in a workspace."""
    _require_running()
    r = requests.post(
        _api(f"/api/v1/workspaces/{workspace}/query"),
        json={"text": text, "top_k": top_k},
        timeout=30,
    )
    if r.status_code == 404:
        click.echo(f"Workspace '{workspace}' not found.", err=True)
        sys.exit(1)
    r.raise_for_status()
    _print_json(r.json())


# ---------------------------------------------------------------------------
# answer
# ---------------------------------------------------------------------------


@cli.command("answer")
@click.argument("text")
@click.option("--workspace", "-w", required=True, help="Workspace to answer from")
def answer_cmd(text: str, workspace: str):
    """RAG answer generation from a workspace."""
    _require_running()
    r = requests.post(
        _api(f"/api/v1/workspaces/{workspace}/answer"),
        json={"text": text},
        timeout=120,
    )
    if r.status_code == 404:
        click.echo(f"Workspace '{workspace}' not found.", err=True)
        sys.exit(1)
    r.raise_for_status()
    result = r.json()
    answer = result.get("answer") or result.get("final_answer") or json.dumps(result, indent=2)
    click.echo(answer)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    cli()


if __name__ == "__main__":
    main()
