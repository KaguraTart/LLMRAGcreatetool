"""
llmragd — Entry Point

Usage:
    python -m daemon                  # start on default port 7474
    python -m daemon --port 8080
    python -m daemon --host 0.0.0.0
"""

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

import uvicorn

_PID_DIR = Path.home() / ".llmrag"
_PID_FILE = _PID_DIR / "llmragd.pid"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("llmragd")


def _write_pid():
    _PID_DIR.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(os.getpid()))


def _remove_pid():
    if _PID_FILE.exists():
        _PID_FILE.unlink()


def _handle_signal(signum, frame):
    logger.info("Received shutdown signal, stopping llmragd...")
    _remove_pid()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="llmragd — LLM RAG Daemon")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7474, help="Bind port (default: 7474)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    parser.add_argument("--log-level", default="info", help="Uvicorn log level")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    _write_pid()
    logger.info(f"Starting llmragd on {args.host}:{args.port} (PID {os.getpid()})")
    logger.info(f"PID file: {_PID_FILE}")

    try:
        uvicorn.run(
            "daemon.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
        )
    finally:
        _remove_pid()


if __name__ == "__main__":
    main()
