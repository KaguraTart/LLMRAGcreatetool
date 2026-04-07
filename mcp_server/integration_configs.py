"""MCP client integration config templates."""

from __future__ import annotations

from typing import Any


def _server_block(
    daemon_url: str,
    command: str = "python",
    args: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "command": command,
        "args": args or ["-m", "mcp_server"],
        "env": {
            "LLMRAG_DAEMON_URL": daemon_url,
        },
    }


def claude_desktop_config(daemon_url: str) -> dict[str, Any]:
    return {
        "mcpServers": {
            "llmrag": _server_block(daemon_url=daemon_url),
        }
    }


def cursor_config(daemon_url: str) -> dict[str, Any]:
    return {
        "mcpServers": {
            "llmrag": _server_block(daemon_url=daemon_url),
        }
    }


def continue_dev_config(daemon_url: str) -> dict[str, Any]:
    return {
        "mcpServers": [
            {
                "name": "llmrag",
                **_server_block(daemon_url=daemon_url),
            }
        ]
    }


def openclaw_config(daemon_url: str) -> dict[str, Any]:
    return {
        "plugins": {
            "llmrag": {
                "transport": "stdio",
                **_server_block(daemon_url=daemon_url),
            }
        }
    }


def claude_code_config(daemon_url: str) -> dict[str, Any]:
    return {
        "mcp": {
            "servers": {
                "llmrag": _server_block(daemon_url=daemon_url),
            }
        }
    }


def jetbrains_config(daemon_url: str) -> dict[str, Any]:
    return {
        "llmrag": {
            "daemonUrl": daemon_url,
            "mcpServer": _server_block(daemon_url=daemon_url),
        }
    }
