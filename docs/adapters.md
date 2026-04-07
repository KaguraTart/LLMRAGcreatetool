# Adapter Setup Guide

## OpenClaw plugin adapter

- Module: `adapters/openclaw_plugin.py`
- Config fields:
  - `daemon_url`
  - `timeout`
  - `auth_token` or `auth_token_env` (`LLMRAG_OPENCLAW_TOKEN`)
- Security:
  - Non-local `daemon_url` must use `https://` (plain HTTP is only allowed for localhost loopback).
- Request contract:
  - `{"tool": "workspace_list", "arguments": {}}`
- Auth:
  - If token is configured, pass `Authorization: Bearer <token>`

Supported tools:

- `daemon_health`
- `daemon_status`
- `workspace_list`
- `workspace_create`
- `workspace_delete`
- `index_path`
- `query_workspace`
- `answer_workspace`

## Claude Code subprocess adapter

- Module: `adapters/claude_code_subprocess.py`
- Protocol: JSONL request/response on stdin/stdout
- Lifecycle:
  - `start()` → boot subprocess
  - `send(payload)` → send request with timeout/retry handling
  - `restart()` on transient failures
  - `close()` for cleanup

Default command runs MCP server:

```bash
python -m mcp_server
```

## Integration snippet generation

```bash
python -m mcp_server --print-config openclaw
python -m mcp_server --print-config claude-code
python -m mcp_server --print-config jetbrains
```
