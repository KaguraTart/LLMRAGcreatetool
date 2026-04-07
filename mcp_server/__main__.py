"""Entry point for LLMRAG MCP server."""

from __future__ import annotations

import argparse
import json

from .integration_configs import claude_desktop_config, continue_dev_config, cursor_config
from .server import mcp


def main():
    parser = argparse.ArgumentParser(description="LLMRAG MCP server")
    parser.add_argument(
        "--print-config",
        choices=["claude-desktop", "cursor", "continue-dev"],
        help="Print integration config snippet and exit",
    )
    parser.add_argument(
        "--daemon-url",
        default="http://127.0.0.1:7474",
        help="Daemon URL used for --print-config output",
    )
    args = parser.parse_args()

    if args.print_config:
        if args.print_config == "claude-desktop":
            print(json.dumps(claude_desktop_config(args.daemon_url), indent=2, ensure_ascii=False))
        elif args.print_config == "cursor":
            print(json.dumps(cursor_config(args.daemon_url), indent=2, ensure_ascii=False))
        elif args.print_config == "continue-dev":
            print(json.dumps(continue_dev_config(args.daemon_url), indent=2, ensure_ascii=False))
        return

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

