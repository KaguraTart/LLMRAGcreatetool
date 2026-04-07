"""Adapter integrations for external tools and plugins."""

from .base import AdapterAuthError, AdapterConfigError, AdapterRequestError
from .claude_code_subprocess import ClaudeCodeAdapter, ClaudeCodeAdapterConfig
from .openclaw_plugin import OpenClawAdapter, OpenClawAdapterConfig

__all__ = [
    "AdapterAuthError",
    "AdapterConfigError",
    "AdapterRequestError",
    "ClaudeCodeAdapter",
    "ClaudeCodeAdapterConfig",
    "OpenClawAdapter",
    "OpenClawAdapterConfig",
]
