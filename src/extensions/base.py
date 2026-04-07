"""Extension interfaces and manifest schema."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExtensionManifest:
    name: str
    version: str
    entry_point: str
    capabilities: list[str] = field(default_factory=list)
    enabled: bool = True
    min_core_version: str = "0.1.0"


class ExtensionBase:
    manifest: ExtensionManifest


class RerankerExtension(ExtensionBase):
    def rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        raise NotImplementedError
