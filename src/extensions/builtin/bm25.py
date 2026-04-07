"""Built-in BM25 extension metadata."""

from __future__ import annotations

from src.extensions.base import ExtensionManifest


class BM25Extension:
    manifest = ExtensionManifest(
        name="bm25",
        version="1.0.0",
        entry_point="src.extensions.builtin.bm25:BM25Extension",
        capabilities=["retrieval"],
        enabled=True,
    )
