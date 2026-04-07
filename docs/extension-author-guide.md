# Extension Author Guide

## Manifest schema

Create a JSON manifest with:

- `name`
- `version`
- `entry_point` (`module.path:ClassName`)
- `capabilities` (e.g. `retrieval`, `reranker`, `chunker`)
- `enabled` (optional, defaults true)
- `min_core_version` (optional, defaults `0.1.0`)

## Discovery and loading

- Use `src.extensions.registry.ExtensionRegistry`
- Discover manifests from a directory with `discover(path)`
- Load by entry point with `load_from_manifest(manifest)`

## Compatibility checks

Registry enforces a major-version compatibility check using `min_core_version`.

## Capability lookup

Use `by_capability("reranker")` (or other capability names) to fetch loaded extension instances.

## Built-in extensions

- BM25 metadata extension: `src.extensions.builtin.bm25`
- ColBERT-style reranker: `src.extensions.builtin.colbert_reranker`
