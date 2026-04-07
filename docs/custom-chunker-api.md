# Custom Chunker API

## Contract

A custom chunker extension should expose a class with:

- `manifest.capabilities` containing `chunker`
- A callable method like `chunk(document: dict) -> list[dict]`

Each returned chunk should include:

- `content` (required)
- `metadata` (optional)
- `source/page/category` fields as needed by your pipeline

## Lifecycle

1. Load extension manifest through `ExtensionRegistry`
2. Instantiate extension class via entry point
3. Invoke chunker during document processing before indexing

## Validation rules

- Return a list
- Each chunk must be non-empty text
- Enforce configured minimum chunk size before index insert

## Migration notes

If you currently use `ChunkBuilder` strategies, keep them as default fallback.
Add custom chunkers incrementally and gate with config flags.
