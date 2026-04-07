# JetBrains Plugin Packaging

Project path: `jetbrains-plugin/`

## Local build

```bash
cd jetbrains-plugin
./gradlew buildPlugin
```

## Local test

```bash
cd jetbrains-plugin
./gradlew test
```

## Output artifact

Built plugin ZIP is generated under `jetbrains-plugin/build/distributions/`.

## Notes

- Tool window ID: `LLM RAG`
- Daemon endpoint defaults to `http://127.0.0.1:7474` via `LLMRAG_DAEMON_URL` override support.
