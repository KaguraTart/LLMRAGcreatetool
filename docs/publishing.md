# Publishing Guide

## VS Code Marketplace

- Package extension from `vscode-extension/`
- Build: `npm ci && npm run build`
- Package: `npm run package`
- Publish via workflow: `.github/workflows/publish-vscode.yml`

## npm package (`@llmrag/client`)

- Package path: `npm/llmrag-client`
- Build: `npm ci && npm run build`
- Publish via workflow: `.github/workflows/publish-npm.yml`

## pip package (`llmrag`)

- Build artifacts with `python -m build`
- Publish via workflow: `.github/workflows/publish-pypi.yml`
- Uses trusted publishing or PyPI API token secret.
