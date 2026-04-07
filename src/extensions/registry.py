"""Dynamic extension registry and discovery."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from .base import ExtensionManifest


class ExtensionRegistry:
    def __init__(self, core_version: str = "0.1.0"):
        self.core_version = core_version
        self._manifests: dict[str, ExtensionManifest] = {}
        self._instances: dict[str, object] = {}

    @staticmethod
    def _major(version: str) -> int:
        head = (version or "0").split(".")[0]
        try:
            return int(head)
        except ValueError as exc:
            raise ValueError(f"Invalid semantic version: {version}") from exc

    def _check_compatibility(self, manifest: ExtensionManifest):
        if self._major(self.core_version) < self._major(manifest.min_core_version):
            raise ValueError(
                f"Extension {manifest.name} requires core>={manifest.min_core_version}, got {self.core_version}"
            )

    def register(self, manifest: ExtensionManifest, instance: object):
        if not manifest.enabled:
            return
        self._check_compatibility(manifest)
        self._manifests[manifest.name] = manifest
        self._instances[manifest.name] = instance

    def get(self, name: str):
        return self._instances[name]

    def list(self) -> list[str]:
        return sorted(self._instances.keys())

    def by_capability(self, capability: str) -> list[object]:
        out = []
        for name, manifest in self._manifests.items():
            if capability in manifest.capabilities:
                out.append(self._instances[name])
        return out

    def discover(self, directory: str | Path) -> list[ExtensionManifest]:
        path = Path(directory)
        manifests: list[ExtensionManifest] = []
        if not path.exists():
            return manifests
        for mf in sorted(path.glob("*.json")):
            data = json.loads(mf.read_text(encoding="utf-8"))
            manifests.append(ExtensionManifest(**data))
        return manifests

    def load_from_manifest(self, manifest: ExtensionManifest):
        module_name, class_name = manifest.entry_point.split(":", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        instance = cls()
        self.register(manifest, instance)
        return instance
