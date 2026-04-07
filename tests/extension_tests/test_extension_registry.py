import json
import tempfile
import unittest

from src.extensions import ExtensionRegistry
from src.extensions.base import ExtensionManifest


class _DemoExtension:
    pass


class ExtensionRegistryTests(unittest.TestCase):
    def test_register_and_capability_lookup(self):
        registry = ExtensionRegistry(core_version="0.1.0")
        manifest = ExtensionManifest(
            name="demo",
            version="1.0.0",
            entry_point="tests.extensions.test_extension_registry:_DemoExtension",
            capabilities=["reranker"],
        )
        ext = _DemoExtension()
        registry.register(manifest, ext)

        self.assertEqual(registry.list(), ["demo"])
        self.assertEqual(registry.by_capability("reranker"), [ext])

    def test_discover_manifest_files(self):
        registry = ExtensionRegistry()
        manifest = {
            "name": "demo",
            "version": "1.0.0",
            "entry_point": "tests.extensions.test_extension_registry:_DemoExtension",
            "capabilities": ["retrieval"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with open(f"{tmp}/demo.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            discovered = registry.discover(tmp)

        self.assertEqual(len(discovered), 1)
        self.assertEqual(discovered[0].name, "demo")


if __name__ == "__main__":
    unittest.main()
