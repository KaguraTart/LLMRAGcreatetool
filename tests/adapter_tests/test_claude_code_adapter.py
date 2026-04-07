import os
import tempfile
import textwrap
import unittest

from adapters.claude_code_subprocess import ClaudeCodeAdapter, ClaudeCodeAdapterConfig


class ClaudeCodeAdapterTests(unittest.TestCase):
    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "echo_jsonl.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(
                    textwrap.dedent(
                        """
                        import json
                        import sys

                        for line in sys.stdin:
                            req = json.loads(line)
                            print(json.dumps({"ok": True, "echo": req}), flush=True)
                        """
                    )
                )

            adapter = ClaudeCodeAdapter(
                ClaudeCodeAdapterConfig(
                    command=["python", script_path],
                    request_timeout=3.0,
                    max_retries=0,
                )
            )
            try:
                adapter.start()
                result = adapter.send({"tool": "ping", "arguments": {"x": 1}})
                self.assertTrue(result["ok"])
                self.assertEqual(result["echo"]["tool"], "ping")
            finally:
                adapter.close()


if __name__ == "__main__":
    unittest.main()
