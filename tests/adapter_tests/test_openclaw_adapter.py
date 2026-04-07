import unittest
from unittest.mock import patch

from adapters.base import AdapterAuthError, AdapterRequestError
from adapters.openclaw_plugin import OpenClawAdapter, OpenClawAdapterConfig


class OpenClawAdapterTests(unittest.TestCase):
    @patch("adapters.openclaw_plugin.LLMRAGDaemonClient")
    def test_dispatch_workspace_list(self, client_cls):
        mock_client = client_cls.return_value
        mock_client.workspace_list.return_value = [{"name": "demo"}]

        adapter = OpenClawAdapter(OpenClawAdapterConfig())
        result = adapter.handle_request({"tool": "workspace_list", "arguments": {}})

        self.assertTrue(result["ok"])
        self.assertEqual(result["result"], [{"name": "demo"}])

    @patch("adapters.openclaw_plugin.LLMRAGDaemonClient")
    def test_auth_rejected_with_wrong_bearer(self, client_cls):
        _ = client_cls.return_value
        adapter = OpenClawAdapter(OpenClawAdapterConfig(auth_token="secret"))

        with self.assertRaises(AdapterAuthError):
            adapter.handle_request({"tool": "daemon_health", "arguments": {}}, auth_header="Bearer wrong")

    @patch("adapters.openclaw_plugin.LLMRAGDaemonClient")
    def test_unknown_tool_raises(self, client_cls):
        _ = client_cls.return_value
        adapter = OpenClawAdapter(OpenClawAdapterConfig())

        with self.assertRaises(AdapterRequestError):
            adapter.handle_request({"tool": "not_exists", "arguments": {}})


if __name__ == "__main__":
    unittest.main()
