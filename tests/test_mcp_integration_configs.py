import unittest

from mcp_server.integration_configs import (
    claude_desktop_config,
    continue_dev_config,
    cursor_config,
)


class MCPIntegrationConfigTests(unittest.TestCase):
    def setUp(self):
        self.daemon_url = "http://127.0.0.1:7474"

    def test_claude_desktop_config(self):
        config = claude_desktop_config(self.daemon_url)
        server = config["mcpServers"]["llmrag"]
        self.assertEqual(server["command"], "python")
        self.assertEqual(server["args"], ["-m", "mcp_server"])
        self.assertEqual(server["env"]["LLMRAG_DAEMON_URL"], self.daemon_url)

    def test_cursor_config(self):
        config = cursor_config(self.daemon_url)
        server = config["mcpServers"]["llmrag"]
        self.assertEqual(server["command"], "python")
        self.assertEqual(server["args"], ["-m", "mcp_server"])
        self.assertEqual(server["env"]["LLMRAG_DAEMON_URL"], self.daemon_url)

    def test_continue_dev_config(self):
        config = continue_dev_config(self.daemon_url)
        self.assertIsInstance(config["mcpServers"], list)
        server = config["mcpServers"][0]
        self.assertEqual(server["name"], "llmrag")
        self.assertEqual(server["command"], "python")
        self.assertEqual(server["args"], ["-m", "mcp_server"])
        self.assertEqual(server["env"]["LLMRAG_DAEMON_URL"], self.daemon_url)


if __name__ == "__main__":
    unittest.main()
