import unittest
from unittest.mock import patch

from mcp_server.server import (
    answer_workspace,
    daemon_health,
    daemon_status,
    index_path,
    query_workspace,
    workspace_create,
    workspace_delete,
    workspace_list,
)


class MCPServerToolTests(unittest.TestCase):
    @patch("mcp_server.server._client.health")
    def test_daemon_health(self, mock_health):
        mock_health.return_value = {"status": "ok"}
        self.assertEqual(daemon_health(), {"status": "ok"})

    @patch("mcp_server.server._client.status")
    def test_daemon_status(self, mock_status):
        mock_status.return_value = {"daemon": "llmragd"}
        self.assertEqual(daemon_status(), {"daemon": "llmragd"})

    @patch("mcp_server.server._client.workspace_list")
    def test_workspace_list(self, mock_list):
        mock_list.return_value = [{"name": "demo"}]
        self.assertEqual(workspace_list(), [{"name": "demo"}])

    @patch("mcp_server.server._client.workspace_create")
    def test_workspace_create(self, mock_create):
        mock_create.return_value = {"name": "demo"}
        self.assertEqual(workspace_create("demo"), {"name": "demo"})

    @patch("mcp_server.server._client.workspace_delete")
    def test_workspace_delete(self, mock_delete):
        mock_delete.return_value = {"deleted": True}
        self.assertEqual(workspace_delete("demo"), {"deleted": True})

    @patch("mcp_server.server._client.index_start")
    @patch("mcp_server.server._client.index_wait")
    def test_index_path_wait(self, mock_wait, mock_start):
        mock_start.return_value = "job-1"
        mock_wait.return_value = {"status": "done", "progress": 100}
        result = index_path("demo", "/tmp/file.txt", wait=True)
        self.assertEqual(result["job_id"], "job-1")
        self.assertEqual(result["job"]["status"], "done")

    @patch("mcp_server.server._client.index_start")
    def test_index_path_no_wait(self, mock_start):
        mock_start.return_value = "job-2"
        result = index_path("demo", "/tmp/file.txt", wait=False)
        self.assertEqual(result, {"job_id": "job-2", "status": "pending"})

    @patch("mcp_server.server._client.query")
    def test_query_workspace(self, mock_query):
        mock_query.return_value = {"results": []}
        self.assertEqual(query_workspace("demo", "hello", 3), {"results": []})

    @patch("mcp_server.server._client.answer")
    def test_answer_workspace(self, mock_answer):
        mock_answer.return_value = {"answer": "ok"}
        self.assertEqual(answer_workspace("demo", "hello"), {"answer": "ok"})


if __name__ == "__main__":
    unittest.main()
