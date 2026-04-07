/**
 * VS Code Extension Entry Point
 *
 * Activates on startup, wires up:
 *  - DaemonClient
 *  - WorkspaceTreeProvider (sidebar)
 *  - Chat participant (@llmrag)
 *  - Command palette commands
 *  - FileSystemWatcher for auto-index
 */

import * as child_process from "child_process";
import * as path from "path";
import * as vscode from "vscode";

import { DaemonClient } from "./daemon-client";
import { registerChatParticipant } from "./providers/chat-participant";
import { WorkspaceTreeProvider } from "./providers/workspace-tree";
import { trackJob } from "./progress";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getClient(context: vscode.ExtensionContext): DaemonClient {
  const cfg = vscode.workspace.getConfiguration("llmrag");
  const host = cfg.get<string>("daemonHost", "127.0.0.1");
  const port = cfg.get<number>("daemonPort", 7474);
  return new DaemonClient(host, port);
}

function getPythonPath(): string {
  return (
    vscode.workspace.getConfiguration("llmrag").get<string>("pythonPath") ||
    "python"
  );
}

function getRepoRoot(context: vscode.ExtensionContext): string {
  // Extension lives at <repo>/vscode-extension; repo root is one level up.
  return path.dirname(context.extensionPath);
}

// ---------------------------------------------------------------------------
// Activation
// ---------------------------------------------------------------------------

export async function activate(
  context: vscode.ExtensionContext
): Promise<void> {
  const client = getClient(context);

  // Sidebar tree view
  const treeProvider = new WorkspaceTreeProvider(client, context);
  const treeView = vscode.window.createTreeView("llmrag.workspacesView", {
    treeDataProvider: treeProvider,
    showCollapseAll: false,
  });
  context.subscriptions.push(treeView);

  // Chat participant
  registerChatParticipant(context, client);

  // ------------------------------------------------------------------
  // Commands
  // ------------------------------------------------------------------

  // daemon start
  context.subscriptions.push(
    vscode.commands.registerCommand("llmrag.daemonStart", async () => {
      const alive = await client.getHealth();
      if (alive) {
        vscode.window.showInformationMessage("llmragd is already running.");
        return;
      }
      const python = getPythonPath();
      const repoRoot = getRepoRoot(context);
      const proc = child_process.spawn(python, ["-m", "daemon"], {
        cwd: repoRoot,
        detached: true,
        stdio: "ignore",
      });
      proc.unref();
      vscode.window.showInformationMessage(
        `Starting llmragd (PID ${proc.pid})…`
      );
      // Poll for readiness
      for (let i = 0; i < 20; i++) {
        await new Promise((r) => setTimeout(r, 500));
        if (await client.getHealth()) {
          vscode.window.showInformationMessage("llmragd is ready.");
          treeProvider.refresh();
          return;
        }
      }
      vscode.window.showWarningMessage(
        "llmragd may not have started. Check the terminal."
      );
    })
  );

  // daemon stop
  context.subscriptions.push(
    vscode.commands.registerCommand("llmrag.daemonStop", async () => {
      const alive = await client.getHealth();
      if (!alive) {
        vscode.window.showInformationMessage("llmragd is not running.");
        return;
      }
      // Use the CLI to stop (sends SIGTERM via PID file)
      const python = getPythonPath();
      const repoRoot = getRepoRoot(context);
      child_process.exec(
        `${python} cli/llmrag.py daemon stop`,
        { cwd: repoRoot },
        (err) => {
          if (err) {
            vscode.window.showErrorMessage(`Failed to stop daemon: ${err.message}`);
          } else {
            vscode.window.showInformationMessage("llmragd stopped.");
            treeProvider.refresh();
          }
        }
      );
    })
  );

  // daemon status
  context.subscriptions.push(
    vscode.commands.registerCommand("llmrag.daemonStatus", async () => {
      const alive = await client.getHealth();
      if (!alive) {
        vscode.window.showInformationMessage("llmragd: not running");
        return;
      }
      try {
        const status = await client.getStatus();
        vscode.window.showInformationMessage(
          `llmragd v${status.version} — ${status.workspace_count} workspace(s)`
        );
      } catch (err) {
        vscode.window.showErrorMessage(`Status error: ${err}`);
      }
    })
  );

  // workspace create
  context.subscriptions.push(
    vscode.commands.registerCommand("llmrag.workspaceCreate", async () => {
      const name = await vscode.window.showInputBox({
        prompt: "Workspace name",
        placeHolder: "my-project",
        validateInput: (v) =>
          /^[a-z0-9_-]+$/i.test(v) ? null : "Use letters, numbers, _ or -",
      });
      if (!name) {
        return;
      }
      const alive = await client.getHealth();
      if (!alive) {
        vscode.window.showErrorMessage(
          "llmragd is not running. Start it first."
        );
        return;
      }
      try {
        await client.createWorkspace(name);
        vscode.window.showInformationMessage(`Workspace '${name}' created.`);
        treeProvider.refresh();
      } catch (err) {
        vscode.window.showErrorMessage(`Failed to create workspace: ${err}`);
      }
    })
  );

  // workspace delete
  context.subscriptions.push(
    vscode.commands.registerCommand(
      "llmrag.workspaceDelete",
      async (item: { workspace: { name: string } } | undefined) => {
        const name = item?.workspace?.name || (await vscode.window.showInputBox({ prompt: "Workspace name to delete" }));
        if (!name) {
          return;
        }
        const confirm = await vscode.window.showWarningMessage(
          `Delete workspace '${name}' and all indexed data?`,
          { modal: true },
          "Delete"
        );
        if (confirm !== "Delete") {
          return;
        }
        try {
          await client.deleteWorkspace(name);
          vscode.window.showInformationMessage(`Workspace '${name}' deleted.`);
          // Clear active if it was this one
          const cfg = vscode.workspace.getConfiguration("llmrag");
          if (cfg.get<string>("activeWorkspace") === name) {
            await cfg.update("activeWorkspace", "", vscode.ConfigurationTarget.Global);
          }
          treeProvider.refresh();
        } catch (err) {
          vscode.window.showErrorMessage(`Failed to delete workspace: ${err}`);
        }
      }
    )
  );

  // workspace set active
  context.subscriptions.push(
    vscode.commands.registerCommand(
      "llmrag.workspaceSetActive",
      async (item: { workspace: { name: string } } | undefined) => {
        const name = item?.workspace?.name || (await vscode.window.showInputBox({ prompt: "Workspace name to activate" }));
        if (!name) {
          return;
        }
        const cfg = vscode.workspace.getConfiguration("llmrag");
        await cfg.update("activeWorkspace", name, vscode.ConfigurationTarget.Global);
        vscode.window.showInformationMessage(`Active workspace set to '${name}'.`);
        treeProvider.refresh();
      }
    )
  );

  // workspace refresh
  context.subscriptions.push(
    vscode.commands.registerCommand("llmrag.workspaceRefresh", () => {
      treeProvider.refresh();
    })
  );

  // index current file
  context.subscriptions.push(
    vscode.commands.registerCommand("llmrag.indexCurrentFile", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage("No active file.");
        return;
      }
      const filePath = editor.document.uri.fsPath;
      await _indexPath(client, treeProvider, filePath);
    })
  );

  // index folder
  context.subscriptions.push(
    vscode.commands.registerCommand(
      "llmrag.indexFolder",
      async (item: { workspace: { name: string } } | undefined) => {
        const uris = await vscode.window.showOpenDialog({
          canSelectFiles: false,
          canSelectFolders: true,
          canSelectMany: false,
          openLabel: "Index this folder",
        });
        if (!uris || uris.length === 0) {
          return;
        }
        const folderPath = uris[0].fsPath;
        const workspaceName = item?.workspace?.name;
        await _indexPath(client, treeProvider, folderPath, workspaceName);
      }
    )
  );

  // query workspace
  context.subscriptions.push(
    vscode.commands.registerCommand("llmrag.queryWorkspace", async () => {
      const cfg = vscode.workspace.getConfiguration("llmrag");
      const workspace = cfg.get<string>("activeWorkspace", "");
      if (!workspace) {
        vscode.window.showErrorMessage("No active workspace set.");
        return;
      }
      const text = await vscode.window.showInputBox({ prompt: "Search query" });
      if (!text) {
        return;
      }
      try {
        const results = await client.query(workspace, text);
        const items = results.map(
          (r, i) => `${i + 1}. [score=${r.score.toFixed(3)}] ${r.content.slice(0, 120)}…`
        );
        vscode.window.showQuickPick(items, {
          placeHolder: `${results.length} results from '${workspace}'`,
        });
      } catch (err) {
        vscode.window.showErrorMessage(`Query failed: ${err}`);
      }
    })
  );

  // ------------------------------------------------------------------
  // Auto-index file watcher
  // ------------------------------------------------------------------
  _setupAutoIndex(context, client, treeProvider);
}

// ---------------------------------------------------------------------------
// Index helper
// ---------------------------------------------------------------------------

async function _indexPath(
  client: DaemonClient,
  treeProvider: WorkspaceTreeProvider,
  filePath: string,
  workspaceName?: string
): Promise<void> {
  const cfg = vscode.workspace.getConfiguration("llmrag");
  const workspace = workspaceName || cfg.get<string>("activeWorkspace", "");
  if (!workspace) {
    vscode.window.showErrorMessage(
      "No active workspace. Set one in LLM RAG settings."
    );
    return;
  }
  const alive = await client.getHealth();
  if (!alive) {
    vscode.window.showErrorMessage("llmragd is not running. Start it first.");
    return;
  }
  try {
    const job = await client.indexPath(workspace, filePath);
    trackJob(client, workspace, job.id, `Indexing into '${workspace}'`);
    treeProvider.refresh();
  } catch (err) {
    vscode.window.showErrorMessage(`Failed to start indexing: ${err}`);
  }
}

// ---------------------------------------------------------------------------
// Auto-index watcher
// ---------------------------------------------------------------------------

let _autoIndexTimer: NodeJS.Timeout | undefined;

function _setupAutoIndex(
  context: vscode.ExtensionContext,
  client: DaemonClient,
  treeProvider: WorkspaceTreeProvider
): void {
  const cfg = vscode.workspace.getConfiguration("llmrag");
  if (!cfg.get<boolean>("autoIndex")) {
    return;
  }

  const globPattern = cfg.get<string>(
    "autoIndexGlob",
    "**/*.{md,txt,pdf,docx}"
  );
  const watcher = vscode.workspace.createFileSystemWatcher(globPattern);

  const _debouncedIndex = (uri: vscode.Uri) => {
    if (_autoIndexTimer) {
      clearTimeout(_autoIndexTimer);
    }
    _autoIndexTimer = setTimeout(() => {
      _indexPath(client, treeProvider, uri.fsPath);
    }, 500);
  };

  watcher.onDidChange(_debouncedIndex);
  watcher.onDidCreate(_debouncedIndex);
  context.subscriptions.push(watcher);
}

export function deactivate(): void {
  if (_autoIndexTimer) {
    clearTimeout(_autoIndexTimer);
  }
}
