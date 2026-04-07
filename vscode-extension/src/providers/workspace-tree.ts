/**
 * WorkspaceTreeProvider — Activity bar tree view for llmrag workspaces.
 */

import * as vscode from "vscode";
import { DaemonClient, Workspace } from "../daemon-client";

export class WorkspaceTreeItem extends vscode.TreeItem {
  constructor(
    public readonly workspace: Workspace,
    public readonly isActive: boolean
  ) {
    const label = isActive
      ? `$(check) ${workspace.name}`
      : workspace.name;
    super(label, vscode.TreeItemCollapsibleState.None);

    this.contextValue = "workspace";
    this.description = `${workspace.doc_count ?? 0} docs`;
    this.tooltip = new vscode.MarkdownString(
      `**${workspace.name}**\n\nCreated: ${workspace.created_at}\nDocs: ${workspace.doc_count ?? 0}`
    );
    this.iconPath = isActive
      ? new vscode.ThemeIcon("database", new vscode.ThemeColor("charts.green"))
      : new vscode.ThemeIcon("database");
  }
}

export class WorkspaceTreeProvider
  implements vscode.TreeDataProvider<WorkspaceTreeItem>
{
  private _onDidChangeTreeData = new vscode.EventEmitter<
    WorkspaceTreeItem | undefined | null | void
  >();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  constructor(
    private readonly client: DaemonClient,
    private readonly context: vscode.ExtensionContext
  ) {}

  refresh(): void {
    this._onDidChangeTreeData.fire();
  }

  getTreeItem(element: WorkspaceTreeItem): vscode.TreeItem {
    return element;
  }

  async getChildren(
    element?: WorkspaceTreeItem
  ): Promise<WorkspaceTreeItem[]> {
    if (element) {
      return [];
    }

    const alive = await this.client.getHealth();
    if (!alive) {
      vscode.window.showWarningMessage(
        "llmragd is not running. Use 'LLM RAG: Start Daemon'."
      );
      return [];
    }

    try {
      const workspaces = await this.client.listWorkspaces();
      const active = vscode.workspace
        .getConfiguration("llmrag")
        .get<string>("activeWorkspace", "");

      // Enrich with doc counts
      const enriched = await Promise.all(
        workspaces.map(async (ws) => {
          try {
            const detail = await this.client.getWorkspace(ws.name);
            return { ...ws, doc_count: detail.doc_count };
          } catch {
            return ws;
          }
        })
      );

      return enriched.map(
        (ws) => new WorkspaceTreeItem(ws, ws.name === active)
      );
    } catch (err) {
      vscode.window.showErrorMessage(`Failed to list workspaces: ${err}`);
      return [];
    }
  }
}
