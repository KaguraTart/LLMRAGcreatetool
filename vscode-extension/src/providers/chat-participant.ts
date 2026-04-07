/**
 * Chat Participant — registers @llmrag in VS Code Chat.
 *
 * Usage in chat:  @llmrag What is the purpose of the ChunkBuilder class?
 */

import * as vscode from "vscode";
import { DaemonClient } from "../daemon-client";

export function registerChatParticipant(
  context: vscode.ExtensionContext,
  client: DaemonClient
): void {
  const participant = vscode.chat.createChatParticipant(
    "llmrag.chat",
    async (
      request: vscode.ChatRequest,
      _chatContext: vscode.ChatContext,
      stream: vscode.ChatResponseStream,
      token: vscode.CancellationToken
    ) => {
      const config = vscode.workspace.getConfiguration("llmrag");
      const workspace = config.get<string>("activeWorkspace", "");

      if (!workspace) {
        stream.markdown(
          "**No active workspace set.**\n\n" +
            "Create a workspace first:\n" +
            "1. Open the LLM RAG sidebar\n" +
            "2. Click **+** to create a workspace\n" +
            "3. Right-click → **Set Active Workspace**\n\n" +
            "Or run: `LLM RAG: Create Workspace` from the Command Palette."
        );
        return;
      }

      const alive = await client.getHealth();
      if (!alive) {
        stream.markdown(
          "**llmragd is not running.**\n\nRun `LLM RAG: Start Daemon` from the Command Palette."
        );
        return;
      }

      if (token.isCancellationRequested) {
        return;
      }

      stream.progress(`Querying workspace **${workspace}**…`);

      try {
        const result = await client.answer(workspace, request.prompt);
        const answer =
          (result.answer as string) ||
          (result.final_answer as string) ||
          JSON.stringify(result, null, 2);

        stream.markdown(answer);

        // Show sources if available
        const sources = result.sources as
          | Array<{ content: string; metadata?: Record<string, unknown> }>
          | undefined;
        if (sources && sources.length > 0) {
          stream.markdown("\n\n---\n**Sources:**\n");
          for (const src of sources.slice(0, 5)) {
            const label =
              (src.metadata?.source as string) || "Unknown source";
            stream.markdown(`- *${label}*`);
          }
        }
      } catch (err) {
        stream.markdown(
          `**Error querying workspace '${workspace}':**\n\n${err}`
        );
      }
    }
  );

  participant.iconPath = new vscode.ThemeIcon("database");
  context.subscriptions.push(participant);
}
