/**
 * Progress tracker — subscribes to SSE job stream and shows a VS Code progress notification.
 */

import * as vscode from "vscode";
import { DaemonClient } from "./daemon-client";

interface ProgressEvent {
  progress: number;
  message: string;
  status: string;
}

export function trackJob(
  client: DaemonClient,
  workspace: string,
  jobId: string,
  label: string
): void {
  vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: label,
      cancellable: false,
    },
    (progress) => {
      return new Promise<void>((resolve, reject) => {
        const es = client.streamJob(workspace, jobId);
        let lastProgress = 0;

        es.addEventListener("progress", (event: MessageEvent) => {
          try {
            const data: ProgressEvent = JSON.parse(event.data);
            const delta = data.progress - lastProgress;
            if (delta > 0) {
              progress.report({
                increment: delta,
                message: data.message,
              });
              lastProgress = data.progress;
            }
          } catch {
            // ignore parse errors
          }
        });

        es.addEventListener("end", (event: MessageEvent) => {
          es.close();
          try {
            const data: { status: string } = JSON.parse(event.data);
            if (data.status === "failed") {
              vscode.window.showErrorMessage(
                `${label} failed. Check the output for details.`
              );
            } else {
              vscode.window.showInformationMessage(`${label} completed.`);
            }
          } catch {
            // ignore
          }
          resolve();
        });

        es.onerror = () => {
          es.close();
          reject(new Error("SSE connection error"));
        };

        // Safety timeout: 30 minutes
        setTimeout(() => {
          es.close();
          resolve();
        }, 30 * 60 * 1000);
      });
    }
  );
}
