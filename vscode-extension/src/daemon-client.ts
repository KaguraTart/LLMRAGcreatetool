/**
 * DaemonClient — HTTP client for the llmragd REST API.
 *
 * All methods throw on non-2xx responses.
 */

export interface Workspace {
  name: string;
  chroma_dir: string;
  created_at: string;
  config_overrides: Record<string, unknown>;
  doc_count?: number;
}

export interface Job {
  id: string;
  workspace: string;
  type: string;
  status: "pending" | "running" | "done" | "failed";
  progress: number;
  message: string;
  started_at: string | null;
  finished_at: string | null;
  result: Record<string, unknown> | null;
}

export interface SearchResult {
  id: string;
  content: string;
  score: number;
  metadata: Record<string, unknown>;
}

export interface AnswerResult {
  answer?: string;
  final_answer?: string;
  sources?: SearchResult[];
  [key: string]: unknown;
}

export interface StatusResponse {
  daemon: string;
  version: string;
  workspace_count: number;
  workspaces: Array<{ name: string; doc_count: number }>;
}

export class DaemonClient {
  private baseUrl: string;

  constructor(host = "127.0.0.1", port = 7474) {
    this.baseUrl = `http://${host}:${port}`;
  }

  private async _fetch<T>(
    method: string,
    path: string,
    body?: unknown
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const init: RequestInit = {
      method,
      headers: { "Content-Type": "application/json" },
    };
    if (body !== undefined) {
      init.body = JSON.stringify(body);
    }
    const res = await fetch(url, init);
    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText);
      throw new Error(`llmragd ${method} ${path} → ${res.status}: ${text}`);
    }
    if (res.status === 204) {
      return undefined as unknown as T;
    }
    return res.json() as Promise<T>;
  }

  // System

  async getHealth(): Promise<boolean> {
    try {
      await this._fetch<{ status: string }>("GET", "/health");
      return true;
    } catch {
      return false;
    }
  }

  async getStatus(): Promise<StatusResponse> {
    return this._fetch<StatusResponse>("GET", "/api/v1/status");
  }

  // Workspaces

  async listWorkspaces(): Promise<Workspace[]> {
    return this._fetch<Workspace[]>("GET", "/api/v1/workspaces");
  }

  async createWorkspace(
    name: string,
    configOverrides: Record<string, unknown> = {}
  ): Promise<Workspace> {
    return this._fetch<Workspace>("POST", "/api/v1/workspaces", {
      name,
      config_overrides: configOverrides,
    });
  }

  async getWorkspace(name: string): Promise<Workspace> {
    return this._fetch<Workspace>("GET", `/api/v1/workspaces/${name}`);
  }

  async deleteWorkspace(name: string): Promise<void> {
    return this._fetch<void>("DELETE", `/api/v1/workspaces/${name}`);
  }

  // Indexing

  async indexPath(workspace: string, path: string): Promise<Job> {
    return this._fetch<Job>(
      "POST",
      `/api/v1/workspaces/${workspace}/index`,
      { path }
    );
  }

  async getJob(workspace: string, jobId: string): Promise<Job> {
    return this._fetch<Job>(
      "GET",
      `/api/v1/workspaces/${workspace}/jobs/${jobId}`
    );
  }

  /**
   * Returns an EventSource connected to the SSE progress stream for a job.
   * Caller is responsible for closing it.
   */
  streamJob(workspace: string, jobId: string): EventSource {
    const url = `${this.baseUrl}/api/v1/workspaces/${workspace}/jobs/${jobId}/stream`;
    return new EventSource(url);
  }

  // Query / Answer

  async query(
    workspace: string,
    text: string,
    topK = 5
  ): Promise<SearchResult[]> {
    const res = await this._fetch<{ results: SearchResult[] }>(
      "POST",
      `/api/v1/workspaces/${workspace}/query`,
      { text, top_k: topK }
    );
    return res.results;
  }

  async answer(workspace: string, text: string): Promise<AnswerResult> {
    return this._fetch<AnswerResult>(
      "POST",
      `/api/v1/workspaces/${workspace}/answer`,
      { text }
    );
  }
}
