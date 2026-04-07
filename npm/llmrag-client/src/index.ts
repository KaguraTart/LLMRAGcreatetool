export interface QueryResponse {
  results: Array<Record<string, unknown>>;
}

export class LLMRAGClient {
  constructor(private readonly baseUrl = "http://127.0.0.1:7474") {
    const isHttp = this.baseUrl.startsWith("http://");
    const isLoopback = /:\/\/(127\.0\.0\.1|localhost|\[::1\]|::1)(:|\/|$)/.test(this.baseUrl);
    if (isHttp && !isLoopback) {
      throw new Error("Non-local daemon endpoints must use https");
    }
  }

  async health(): Promise<Record<string, unknown>> {
    const response = await fetch(`${this.baseUrl}/health`);
    if (!response.ok) {
      throw new Error(`health failed: ${response.status}`);
    }
    return response.json();
  }

  async query(workspace: string, text: string, topK = 5): Promise<QueryResponse> {
    const response = await fetch(`${this.baseUrl}/api/v1/workspaces/${workspace}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, top_k: topK })
    });
    if (!response.ok) {
      throw new Error(`query failed: ${response.status}`);
    }
    return response.json();
  }
}
