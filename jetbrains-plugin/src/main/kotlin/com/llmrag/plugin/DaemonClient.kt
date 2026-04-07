package com.llmrag.plugin

import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Duration

class DaemonClient(
    private val baseUrl: String = System.getenv("LLMRAG_DAEMON_URL") ?: "http://127.0.0.1:7474",
    private val timeoutSeconds: Long = 10,
) {
    init {
        val isHttp = baseUrl.startsWith("http://")
        val isLoopback = Regex("""://(127\\.0\\.0\\.1|localhost|\\[::1]|::1)(:|/|$)""").containsMatchIn(baseUrl)
        require(!isHttp || isLoopback) { "Non-local daemon endpoints must use https" }
    }

    private val http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(timeoutSeconds)).build()

    fun health(): String {
        val request = HttpRequest.newBuilder(URI.create("$baseUrl/health"))
            .timeout(Duration.ofSeconds(timeoutSeconds))
            .GET()
            .build()
        return http.send(request, HttpResponse.BodyHandlers.ofString()).body()
    }

    fun query(workspace: String, text: String, topK: Int = 5): String {
        val payload = """{"text":${jsonString(text)},"top_k":$topK}"""
        val request = HttpRequest.newBuilder(URI.create("$baseUrl/api/v1/workspaces/$workspace/query"))
            .timeout(Duration.ofSeconds(timeoutSeconds))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(payload))
            .build()
        return http.send(request, HttpResponse.BodyHandlers.ofString()).body()
    }

    private fun jsonString(value: String): String {
        val escaped = value
            .replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\n", "\\n")
        return "\"$escaped\""
    }
}
