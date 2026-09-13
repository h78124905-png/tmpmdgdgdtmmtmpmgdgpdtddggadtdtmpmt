package com.example.lfmmobile

import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.nio.charset.StandardCharsets

/**
 * Web-search facade used by the UI.
 *
 * The actual search backend is now fast-web-search-mcp behind the HTTP bridge
 * in tools/fast-web-search-bridge. The legacy DuckDuckGo Instant Answer API is
 * intentionally no longer called from the Android app.
 *
 * The heuristic is retained temporarily as a compatibility layer. It will be
 * removed when the llama.cpp tool-calling loop is wired in, so the model can
 * decide when to call web_search/news_search/fetch_url/search_and_fetch.
 */
data class SearchResult(
    val title: String,
    val url: String,
    val snippet: String
)

class SearchService {
    companion object {
        /**
         * HTTP endpoint exposed by the companion MCP bridge.
         * Override at runtime with: -Dlfm.mcp.bridge.url=http://host:8787/call
         * when launching a development build.
         */
        private const val DEFAULT_BRIDGE_URL = "http://127.0.0.1:8787/call"

        fun shouldSearch(query: String): Boolean {
            val q = query.lowercase()
            val currentSignals = listOf(
                "today", "latest", "recent", "news", "price", "weather",
                "今", "今日", "最新", "現在", "ニュース", "価格", "値段", "天気",
                "発売", "スペック", "仕様", "いつ", "2026"
            )
            return currentSignals.any { q.contains(it) } || q.endsWith("?") || q.endsWith("？")
        }
    }

    private val bridgeUrl: String =
        System.getProperty("lfm.mcp.bridge.url")?.takeIf { it.isNotBlank() } ?: DEFAULT_BRIDGE_URL

    fun search(query: String, limit: Int = 5): List<SearchResult> {
        val arguments = JSONObject()
            .put("query", query)
            .put("max_results", limit.coerceIn(1, 25))
            .put("region", "jp-jp")
            .put("language", "ja")
            .put("safe_search", "moderate")

        val root = callBridge("web_search", arguments) ?: return emptyList()
        val results = root.optJSONArray("results") ?: return emptyList()
        return parseResults(results, limit)
    }

    fun newsSearch(query: String, limit: Int = 5): List<SearchResult> {
        val arguments = JSONObject()
            .put("query", query)
            .put("max_results", limit.coerceIn(1, 25))
            .put("region", "jp-jp")
            .put("language", "ja")
            .put("time_range", "week")

        val root = callBridge("news_search", arguments) ?: return emptyList()
        val results = root.optJSONArray("results") ?: return emptyList()
        return parseResults(results, limit)
    }

    fun searchAndFetch(query: String, searchResults: Int = 8, fetchResults: Int = 3): String? {
        val arguments = JSONObject()
            .put("query", query)
            .put("search_results", searchResults.coerceIn(1, 20))
            .put("fetch_results", fetchResults.coerceIn(1, 5))
            .put("max_characters_per_page", 20000)
            .put("total_character_budget", 60000)
            .put("timeout_ms", 45000)
        return callBridgeText("search_and_fetch", arguments)
    }

    fun fetchUrl(url: String): String? {
        val arguments = JSONObject()
            .put("url", url)
            .put("output", "markdown")
            .put("max_characters", 20000)
            .put("include_metadata", true)
            .put("include_links", false)
            .put("timeout_ms", 15000)
        return callBridgeText("fetch_url", arguments)
    }

    private fun callBridge(tool: String, arguments: JSONObject): JSONObject? {
        return try {
            val payload = JSONObject()
                .put("tool", tool)
                .put("arguments", arguments)
            val connection = (URL(bridgeUrl).openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                connectTimeout = 5000
                readTimeout = 60000
                doOutput = true
                setRequestProperty("Accept", "application/json")
                setRequestProperty("Content-Type", "application/json; charset=utf-8")
            }
            try {
                connection.outputStream.use { it.write(payload.toString().toByteArray(StandardCharsets.UTF_8)) }
                if (connection.responseCode !in 200..299) return null
                val body = connection.inputStream.bufferedReader().use { it.readText() }
                val response = JSONObject(body)
                val text = extractText(response) ?: return null
                JSONObject(text)
            } finally {
                connection.disconnect()
            }
        } catch (_: Exception) {
            null
        }
    }

    private fun callBridgeText(tool: String, arguments: JSONObject): String? {
        return try {
            val payload = JSONObject()
                .put("tool", tool)
                .put("arguments", arguments)
            val connection = (URL(bridgeUrl).openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                connectTimeout = 5000
                readTimeout = 120000
                doOutput = true
                setRequestProperty("Accept", "application/json")
                setRequestProperty("Content-Type", "application/json; charset=utf-8")
            }
            try {
                connection.outputStream.use { it.write(payload.toString().toByteArray(StandardCharsets.UTF_8)) }
                if (connection.responseCode !in 200..299) return null
                extractText(JSONObject(connection.inputStream.bufferedReader().use { it.readText() }))
            } finally {
                connection.disconnect()
            }
        } catch (_: Exception) {
            null
        }
    }

    private fun extractText(response: JSONObject): String? {
        val direct = response.optString("text")
        if (direct.isNotBlank()) return direct
        val result = response.optJSONObject("result") ?: return null
        val content = result.optJSONArray("content") ?: return null
        val texts = buildString {
            for (i in 0 until content.length()) {
                val item = content.optJSONObject(i) ?: continue
                if (item.optString("type") == "text") {
                    if (isNotEmpty()) append('\n')
                    append(item.optString("text"))
                }
            }
        }
        return texts.ifBlank { null }
    }

    private fun parseResults(array: JSONArray, limit: Int): List<SearchResult> = buildList {
        for (i in 0 until minOf(array.length(), limit)) {
            val item = array.optJSONObject(i) ?: continue
            val title = item.optString("title").ifBlank { "Web result" }
            val url = item.optString("url")
            val snippet = item.optString("snippet").ifBlank { item.optString("description") }
            if (url.isNotBlank()) add(SearchResult(title, url, snippet))
        }
    }

    fun toLlmContext(results: List<SearchResult>): String {
        if (results.isEmpty()) return ""
        return buildString {
            append("<web_context trust=\"untrusted\">\n")
            results.forEachIndexed { index, result ->
                append("<source id=\"").append(index + 1).append("\">\n")
                append("<title>").append(result.title).append("</title>\n")
                append("<url>").append(result.url).append("</url>\n")
                append("<text>").append(result.snippet).append("</text>\n")
                append("</source>\n")
            }
            append("</web_context>")
        }
    }
}
