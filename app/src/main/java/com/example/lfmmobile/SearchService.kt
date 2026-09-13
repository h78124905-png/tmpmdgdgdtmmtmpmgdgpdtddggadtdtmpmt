package com.example.lfmmobile

import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.nio.charset.StandardCharsets

data class SearchResult(val title: String, val url: String, val snippet: String)

/**
 * Boundary for the web-search MCP service.
 * The app uses one built-in bridge endpoint; users do not configure an MCP URL.
 */
class SearchService {
    companion object {
        // Local MCP bridge endpoint owned by the app/runtime.
        const val DEFAULT_BRIDGE_URL = "http://127.0.0.1:8787/call"
    }

    fun search(query: String, limit: Int = 5): List<SearchResult> {
        val args = JSONObject()
            .put("query", query)
            .put("max_results", limit.coerceIn(1, 25))
            .put("region", "jp-jp")
            .put("language", "ja")
            .put("safe_search", "moderate")
        val root = callBridgeJson("web_search", args) ?: return emptyList()
        return parseResults(root.optJSONArray("results"), limit)
    }

    fun newsSearch(query: String, limit: Int = 5): List<SearchResult> {
        val args = JSONObject()
            .put("query", query)
            .put("max_results", limit.coerceIn(1, 25))
            .put("region", "jp-jp")
            .put("language", "ja")
            .put("time_range", "week")
        val root = callBridgeJson("news_search", args) ?: return emptyList()
        return parseResults(root.optJSONArray("results"), limit)
    }

    fun searchAndFetch(query: String, searchResults: Int = 8, fetchResults: Int = 3): String? {
        val args = JSONObject()
            .put("query", query)
            .put("search_results", searchResults.coerceIn(1, 20))
            .put("fetch_results", fetchResults.coerceIn(1, 5))
            .put("max_characters_per_page", 20000)
            .put("total_character_budget", 60000)
            .put("timeout_ms", 45000)
        return callBridgeText("search_and_fetch", args)
    }

    fun fetchUrl(url: String): String? {
        val args = JSONObject()
            .put("url", url)
            .put("output", "markdown")
            .put("max_characters", 20000)
            .put("include_metadata", true)
            .put("include_links", false)
            .put("timeout_ms", 15000)
        return callBridgeText("fetch_url", args)
    }

    private fun connection(): HttpURLConnection =
        (URL(DEFAULT_BRIDGE_URL).openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 5000
            readTimeout = 120000
            doOutput = true
            setRequestProperty("Accept", "application/json")
            setRequestProperty("Content-Type", "application/json; charset=utf-8")
        }

    private fun callBridgeJson(tool: String, arguments: JSONObject): JSONObject? = try {
        val c = connection()
        try {
            c.outputStream.use {
                it.write(JSONObject().put("tool", tool).put("arguments", arguments).toString().toByteArray(StandardCharsets.UTF_8))
            }
            if (c.responseCode !in 200..299) return null
            val response = JSONObject(c.inputStream.bufferedReader().use { it.readText() })
            val text = extractText(response) ?: return null
            JSONObject(text)
        } finally { c.disconnect() }
    } catch (_: Exception) { null }

    private fun callBridgeText(tool: String, arguments: JSONObject): String? = try {
        val c = connection()
        try {
            c.outputStream.use {
                it.write(JSONObject().put("tool", tool).put("arguments", arguments).toString().toByteArray(StandardCharsets.UTF_8))
            }
            if (c.responseCode !in 200..299) return null
            extractText(JSONObject(c.inputStream.bufferedReader().use { it.readText() }))
        } finally { c.disconnect() }
    } catch (_: Exception) { null }

    private fun extractText(response: JSONObject): String? {
        response.optString("text").takeIf { it.isNotBlank() }?.let { return it }
        val content = response.optJSONObject("result")?.optJSONArray("content") ?: return null
        return buildString {
            for (i in 0 until content.length()) {
                val item = content.optJSONObject(i) ?: continue
                if (item.optString("type") == "text") {
                    if (isNotEmpty()) append('\n')
                    append(item.optString("text"))
                }
            }
        }.ifBlank { null }
    }

    private fun parseResults(array: JSONArray?, limit: Int): List<SearchResult> = buildList {
        if (array == null) return@buildList
        for (i in 0 until minOf(array.length(), limit)) {
            val item = array.optJSONObject(i) ?: continue
            val url = item.optString("url")
            if (url.isNotBlank()) add(
                SearchResult(
                    item.optString("title").ifBlank { "Web result" },
                    url,
                    item.optString("snippet").ifBlank { item.optString("description") }
                )
            )
        }
    }

    fun toLlmContext(results: List<SearchResult>): String = buildString {
        if (results.isEmpty()) return@buildString
        append("<web_context trust=\"untrusted\">\n")
        results.forEachIndexed { i, r ->
            append("<source id=\"").append(i + 1).append("\">\n")
            append("<title>").append(r.title).append("</title>\n")
            append("<url>").append(r.url).append("</url>\n")
            append("<text>").append(r.snippet).append("</text>\n")
            append("</source>\n")
        }
        append("</web_context>")
    }
}
