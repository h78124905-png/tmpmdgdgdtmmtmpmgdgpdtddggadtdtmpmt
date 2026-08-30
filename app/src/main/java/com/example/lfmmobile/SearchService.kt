package com.example.lfmmobile

import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URLEncoder
import java.net.URL
import java.nio.charset.StandardCharsets

data class SearchResult(
    val title: String,
    val url: String,
    val snippet: String
)

class SearchService {
    companion object {
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

    fun search(query: String, limit: Int = 5): List<SearchResult> {
        val encoded = URLEncoder.encode(query, StandardCharsets.UTF_8.toString())
        val url = URL(
            "https://api.duckduckgo.com/?q=$encoded&format=json&no_html=1&no_redirect=1&skip_disambig=1"
        )
        val connection = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 8000
            readTimeout = 10000
            setRequestProperty("Accept", "application/json")
            setRequestProperty("User-Agent", "LfmMobile/1.0")
        }

        return try {
            if (connection.responseCode !in 200..299) return emptyList()
            val body = connection.inputStream.bufferedReader().use { it.readText() }
            parse(body, limit)
        } finally {
            connection.disconnect()
        }
    }

    private fun parse(body: String, limit: Int): List<SearchResult> {
        val root = JSONObject(body)
        val results = mutableListOf<SearchResult>()
        val abstractText = root.optString("AbstractText")
        val abstractUrl = root.optString("AbstractURL")
        val heading = root.optString("Heading")
        if (abstractText.isNotBlank() && abstractUrl.isNotBlank()) {
            results += SearchResult(heading.ifBlank { "DuckDuckGo Instant Answer" }, abstractUrl, abstractText)
        }

        fun addTopics(array: JSONArray) {
            for (i in 0 until array.length()) {
                if (results.size >= limit) return
                val item = array.optJSONObject(i) ?: continue
                val nested = item.optJSONArray("Topics")
                if (nested != null) {
                    addTopics(nested)
                    continue
                }
                val text = item.optString("Text")
                val firstUrl = item.optString("FirstURL")
                if (text.isNotBlank() && firstUrl.isNotBlank()) {
                    results += SearchResult(item.optString("Name").ifBlank { "Related result" }, firstUrl, text)
                }
            }
        }
        root.optJSONArray("RelatedTopics")?.let { addTopics(it) }
        return results.take(limit)
    }

    fun toLlmContext(results: List<SearchResult>): String {
        if (results.isEmpty()) return ""
        return buildString {
            append("<web_context>\n")
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
