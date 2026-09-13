package com.example.lfmmobile

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.OkHttpClient
import okhttp3.Request
import org.jsoup.Jsoup
import java.net.URI
import java.util.Locale
import java.util.concurrent.TimeUnit

data class SearchResult(val title: String, val url: String, val snippet: String)

/** Direct HTML web search for Phase 2. No MCP/Node/bridge is used. */
class WebSearchService {
    private val client = OkHttpClient.Builder()
        .connectTimeout(8, TimeUnit.SECONDS).readTimeout(12, TimeUnit.SECONDS).writeTimeout(8, TimeUnit.SECONDS)
        .followRedirects(true).followSslRedirects(true).build()

    suspend fun search(query: String, max: Int = 5): List<SearchResult> = withContext(Dispatchers.IO) {
        val limit = max.coerceIn(1, 8)
        val q = query.trim().take(1000)
        if (q.isBlank()) return@withContext emptyList()
        val ddg = runCatching { fetchDdg(q) }.getOrNull()
        val ddgResults = ddg?.let { parseDdg(it) }.orEmpty()
        if (ddgResults.isNotEmpty()) return@withContext deduplicate(ddgResults).take(limit)
        val bing = runCatching { fetchBing(q) }.getOrNull()
        deduplicate(bing?.let { parseBing(it) }.orEmpty()).take(limit)
    }

    private fun fetchDdg(query: String): String? = execute("https://html.duckduckgo.com/html/".toHttpUrl().newBuilder().addQueryParameter("q", query).build().toString())
    private fun fetchBing(query: String): String? = execute("https://www.bing.com/search".toHttpUrl().newBuilder().addQueryParameter("q", query).build().toString())

    private fun execute(url: String): String? {
        val request = Request.Builder().url(url)
            .header("User-Agent", "Mozilla/5.0 (Android; Mobile) AppleWebKit/537.36 Chrome/128 Mobile Safari/537.36")
            .header("Accept", "text/html,application/xhtml+xml").header("Accept-Language", "ja,en;q=0.8").build()
        client.newCall(request).execute().use { response -> if (!response.isSuccessful) return null; return response.body?.string() }
    }

    private fun parseDdg(html: String): List<SearchResult> = Jsoup.parse(html).select(".result").mapNotNull { result ->
        val link = result.selectFirst(".result__a") ?: return@mapNotNull null
        val title = link.text().trim(); val url = normalizeUrl(link.attr("href")) ?: return@mapNotNull null
        if (title.isBlank()) null else SearchResult(title, url, result.selectFirst(".result__snippet")?.text()?.trim().orEmpty().take(150))
    }

    private fun parseBing(html: String): List<SearchResult> = Jsoup.parse(html).select("li.b_algo").mapNotNull { result ->
        val link = result.selectFirst("h2 a") ?: return@mapNotNull null
        val title = link.text().trim(); val url = normalizeUrl(link.attr("href")) ?: return@mapNotNull null
        if (title.isBlank()) null else SearchResult(title, url, result.selectFirst(".b_caption p")?.text()?.trim().orEmpty().take(150))
    }

    private fun normalizeUrl(raw: String): String? {
        val value = raw.trim(); if (value.isBlank()) return null
        val candidate = if (value.startsWith("//")) "https:$value" else value
        val uri = runCatching { URI(candidate) }.getOrNull() ?: return null
        return if (uri.scheme in listOf("http", "https") && !uri.host.isNullOrBlank()) uri.toString() else null
    }

    private fun deduplicate(results: List<SearchResult>): List<SearchResult> {
        val accepted = mutableListOf<SearchResult>()
        for (r in results) {
            val host = runCatching { URI(r.url).host?.lowercase(Locale.ROOT) }.getOrNull() ?: continue
            if (accepted.any { e -> runCatching { URI(e.url).host?.lowercase(Locale.ROOT) }.getOrNull() == host && titleSimilarity(e.title, r.title) >= 0.78 }) continue
            accepted += r
        }
        return accepted
    }

    private fun titleSimilarity(a: String, b: String): Double {
        val x = normalizeTitle(a); val y = normalizeTitle(b)
        if (x == y) return 1.0; if (x.isBlank() || y.isBlank()) return 0.0
        return 1.0 - levenshtein(x, y).toDouble() / maxOf(x.length, y.length)
    }
    private fun normalizeTitle(v: String) = v.lowercase(Locale.ROOT).replace(Regex("\\s+"), "").replace(Regex("[\\p{Punct}\\p{Symbol}]"), "")
    private fun levenshtein(a: String, b: String): Int {
        if (a == b) return 0; if (a.isEmpty()) return b.length; if (b.isEmpty()) return a.length
        var prev = IntArray(b.length + 1) { it }
        for (i in a.indices) { val cur = IntArray(b.length + 1); cur[0] = i + 1; for (j in b.indices) cur[j + 1] = minOf(cur[j] + 1, prev[j + 1] + 1, prev[j] + if (a[i] == b[j]) 0 else 1); prev = cur }
        return prev[b.length]
    }

    fun formatToolResult(results: List<SearchResult>): String = buildString {
        append("<tool_result name=\"web_search\" trust=\"untrusted\">\n")
        if (results.isEmpty()) append("No search results found.\n")
        results.forEachIndexed { i, r -> append(i + 1).append(". **").append(r.title).append("**\n").append(r.url).append('\n').append(r.snippet.take(150)).append('\n') }
        append("</tool_result>")
    }
}

/** Source-compatible alias for the current direct service; no bridge or MCP is involved. */
typealias SearchService = WebSearchService
