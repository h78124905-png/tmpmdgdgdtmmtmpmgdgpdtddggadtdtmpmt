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
        .connectTimeout(8, TimeUnit.SECONDS)
        .readTimeout(12, TimeUnit.SECONDS)
        .writeTimeout(8, TimeUnit.SECONDS)
        .followRedirects(true)
        .followSslRedirects(true)
        .build()

    suspend fun search(query: String, max: Int = 5): List<SearchResult> = withContext(Dispatchers.IO) {
        val limit = max.coerceIn(1, 8)
        val normalizedQuery = query.trim().take(1000)
        if (normalizedQuery.isBlank()) return@withContext emptyList()

        val ddg = runCatching { fetchDdg(normalizedQuery) }.getOrNull()
        val results = ddg?.let { parseDdg(it) }.orEmpty()
        if (results.isNotEmpty()) return@withContext deduplicate(results).take(limit)

        val bing = runCatching { fetchBing(normalizedQuery) }.getOrNull()
        val bingResults = bing?.let { parseBing(it) }.orEmpty()
        deduplicate(bingResults).take(limit)
    }

    private fun fetchDdg(query: String): String? {
        val url = "https://html.duckduckgo.com/html/".toHttpUrl().newBuilder()
            .addQueryParameter("q", query)
            .build()
        return execute(url.toString(), "Mozilla/5.0 (Android; Mobile) AppleWebKit/537.36 Chrome/128 Mobile Safari/537.36")
    }

    private fun fetchBing(query: String): String? {
        val url = "https://www.bing.com/search".toHttpUrl().newBuilder()
            .addQueryParameter("q", query)
            .build()
        return execute(url.toString(), "Mozilla/5.0 (Android; Mobile) AppleWebKit/537.36 Chrome/128 Mobile Safari/537.36")
    }

    private fun execute(url: String, userAgent: String): String? {
        val request = Request.Builder()
            .url(url)
            .header("User-Agent", userAgent)
            .header("Accept", "text/html,application/xhtml+xml")
            .header("Accept-Language", "ja,en;q=0.8")
            .build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) return null
            return response.body?.string()
        }
    }

    private fun parseDdg(html: String): List<SearchResult> {
        val doc = Jsoup.parse(html)
        return doc.select(".result").mapNotNull { result ->
            val link = result.selectFirst(".result__a") ?: return@mapNotNull null
            val title = link.text().trim()
            val url = normalizeUrl(link.attr("href")) ?: return@mapNotNull null
            val snippet = result.selectFirst(".result__snippet")?.text()?.trim().orEmpty()
            if (title.isBlank()) null else SearchResult(title, url, snippet.take(150))
        }
    }

    private fun parseBing(html: String): List<SearchResult> {
        val doc = Jsoup.parse(html)
        return doc.select("li.b_algo").mapNotNull { result ->
            val link = result.selectFirst("h2 a") ?: return@mapNotNull null
            val title = link.text().trim()
            val url = normalizeUrl(link.attr("href")) ?: return@mapNotNull null
            val snippet = result.selectFirst(".b_caption p")?.text()?.trim().orEmpty()
            if (title.isBlank()) null else SearchResult(title, url, snippet.take(150))
        }
    }

    private fun normalizeUrl(raw: String): String? {
        val value = raw.trim()
        if (value.isBlank()) return null
        val candidate = if (value.startsWith("//")) "https:$value" else value
        val uri = runCatching { URI(candidate) }.getOrNull() ?: return null
        if (uri.scheme !in listOf("http", "https") || uri.host.isNullOrBlank()) return null
        return uri.toString()
    }

    private fun deduplicate(results: List<SearchResult>): List<SearchResult> {
        val accepted = mutableListOf<SearchResult>()
        for (result in results) {
            val host = runCatching { URI(result.url).host?.lowercase(Locale.ROOT) }.getOrNull() ?: continue
            if (accepted.any { existing ->
                    val existingHost = runCatching { URI(existing.url).host?.lowercase(Locale.ROOT) }.getOrNull()
                    existingHost == host && titleSimilarity(existing.title, result.title) >= 0.78
                }) continue
            accepted += result
        }
        return accepted
    }

    private fun titleSimilarity(a: String, b: String): Double {
        val x = normalizeTitle(a)
        val y = normalizeTitle(b)
        if (x == y) return 1.0
        if (x.isBlank() || y.isBlank()) return 0.0
        val distance = levenshtein(x, y)
        return 1.0 - distance.toDouble() / maxOf(x.length, y.length)
    }

    private fun normalizeTitle(value: String): String = value.lowercase(Locale.ROOT)
        .replace(Regex("\\s+"), "")
        .replace(Regex("[\\p{Punct}\\p{Symbol}]"), "")

    private fun levenshtein(a: String, b: String): Int {
        if (a == b) return 0
        if (a.isEmpty()) return b.length
        if (b.isEmpty()) return a.length
        var previous = IntArray(b.length + 1) { it }
        for (i in a.indices) {
            val current = IntArray(b.length + 1)
            current[0] = i + 1
            for (j in b.indices) {
                current[j + 1] = minOf(
                    current[j] + 1,
                    previous[j + 1] + 1,
                    previous[j] + if (a[i] == b[j]) 0 else 1
                )
            }
            previous = current
        }
        return previous[b.length]
    }

    fun formatToolResult(results: List<SearchResult>): String = buildString {
        append("<tool_result name=\"web_search\" trust=\"untrusted\">\n")
        if (results.isEmpty()) {
            append("No search results found.\n")
        } else {
            results.forEachIndexed { index, result ->
                append(index + 1).append(". **").append(result.title).append("**\n")
                append(result.url).append('\n')
                if (result.snippet.isNotBlank()) append(result.snippet.take(150)).append('\n')
            }
        }
        append("</tool_result>")
    }
}
