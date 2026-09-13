package com.example.lfmmobile

import android.util.Log
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

data class WebFetchResult(val url: String, val title: String, val text: String, val statusCode: Int)

/** Direct Android web access. Search uses HTML endpoints with strict diagnostics and fallback. */
class WebSearchService {
    companion object { private const val TAG = "WebSearchService" }

    private val client = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(8, TimeUnit.SECONDS)
        .writeTimeout(5, TimeUnit.SECONDS)
        .followRedirects(true)
        .followSslRedirects(true)
        .build()

    suspend fun search(query: String, max: Int = 5): List<SearchResult> = withContext(Dispatchers.IO) {
        val limit = max.coerceIn(1, 8)
        val q = query.trim().take(1000)
        require(q.isNotBlank()) { "SEARCH_INVALID_QUERY" }

        val engines = listOf(
            "duckduckgo" to "https://html.duckduckgo.com/html/".toHttpUrl().newBuilder().addQueryParameter("q", q).build().toString(),
            "duckduckgo_lite" to "https://lite.duckduckgo.com/lite/".toHttpUrl().newBuilder().addQueryParameter("q", q).build().toString(),
            "bing" to "https://www.bing.com/search".toHttpUrl().newBuilder().addQueryParameter("q", q).build().toString()
        )
        var lastError = "SEARCH_EMPTY"
        for ((engine, url) in engines) {
            val response = runCatching { execute(url) }.getOrElse {
                lastError = "SEARCH_NETWORK_ERROR:$engine:${it.javaClass.simpleName}:${it.message ?: "unknown"}"
                Log.e(TAG, lastError, it)
                continue
            }
            if (response == null) {
                lastError = "SEARCH_NO_RESPONSE:$engine"
                continue
            }
            Log.i(TAG, "[$engine] HTTP ${response.statusCode} contentType=${response.contentType} bytes=${response.body.length} finalUrl=${response.finalUrl}")
            Log.d(TAG, "[$engine] bodyPrefix=${response.body.take(300).replace("\n", " ")}")
            if (response.statusCode !in 200..299) {
                lastError = "SEARCH_HTTP_ERROR:$engine:${response.statusCode}"
                continue
            }
            if (response.contentType.isNotBlank() && !response.contentType.lowercase(Locale.ROOT).contains("html")) {
                lastError = "SEARCH_CONTENT_TYPE_ERROR:$engine:${response.contentType}"
                continue
            }
            val parsed = when (engine) {
                "duckduckgo", "duckduckgo_lite" -> parseDdg(response.body)
                else -> parseBing(response.body)
            }
            Log.i(TAG, "[$engine] parsedResults=${parsed.size}")
            if (parsed.isNotEmpty()) return@withContext deduplicate(parsed).take(limit)
            lastError = "SEARCH_PARSE_EMPTY:$engine"
        }
        throw IllegalStateException(lastError)
    }

    suspend fun fetchUrl(url: String): WebFetchResult = withContext(Dispatchers.IO) {
        val normalized = normalizeUrl(url) ?: throw IllegalArgumentException("FETCH_INVALID_URL")
        val response = execute(normalized) ?: throw IllegalStateException("FETCH_NO_RESPONSE")
        Log.i(TAG, "[fetch] HTTP ${response.statusCode} contentType=${response.contentType} bytes=${response.body.length} finalUrl=${response.finalUrl}")
        if (response.statusCode !in 200..299) throw IllegalStateException("FETCH_HTTP_ERROR:${response.statusCode}")
        if (!response.contentType.lowercase(Locale.ROOT).contains("html")) throw IllegalStateException("FETCH_CONTENT_TYPE_ERROR:${response.contentType}")
        val doc = Jsoup.parse(response.body, response.finalUrl)
        doc.select("script,style,noscript,svg,nav,footer,header,form").remove()
        val title = doc.title().trim()
        val text = doc.body()?.text()?.replace(Regex("\\s+"), " ")?.trim().orEmpty().take(12000)
        if (text.isBlank()) throw IllegalStateException("FETCH_EMPTY_BODY")
        WebFetchResult(response.finalUrl, title, text, response.statusCode)
    }

    private data class HttpResponse(val statusCode: Int, val contentType: String, val finalUrl: String, val body: String)

    private fun execute(url: String): HttpResponse? {
        val request = Request.Builder().url(url)
            .header("User-Agent", "Mozilla/5.0 (Linux; Android 16) AppleWebKit/537.36 Chrome/128 Mobile Safari/537.36")
            .header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
            .header("Accept-Language", "ja,en;q=0.8")
            .build()
        client.newCall(request).execute().use { response ->
            val body = response.body?.string() ?: return null
            return HttpResponse(response.code, response.header("Content-Type", "").orEmpty(), response.request.url.toString(), body)
        }
    }

    private fun parseDdg(html: String): List<SearchResult> = Jsoup.parse(html).select(".result, tr.result, .result-link").mapNotNull { result ->
        val link = result.selectFirst(".result__a, a.result-link, a[href^=http]") ?: return@mapNotNull null
        val title = link.text().trim(); val url = normalizeUrl(link.attr("href")) ?: return@mapNotNull null
        if (title.isBlank()) null else SearchResult(title, url, result.selectFirst(".result__snippet")?.text()?.trim().orEmpty().take(300))
    }

    private fun parseBing(html: String): List<SearchResult> = Jsoup.parse(html).select("li.b_algo").mapNotNull { result ->
        val link = result.selectFirst("h2 a") ?: return@mapNotNull null
        val title = link.text().trim(); val url = normalizeUrl(link.attr("href")) ?: return@mapNotNull null
        if (title.isBlank()) null else SearchResult(title, url, result.selectFirst(".b_caption p")?.text()?.trim().orEmpty().take(300))
    }

    private fun normalizeUrl(raw: String): String? {
        val value = raw.trim(); if (value.isBlank()) return null
        val candidate = if (value.startsWith("//")) "https:$value" else value
        val uri = runCatching { URI(candidate) }.getOrNull() ?: return null
        return if (uri.scheme?.lowercase(Locale.ROOT) in listOf("http", "https") && !uri.host.isNullOrBlank()) uri.toString() else null
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
        results.forEachIndexed { i, r -> append(i + 1).append(". **").append(r.title).append("**\n").append(r.url).append('\n').append(r.snippet.take(300)).append('\n') }
        append("</tool_result>")
    }

    fun formatFetchResult(result: WebFetchResult): String = buildString {
        append("<tool_result name=\"fetch_url\" trust=\"untrusted\">\n")
        append("URL: ").append(result.url).append('\n')
        if (result.title.isNotBlank()) append("Title: ").append(result.title).append('\n')
        append(result.text.take(4000)).append('\n')
        append("</tool_result>")
    }
}

typealias SearchService = WebSearchService
