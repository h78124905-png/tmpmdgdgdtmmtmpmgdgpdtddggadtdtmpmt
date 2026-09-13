package com.example.lfmmobile

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

data class AgentResult(val answer: String, val thinking: String, val sources: List<SearchResult>, val error: String = "")

class ToolAgent(private val engine: LlamaEngine, private val search: SearchService) {
    companion object {
        private const val MAX_TOOL_CALLS = 4

        fun toolDefinitions(): JSONArray = JSONArray().apply {
            put(functionTool("web_search", "Search the web for current or factual information.", JSONObject().put("type", "object").put("properties", JSONObject().put("query", stringSchema("Search query")).put("max_results", integerSchema(1, 8))).put("required", JSONArray().put("query"))))
            put(functionTool("news_search", "Search recent news and current events.", JSONObject().put("type", "object").put("properties", JSONObject().put("query", stringSchema("News search query")).put("max_results", integerSchema(1, 8))).put("required", JSONArray().put("query"))))
            put(functionTool("search_and_fetch", "Search the web and fetch the most relevant pages when page content is needed.", JSONObject().put("type", "object").put("properties", JSONObject().put("query", stringSchema("Search query")).put("search_results", integerSchema(1, 8)).put("fetch_results", integerSchema(1, 3))).put("required", JSONArray().put("query"))))
            put(functionTool("fetch_url", "Fetch a specific public URL.", JSONObject().put("type", "object").put("properties", JSONObject().put("url", stringSchema("Public URL")).put("required", JSONArray().put("url"))))
        }

        private fun functionTool(name: String, description: String, parameters: JSONObject): JSONObject = JSONObject().put("type", "function").put("function", JSONObject().put("name", name).put("description", description).put("parameters", parameters))
        private fun stringSchema(description: String): JSONObject = JSONObject().put("type", "string").put("description", description)
        private fun integerSchema(min: Int, max: Int): JSONObject = JSONObject().put("type", "integer").put("minimum", min).put("maximum", max)
    }

    suspend fun run(initialMessages: JSONArray, maxTokens: Int): AgentResult = withContext(Dispatchers.Default) {
        val messages = initialMessages
        val tools = toolDefinitions()
        val sources = mutableListOf<SearchResult>()
        var thinking = ""

        repeat(MAX_TOOL_CALLS + 1) {
            val raw = engine.generateToolStep(messages.toString(), tools.toString(), maxTokens.coerceAtMost(1024))
            val step = try { JSONObject(raw) } catch (e: Exception) { return@withContext AgentResult("", thinking, sources, "Invalid native tool-step response: ${e.message}") }
            when (step.optString("type")) {
                "final" -> return@withContext AgentResult(step.optString("content"), thinking + step.optString("reasoning"), sources)
                "error" -> return@withContext AgentResult("", thinking, sources, step.optString("error", "tool generation failed"))
                "tool_calls" -> {
                    val calls = step.optJSONArray("calls") ?: return@withContext AgentResult("", thinking, sources, "Tool call list missing")
                    thinking += step.optString("reasoning")
                    if (calls.length() == 0 || calls.length() > MAX_TOOL_CALLS) return@withContext AgentResult("", thinking, sources, "Invalid number of tool calls")
                    val assistantCalls = JSONArray()
                    for (i in 0 until calls.length()) {
                        val c = calls.optJSONObject(i) ?: return@withContext AgentResult("", thinking, sources, "Invalid tool call")
                        val name = c.optString("name")
                        val argsText = c.optString("arguments")
                        val id = c.optString("id").ifBlank { "call_${i + 1}" }
                        val args = try { JSONObject(argsText) } catch (_: Exception) { return@withContext AgentResult("", thinking, sources, "Invalid arguments for $name") }
                        if (!allowed(name, args)) return@withContext AgentResult("", thinking, sources, "Rejected tool call: $name")
                        assistantCalls.put(JSONObject().put("id", id).put("type", "function").put("function", JSONObject().put("name", name).put("arguments", args.toString())))
                    }
                    messages.put(JSONObject().put("role", "assistant").put("tool_calls", assistantCalls))
                    for (i in 0 until calls.length()) {
                        val c = calls.getJSONObject(i)
                        val name = c.getString("name")
                        val args = JSONObject(c.getString("arguments"))
                        val id = c.optString("id").ifBlank { "call_${i + 1}" }
                        val result = withContext(Dispatchers.IO) { execute(name, args) }
                        sources += result.sources
                        val fenced = "<tool_result name=\"$name\" trust=\"untrusted\">\n${result.text.take(60000)}\n</tool_result>"
                        messages.put(JSONObject().put("role", "tool").put("tool_call_id", id).put("tool_name", name).put("content", fenced))
                    }
                }
                else -> return@withContext AgentResult("", thinking, sources, "Unknown native tool-step type")
            }
        }
        AgentResult("", thinking, sources, "Tool-call limit reached")
    }

    private fun allowed(name: String, args: JSONObject): Boolean = when (name) {
        "web_search", "news_search", "search_and_fetch" -> args.optString("query").isNotBlank() && args.optString("query").length <= 1000
        "fetch_url" -> { val u = args.optString("url"); (u.startsWith("https://") || u.startsWith("http://")) && u.length <= 4096 }
        else -> false
    }

    private data class Exec(val text: String, val sources: List<SearchResult> = emptyList())

    private fun execute(name: String, args: JSONObject): Exec = when (name) {
        "web_search" -> { val r = search.search(args.getString("query"), args.optInt("max_results", 5)); Exec(search.toLlmContext(r), r) }
        "news_search" -> { val r = search.newsSearch(args.getString("query"), args.optInt("max_results", 5)); Exec(search.toLlmContext(r), r) }
        "search_and_fetch" -> Exec(search.searchAndFetch(args.getString("query"), args.optInt("search_results", 8), args.optInt("fetch_results", 3)) ?: "<error>web search failed</error>")
        "fetch_url" -> Exec(search.fetchUrl(args.getString("url")) ?: "<error>URL fetch failed</error>")
        else -> Exec("<error>unknown tool</error>")
    }
}
