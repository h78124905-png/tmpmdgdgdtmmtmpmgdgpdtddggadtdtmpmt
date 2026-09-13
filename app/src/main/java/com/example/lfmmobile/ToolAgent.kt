package com.example.lfmmobile

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

data class AgentResult(val answer: String, val thinking: String, val sources: List<SearchResult>, val error: String = "", val timing: String = "")
data class ToolCallTrace(val name: String, val arguments: String, val id: String)

class ToolAgent(private val engine: LlamaEngine, private val webSearch: WebSearchService = WebSearchService()) {
    companion object {
        private const val TAG = "ToolAgent"
        private const val MAX_TOOL_CALLS = 3
        private const val MAX_HISTORY_MESSAGES = 6
        private const val MAX_MESSAGE_CHARS = 2400
        fun toolDefinitions(): JSONArray = JSONArray().apply {
            put(functionTool("web_search", "Search the web for current or factual information.", JSONObject().put("type", "object").put("properties", JSONObject().put("query", stringSchema("Search query")).put("max_results", integerSchema(1, 8))).put("required", JSONArray().put("query"))))
            put(functionTool("fetch_url", "Fetch and read a specific public HTTP or HTTPS URL.", JSONObject().put("type", "object").put("properties", JSONObject().put("url", stringSchema("Public URL"))).put("required", JSONArray().put("url"))))
        }
        private fun functionTool(name: String, description: String, parameters: JSONObject) = JSONObject().put("type", "function").put("function", JSONObject().put("name", name).put("description", description).put("parameters", parameters))
        private fun stringSchema(description: String) = JSONObject().put("type", "string").put("description", description)
        private fun integerSchema(min: Int, max: Int) = JSONObject().put("type", "integer").put("minimum", min).put("maximum", max)
    }

    suspend fun run(initialMessages: JSONArray, maxTokens: Int, onProgress: (String, Long) -> Unit = { _, _ -> }, onToken: (String) -> Unit = {}): AgentResult = withContext(Dispatchers.Default) {
        val messages = compactMessages(initialMessages); val tools = toolDefinitions(); val sources = mutableListOf<SearchResult>(); val seenCalls = mutableSetOf<String>()
        var thinking = ""; var toolCallsUsed = 0; var iterations = 0; var lastTiming = ""
        while (iterations <= MAX_TOOL_CALLS) {
            iterations++
            val raw = try { onProgress("tool_step_start", 0L); engine.generateToolStep(messages.toString(), tools.toString(), maxTokens.coerceAtMost(384), onProgress, onToken) } catch (e: Exception) {
                Log.e(TAG, "native tool-step failed", e); return@withContext AgentResult("", thinking, sources, e.message ?: e::class.java.simpleName, lastTiming)
            }
            val step = try { JSONObject(raw) } catch (e: Exception) { return@withContext AgentResult("", thinking, sources, "Invalid native tool-step response: ${e.message}", lastTiming) }
            step.optJSONObject("timing")?.let { lastTiming = it.toString(); Log.i(TAG, "[tool][timing] $lastTiming") }
            when (step.optString("type")) {
                "final" -> {
                    val rawContent = step.optString("content")
                    val rawCall = Regex("<\\|tool_call_start\\|>\\s*\\[\\s*web_search\\(query=['\"](.*?)['\"](?:,\\s*max_results=([0-9]+))?\\)\\s*\\]\\s*<\\|tool_call_end\\|>", RegexOption.DOT_MATCHES_ALL).find(rawContent)
                    if (rawCall != null) {
                        val query = rawCall.groupValues[1].trim()
                        val maxResults = rawCall.groupValues.getOrNull(2)?.toIntOrNull()?.coerceIn(1, 8) ?: 5
                        val found = webSearch.search(query, maxResults)
                        sources += found
                        messages.put(JSONObject().put("role", "assistant").put("tool_calls", JSONArray().put(JSONObject().put("id", "raw_call_1").put("type", "function").put("function", JSONObject().put("name", "web_search").put("arguments", JSONObject().put("query", query).put("max_results", maxResults).toString())))))
                        messages.put(JSONObject().put("role", "tool").put("tool_call_id", "raw_call_1").put("tool_name", "web_search").put("content", webSearch.formatToolResult(found)))
                        continue
                    }
                    return@withContext AgentResult(cleanModelText(rawContent), thinking + cleanModelText(step.optString("reasoning")), sources, timing = lastTiming)
                }
                "error" -> return@withContext AgentResult("", thinking, sources, step.optString("error", "tool generation failed"), lastTiming)
                "tool_calls" -> {
                    val calls = step.optJSONArray("calls") ?: return@withContext AgentResult("", thinking, sources, "Tool call list missing", lastTiming)
                    thinking += step.optString("reasoning"); if (calls.length() == 0) return@withContext AgentResult("", thinking, sources, "Empty tool-call list", lastTiming)
                    if (toolCallsUsed + calls.length() > MAX_TOOL_CALLS) { messages.put(JSONObject().put("role", "system").put("content", "Tool-call limit reached. Do not call any more tools. Give the best final answer using the information already available.")); continue }
                    val assistantCalls = JSONArray(); val parsedCalls = mutableListOf<ToolCallTrace>()
                    for (i in 0 until calls.length()) {
                        val c = calls.optJSONObject(i) ?: return@withContext AgentResult("", thinking, sources, "Invalid tool call", lastTiming)
                        val name = c.optString("name"); val argsText = c.optString("arguments"); val id = c.optString("id").ifBlank { "call_${toolCallsUsed + i + 1}" }
                        val args = try { JSONObject(argsText) } catch (_: Exception) { return@withContext AgentResult("", thinking, sources, "Invalid arguments for $name", lastTiming) }
                        if (!allowed(name, args)) return@withContext AgentResult("", thinking, sources, "Rejected tool call: $name", lastTiming)
                        val normalizedArgs = args.toString(); val duplicateKey = "$name\u0000$normalizedArgs"
                        if (!seenCalls.add(duplicateKey)) { messages.put(JSONObject().put("role", "system").put("content", "The same tool call was already attempted. Do not repeat it. Give the best final answer.")); continue }
                        parsedCalls += ToolCallTrace(name, normalizedArgs, id)
                        assistantCalls.put(JSONObject().put("id", id).put("type", "function").put("function", JSONObject().put("name", name).put("arguments", normalizedArgs)))
                    }
                    if (parsedCalls.isEmpty()) continue
                    messages.put(JSONObject().put("role", "assistant").put("tool_calls", assistantCalls)); toolCallsUsed += parsedCalls.size
                    for (call in parsedCalls) {
                        val args = JSONObject(call.arguments)
                        val result = try {
                            when (call.name) {
                                "web_search" -> { val found = webSearch.search(args.optString("query"), args.optInt("max_results", 5).coerceIn(1, 8)); sources += found; webSearch.formatToolResult(found) }
                                "fetch_url" -> webSearch.formatFetchResult(webSearch.fetchUrl(args.optString("url")))
                                else -> "Unknown tool"
                            }
                        } catch (e: Exception) {
                            val message = e.message ?: e::class.java.simpleName; Log.e(TAG, "tool execution failed name=${call.name}: $message", e); "<tool_error name=\"${call.name}\">$message</tool_error>"
                        }
                        messages.put(JSONObject().put("role", "tool").put("tool_call_id", call.id).put("tool_name", call.name).put("content", result))
                    }
                }
                else -> return@withContext AgentResult("", thinking, sources, "Unknown native tool-step type", lastTiming)
            }
        }
        AgentResult("", thinking, sources, "Tool-call loop exhausted", lastTiming)
    }

    private fun compactMessages(input: JSONArray): JSONArray {
        val output = JSONArray()
        if (input.length() == 0) return output
        output.put(input.optJSONObject(0) ?: JSONObject().put("role", "system").put("content", "Use tools when needed."))
        val start = maxOf(1, input.length() - MAX_HISTORY_MESSAGES + 1)
        for (i in start until input.length()) {
            val message = input.optJSONObject(i) ?: continue
            val copy = JSONObject(message.toString())
            if (copy.has("content") && !copy.isNull("content")) {
                val content = copy.optString("content")
                if (content.length > MAX_MESSAGE_CHARS) copy.put("content", content.take(MAX_MESSAGE_CHARS) + "…")
            }
            output.put(copy)
        }
        return output
    }

    private fun cleanModelText(value: String): String = value
        .replace(Regex("<think>.*?</think>", RegexOption.DOT_MATCHES_ALL), "")
        .replace(Regex("<\\|think\\|>.*?<\\|/think\\|>", RegexOption.DOT_MATCHES_ALL), "")
        .replace("<|im_start|>assistant", "")
        .replace("<|im_end|>", "")
        .trim()

    private fun allowed(name: String, args: JSONObject): Boolean = when (name) {
        "web_search" -> args.optString("query").isNotBlank() && args.optString("query").length <= 1000
        "fetch_url" -> { val u = args.optString("url"); (u.startsWith("https://") || u.startsWith("http://")) && u.length <= 4096 }
        else -> false
    }
}
