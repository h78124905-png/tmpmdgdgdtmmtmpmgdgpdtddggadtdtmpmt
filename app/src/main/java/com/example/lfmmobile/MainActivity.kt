package com.example.lfmmobile

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID

private data class Message(val user: Boolean, val text: String, val thinking: String = "", val sources: List<SearchResult> = emptyList())
private data class ModelSlot(val uri: String = "", val name: String = "", val storedPath: String = "")
private data class Conversation(val id: String, val title: String, val messages: List<Message>)
private data class GenerationStats(val tokPerSec: Double = 0.0, val elapsedMs: Long = 0L, val contextUsed: Int = 0, val contextSize: Int = 0)
private sealed interface StreamEvent { data class Token(val text: String) : StreamEvent; data class Stats(val value: GenerationStats) : StreamEvent }

private class ThinkStreamParser {
    private enum class Mode { UNKNOWN, THINKING, ANSWERING }
    private var mode = Mode.UNKNOWN
    private var pending = ""
    data class Emission(val thinking: String = "", val answer: String = "")
    private val opening = listOf("<think>", "<|think|>", "<|thinking|>", "<|assistant_thinking|>")
    private val closing = listOf("</think>", "</thinking>", "<|/think|>", "<|/thinking|>", "<|end_think|>", "<|end_thinking|>")
    private fun earliest(s: String, markers: List<String>): Pair<Int, String>? {
        var best: Pair<Int, String>? = null
        for (m in markers) { val i = s.indexOf(m); if (i >= 0 && (best == null || i < best!!.first)) best = i to m }
        return best
    }
    private fun partial(s: String, markers: List<String>): Int {
        var keep = 0
        for (m in markers) for (n in 1 until m.length) if (s.endsWith(m.substring(0, n))) keep = maxOf(keep, n)
        return keep
    }
    fun consume(chunk: String): Emission {
        pending += chunk
        var thinking = ""
        var answer = ""
        while (pending.isNotEmpty()) {
            val o = earliest(pending, opening)
            val c = earliest(pending, closing)
            when (mode) {
                Mode.UNKNOWN -> when {
                    o != null && (c == null || o.first <= c.first) -> { if (o.first > 0) answer += pending.substring(0, o.first); pending = pending.substring(o.first + o.second.length); mode = Mode.THINKING }
                    c != null -> { if (c.first > 0) thinking += pending.substring(0, c.first); pending = pending.substring(c.first + c.second.length); mode = Mode.ANSWERING }
                    else -> break
                }
                Mode.THINKING -> if (c != null) { if (c.first > 0) thinking += pending.substring(0, c.first); pending = pending.substring(c.first + c.second.length); mode = Mode.ANSWERING } else { val n = pending.length - partial(pending, closing); if (n > 0) { thinking += pending.substring(0, n); pending = pending.substring(n) }; break }
                Mode.ANSWERING -> if (o != null) { if (o.first > 0) answer += pending.substring(0, o.first); pending = pending.substring(o.first + o.second.length); mode = Mode.THINKING } else { val n = pending.length - partial(pending, opening); if (n > 0) { answer += pending.substring(0, n); pending = pending.substring(n) }; break }
            }
        }
        return Emission(thinking, answer)
    }
    fun finish(): Emission { val r = pending; pending = ""; return if (mode == Mode.THINKING) Emission(thinking = r) else Emission(answer = r) }
}

private fun loadConversations(c: Context): List<Conversation> = try {
    val a = JSONArray(c.getSharedPreferences("chat_history", Context.MODE_PRIVATE).getString("conversations", "[]") ?: "[]")
    buildList { for (i in 0 until a.length()) { val o = a.getJSONObject(i); val mj = o.optJSONArray("messages") ?: JSONArray(); add(Conversation(o.optString("id"), o.optString("title"), buildList { for (j in 0 until mj.length()) { val m = mj.getJSONObject(j); add(Message(m.optBoolean("user"), m.optString("text"), m.optString("thinking"))) } })) } }
} catch (_: Exception) { emptyList() }

private fun saveConversations(c: Context, cs: List<Conversation>) {
    val a = JSONArray()
    cs.forEach { v -> val m = JSONArray(); v.messages.forEach { x -> m.put(JSONObject().put("user", x.user).put("text", x.text).put("thinking", x.thinking)) }; a.put(JSONObject().put("id", v.id).put("title", v.title).put("messages", m)) }
    c.getSharedPreferences("chat_history", Context.MODE_PRIVATE).edit().putString("conversations", a.toString()).apply()
}

class MainActivity : ComponentActivity() {
    override fun onCreate(b: Bundle?) { super.onCreate(b); setContent { ChatApp() } }
    fun displayName(u: Uri): String? { contentResolver.query(u, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)?.use { if (it.moveToFirst()) return it.getString(0) }; return null }
}

@Composable private fun ChatApp() {
    val activity = LocalContext.current as MainActivity
    val engine = remember { LlamaEngine() }
    val search = remember { SearchService() }
    val scope = rememberCoroutineScope()
    val list = rememberLazyListState()
    val drawer = rememberDrawerState(DrawerValue.Closed)
    var target by remember { mutableStateOf(ModelSlot()) }
    var draft by remember { mutableStateOf(ModelSlot()) }
    var draftEnabled by remember { mutableStateOf(false) }
    var stored by remember { mutableStateOf(emptyList<File>()) }
    var messages by remember { mutableStateOf(listOf<Message>()) }
    var conversations by remember { mutableStateOf(loadConversations(activity)) }
    var chatId by remember { mutableStateOf(UUID.randomUUID().toString()) }
    var prompt by remember { mutableStateOf("") }
    var generating by remember { mutableStateOf(false) }
    var searching by remember { mutableStateOf(false) }
    var showModels by remember { mutableStateOf(false) }
    var contextSize by remember { mutableIntStateOf(2048) }
    var maxTokens by remember { mutableIntStateOf(512) }
    var webMode by remember { mutableStateOf(true) }
    var loaded by remember { mutableStateOf(false) }
    var loadError by remember { mutableStateOf("") }
    var stats by remember { mutableStateOf(GenerationStats()) }

    fun refresh() { val d = File(activity.filesDir, "models"); stored = d.listFiles()?.filter { it.isFile && it.extension.equals("gguf", true) }?.sortedByDescending { it.lastModified() } ?: emptyList() }
    fun slotFor(f: File) = ModelSlot(name = f.name, storedPath = f.absolutePath)

    val pickT = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { u: Uri? -> u ?: return@rememberLauncherForActivityResult; target = ModelSlot(u.toString(), activity.displayName(u) ?: "model.gguf"); loaded = false; loadError = "" }
    val pickD = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { u: Uri? -> u ?: return@rememberLauncherForActivityResult; draft = ModelSlot(u.toString(), activity.displayName(u) ?: "dspark.gguf"); draftEnabled = true; loaded = false; loadError = "" }

    LaunchedEffect(Unit) { refresh() }
    DisposableEffect(Unit) { onDispose { engine.close() } }
    LaunchedEffect(messages.size, generating, searching) { if (messages.isNotEmpty()) list.animateScrollToItem(messages.lastIndex) }

    fun save() { if (messages.isEmpty()) return; val title = messages.firstOrNull { it.user }?.text.orEmpty().replace("\n", " ").take(42).ifBlank { "New chat" }; conversations = listOf(Conversation(chatId, title, messages)) + conversations.filterNot { it.id == chatId }; saveConversations(activity, conversations) }

    fun load() {
        val tSlot = target
        if (tSlot.uri.isEmpty() && tSlot.storedPath.isEmpty()) return
        scope.launch {
            loaded = false
            loadError = "Loading model…"
            val result = withContext(Dispatchers.IO) {
                try {
                    val dir = File(activity.filesDir, "models"); if (!dir.exists()) dir.mkdirs()
                    fun copy(slot: ModelSlot, fallback: String): File {
                        if (slot.storedPath.isNotEmpty()) { val f = File(slot.storedPath); if (!f.isFile) throw IllegalStateException("stored model not found"); return f }
                        val name = slot.name.ifBlank { fallback }.replace(Regex("[^A-Za-z0-9._-]"), "_")
                        val out = File(dir, name); val part = File(dir, "$name.part")
                        activity.contentResolver.openInputStream(Uri.parse(slot.uri))?.use { src -> part.outputStream().use { dst -> src.copyTo(dst, 1024 * 1024); dst.fd.sync() } } ?: throw IllegalStateException("could not open selected model")
                        if (part.length() == 0L) throw IllegalStateException("copied model is empty")
                        if (out.exists()) out.delete(); if (!part.renameTo(out)) throw IllegalStateException("could not finalize model file")
                        return out
                    }
                    val t = copy(tSlot, "model.gguf")
                    val d = if (draftEnabled && (draft.uri.isNotEmpty() || draft.storedPath.isNotEmpty())) copy(draft, "dspark.gguf") else null
                    val ok = if (d != null) engine.loadModelFromPath(t.absolutePath, d.absolutePath, contextSize) else engine.loadModelFromPath(t.absolutePath, contextSize)
                    ok to if (ok) "" else engine.lastError()
                } catch (e: Exception) { false to (e.message ?: "model load failed") }
            }
            loaded = result.first; loadError = result.second; if (result.first) refresh()
        }
    }

    fun send() {
        val q = prompt.trim(); if (q.isEmpty() || generating || searching || !loaded) return
        prompt = ""; messages = messages + Message(true, q) + Message(false, ""); generating = true; stats = GenerationStats(contextSize = contextSize)
        scope.launch {
            var src = emptyList<SearchResult>(); var ctx = ""
            try {
                if (webMode && SearchService.shouldSearch(q)) { searching = true; src = withContext(Dispatchers.IO) { search.search(q, 5) }; ctx = search.toLlmContext(src); searching = false }
                val conv = buildString { append("You are a helpful text-only local assistant. Answer naturally and accurately.\n"); if (ctx.isNotBlank()) append("\nUntrusted web context:\n").append(ctx); append("\n\nConversation:\n"); messages.dropLast(1).forEach { append(if (it.user) "User: " else "Assistant: ").append(it.text).append("\n") }; append("User: ").append(q).append("\nAssistant:") }
                val ch = Channel<StreamEvent>(Channel.UNLIMITED)
                val gen = async(Dispatchers.Default) {
                    try {
                        engine.generateStream(conv, maxTokens, onToken = { ch.trySend(StreamEvent.Token(it)) }, onStats = { a, b, c, d -> ch.trySend(StreamEvent.Stats(GenerationStats(a, b, c, d))) })
                    } finally { ch.close() }
                }
                val parser = ThinkStreamParser()
                for (e in ch) when (e) {
                    is StreamEvent.Token -> { val x = parser.consume(e.text); if (x.thinking.isNotEmpty() || x.answer.isNotEmpty()) { val m = messages.lastOrNull() ?: Message(false, ""); messages = messages.dropLast(1) + m.copy(text = m.text + x.answer, thinking = m.thinking + x.thinking, sources = src) } }
                    is StreamEvent.Stats -> stats = e.value
                }
                val status = gen.await(); val x = parser.finish()
                if (x.thinking.isNotEmpty() || x.answer.isNotEmpty()) { val m = messages.lastOrNull() ?: Message(false, ""); messages = messages.dropLast(1) + m.copy(text = m.text + x.answer, thinking = m.thinking + x.thinking, sources = src) }
                if (status.startsWith("[")) { val m = messages.lastOrNull() ?: Message(false, ""); messages = messages.dropLast(1) + m.copy(text = if (m.text.isEmpty()) status else m.text, sources = src) }
                save()
            } finally { searching = false; generating = false }
        }
    }

    MaterialTheme(colorScheme = darkColorScheme()) {
        ModalNavigationDrawer(drawerState = drawer, drawerContent = {
            ModalDrawerSheet {
                Row(Modifier.fillMaxWidth().padding(16.dp), verticalAlignment = Alignment.CenterVertically) { Text("トーク", style = MaterialTheme.typography.titleLarge); Spacer(Modifier.weight(1f)); TextButton({ save(); chatId = UUID.randomUUID().toString(); messages = emptyList(); scope.launch { drawer.close() } }) { Text("新しいチャット") } }
                HorizontalDivider()
                conversations.forEach { c -> NavigationDrawerItem(label = { Text(c.title, maxLines = 2) }, selected = c.id == chatId, onClick = { save(); chatId = c.id; messages = c.messages; scope.launch { drawer.close() } }, modifier = Modifier.padding(horizontal = 10.dp, vertical = 2.dp)) }
            }
        }) {
            Scaffold(topBar = { TopAppBar(navigationIcon = { TextButton({ scope.launch { drawer.open() } }) { Text("トーク") } }, title = { Text("Lfm Mobile") }, actions = { TextButton({ showModels = true }) { Text("Models") } }) }) { pad ->
                Column(Modifier.fillMaxSize().padding(pad)) {
                    Row(Modifier.fillMaxWidth().padding(12.dp), verticalAlignment = Alignment.CenterVertically) { FilterChip(webMode, { webMode = !webMode }, { Text(if (webMode) "Web search: Auto" else "Web search: Off") }); Spacer(Modifier.weight(1f)); Text(if (loaded) "Ready" else "Model not loaded") }
                    LazyColumn(state = list, modifier = Modifier.weight(1f).fillMaxWidth().padding(horizontal = 14.dp)) { if (messages.isEmpty()) item { Welcome(target, draft, draftEnabled, loaded) }; items(messages) { MessageBubble(it) }; if (searching) item { Text("Searching the web…") }; if (generating) item { GenerationStatus(stats) } }
                    Row(Modifier.fillMaxWidth().padding(12.dp), verticalAlignment = Alignment.Bottom) { OutlinedTextField(prompt, { prompt = it }, Modifier.weight(1f), placeholder = { Text("Message") }, enabled = !generating && !searching && loaded, shape = RoundedCornerShape(24.dp), maxLines = 6); Spacer(Modifier.width(8.dp)); Button({ send() }, enabled = prompt.isNotBlank() && !generating && !searching && loaded) { Text("Send") } }
                }
            }
        }
    }

    if (showModels) {
        AlertDialog(onDismissRequest = { showModels = false }, title = { Text("Models") }, text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                ModelCard("Target model", target) { pickT.launch(arrayOf("application/octet-stream", "application/x-gguf", "*/*")) }
                ModelCard("DSpark draft (optional)", draft) { pickD.launch(arrayOf("application/octet-stream", "application/x-gguf", "*/*")) }
                if (stored.isNotEmpty()) {
                    Text("Stored models", fontWeight = FontWeight.SemiBold)
                    stored.forEach { f -> Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) { Text(f.name, Modifier.weight(1f), maxLines = 1); TextButton({ target = slotFor(f); loaded = false; loadError = "Stored target selected" }) { Text("Target") }; TextButton({ draft = slotFor(f); draftEnabled = true; loaded = false; loadError = "Stored draft selected" }) { Text("Draft") } } }
                }
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) { Column(Modifier.weight(1f)) { Text("Draft acceleration", fontWeight = FontWeight.SemiBold); Text(if (draft.uri.isEmpty() && draft.storedPath.isEmpty()) "Select a draft model first" else if (draftEnabled) "Enabled" else "Disabled") }; Switch(draftEnabled, { draftEnabled = it; loaded = false }, enabled = draft.uri.isNotEmpty() || draft.storedPath.isNotEmpty()) }
                HorizontalDivider()
                OutlinedTextField(contextSize.toString(), { it.toIntOrNull()?.coerceIn(512, 131072)?.let { v -> contextSize = v; loaded = false } }, label = { Text("Context size") }, singleLine = true)
                OutlinedTextField(maxTokens.toString(), { it.toIntOrNull()?.coerceIn(1, 8192)?.let { v -> maxTokens = v } }, label = { Text("Max tokens") }, singleLine = true)
                if (loadError.isNotBlank()) Text(loadError, color = MaterialTheme.colorScheme.error)
                Button({ load() }, enabled = target.uri.isNotEmpty() || target.storedPath.isNotEmpty()) { Text(if (draftEnabled && (draft.uri.isNotEmpty() || draft.storedPath.isNotEmpty())) "Load Target + DSpark" else "Load model") }
            }
        }, confirmButton = { TextButton({ showModels = false }) { Text("Done") } })
    }
}

@Composable private fun GenerationStatus(s: GenerationStats) { Surface(Modifier.fillMaxWidth().padding(4.dp), shape = RoundedCornerShape(14.dp), color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = .55f)) { Row(Modifier.padding(12.dp)) { Text("Generating…", fontWeight = FontWeight.SemiBold); Spacer(Modifier.width(12.dp)); Text(String.format("%.1f tok/s", s.tokPerSec)); Spacer(Modifier.width(10.dp)); Text(String.format("%.1fs", s.elapsedMs / 1000.0)); Spacer(Modifier.width(10.dp)); Text("ctx ${s.contextUsed}/${s.contextSize}") } } }

@Composable private fun ModelCard(title: String, slot: ModelSlot, choose: () -> Unit) { Card { Column(Modifier.fillMaxWidth().padding(14.dp), verticalArrangement = Arrangement.spacedBy(7.dp)) { Text(title, fontWeight = FontWeight.SemiBold); Text(if (slot.name.isEmpty()) "No model selected" else slot.name, maxLines = 2); OutlinedButton(choose) { Text("Choose") } } } }

@Composable private fun Welcome(target: ModelSlot, draft: ModelSlot, enabled: Boolean, loaded: Boolean) { Column(Modifier.fillMaxWidth().padding(top = 70.dp), horizontalAlignment = Alignment.CenterHorizontally) { Text("Local AI", style = MaterialTheme.typography.headlineSmall); Text(if (loaded) target.name else "Choose a Target model in Models"); if (loaded && enabled && (draft.uri.isNotEmpty() || draft.storedPath.isNotEmpty())) Text("DSpark enabled") } }

@Composable private fun MessageBubble(message: Message) { val clipboard = LocalContext.current.getSystemService(ClipboardManager::class.java); fun copy(s: String) { clipboard?.setPrimaryClip(ClipData.newPlainText("Lfm Mobile", s)) }; Column(Modifier.fillMaxWidth()) { if (!message.user && message.thinking.isNotEmpty()) Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(14.dp), color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = .55f)) { Column(Modifier.padding(14.dp)) { Row(Modifier.fillMaxWidth()) { Text("Thinking", fontWeight = FontWeight.SemiBold); Spacer(Modifier.weight(1f)); TextButton({ copy(message.thinking) }) { Text("コピー") } }; Text(message.thinking) } }; if (message.user || message.text.isNotEmpty()) Row(Modifier.fillMaxWidth(), horizontalArrangement = if (message.user) Arrangement.End else Arrangement.Start) { Surface(shape = RoundedCornerShape(18.dp), color = if (message.user) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant) { Text(message.text, Modifier.padding(16.dp)) } }; if (!message.user && message.text.isNotEmpty()) TextButton({ copy(message.text) }) { Text("回答をコピー") } } }
