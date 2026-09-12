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
private data class ModelSlot(val uri: String = "", val name: String = "")
private data class Conversation(val id: String, val title: String, val messages: List<Message>)
private data class GenerationStats(val tokPerSec: Double = 0.0, val elapsedMs: Long = 0L, val contextUsed: Int = 0, val contextSize: Int = 0)
private sealed interface StreamEvent { data class Token(val text: String) : StreamEvent; data class Stats(val value: GenerationStats) : StreamEvent }

private class ThinkStreamParser {
    private enum class Mode { UNKNOWN, THINKING, ANSWERING }
    private var mode = Mode.UNKNOWN
    private var pending = ""
    data class Emission(val thinking: String = "", val answer: String = "")
    private val openingMarkers = listOf("<think>", "<|think|>", "<|thinking|>", "<|assistant_thinking|>")
    private val closingMarkers = listOf("</think>", "</thinking>", "<|/think|>", "<|/thinking|>", "<|end_think|>", "<|end_thinking|>")
    private fun earliestMarker(text: String, markers: List<String>): Pair<Int, String>? {
        var best: Pair<Int, String>? = null
        for (marker in markers) { val index = text.indexOf(marker); if (index >= 0 && (best == null || index < best!!.first)) best = index to marker }
        return best
    }
    fun consume(chunk: String): Emission {
        pending += chunk; var thinkingOut = ""; var answerOut = ""
        while (pending.isNotEmpty()) {
            val open = earliestMarker(pending, openingMarkers); val close = earliestMarker(pending, closingMarkers)
            when (mode) {
                Mode.UNKNOWN -> when {
                    open != null && (close == null || open.first <= close.first) -> { if (open.first > 0) answerOut += pending.substring(0, open.first); pending = pending.substring(open.first + open.second.length); mode = Mode.THINKING }
                    close != null -> { if (close.first > 0) thinkingOut += pending.substring(0, close.first); pending = pending.substring(close.first + close.second.length); mode = Mode.ANSWERING }
                    else -> break
                }
                Mode.THINKING -> if (close != null) { if (close.first > 0) thinkingOut += pending.substring(0, close.first); pending = pending.substring(close.first + close.second.length); mode = Mode.ANSWERING } else { val partial = partialMarkerLength(pending, closingMarkers); val n = pending.length - partial; if (n > 0) { thinkingOut += pending.substring(0, n); pending = pending.substring(n) }; break }
                Mode.ANSWERING -> if (open != null) { if (open.first > 0) answerOut += pending.substring(0, open.first); pending = pending.substring(open.first + open.second.length); mode = Mode.THINKING } else { val partial = partialMarkerLength(pending, openingMarkers); val n = pending.length - partial; if (n > 0) { answerOut += pending.substring(0, n); pending = pending.substring(n) }; break }
            }
        }
        return Emission(thinkingOut, answerOut)
    }
    private fun partialMarkerLength(text: String, markers: List<String>): Int { var keep = 0; for (marker in markers) for (n in 1 until marker.length) if (text.endsWith(marker.substring(0, n))) keep = maxOf(keep, n); return keep }
    fun finish(): Emission { val rest = pending; pending = ""; return when (mode) { Mode.THINKING -> Emission(thinking = rest); else -> Emission(answer = rest) } }
}

private fun loadConversations(context: Context): List<Conversation> = try {
    val array = JSONArray(context.getSharedPreferences("chat_history", Context.MODE_PRIVATE).getString("conversations", "[]") ?: "[]")
    buildList { for (i in 0 until array.length()) { val o = array.getJSONObject(i); val a = o.optJSONArray("messages") ?: JSONArray(); add(Conversation(o.optString("id"), o.optString("title"), buildList { for (j in 0 until a.length()) { val m = a.getJSONObject(j); add(Message(m.optBoolean("user"), m.optString("text"), m.optString("thinking"))) } })) } }
} catch (_: Exception) { emptyList() }
private fun saveConversations(context: Context, conversations: List<Conversation>) {
    val array = JSONArray(); conversations.forEach { c -> val ms = JSONArray(); c.messages.forEach { m -> ms.put(JSONObject().put("user", m.user).put("text", m.text).put("thinking", m.thinking)) }; array.put(JSONObject().put("id", c.id).put("title", c.title).put("messages", ms)) }; context.getSharedPreferences("chat_history", Context.MODE_PRIVATE).edit().putString("conversations", array.toString()).apply()
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) { super.onCreate(savedInstanceState); setContent { ChatApp() } }
    fun displayName(uri: Uri): String? { contentResolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)?.use { if (it.moveToFirst()) return it.getString(0) }; return null }
}

@Composable private fun ChatApp() {
    val activity = LocalContext.current as MainActivity; val engine = remember { LlamaEngine() }; val search = remember { SearchService() }; val scope = rememberCoroutineScope(); val listState = rememberLazyListState(); val drawerState = rememberDrawerState(DrawerValue.Closed)
    var target by remember { mutableStateOf(ModelSlot()) }; var draft by remember { mutableStateOf(ModelSlot()) }; var draftEnabled by remember { mutableStateOf(false) }; var storedModels by remember { mutableStateOf(emptyList<File>()) }
    var messages by remember { mutableStateOf(listOf<Message>()) }; var conversations by remember { mutableStateOf(loadConversations(activity)) }; var currentChatId by remember { mutableStateOf(UUID.randomUUID().toString()) }; var prompt by remember { mutableStateOf("") }; var generating by remember { mutableStateOf(false) }; var searching by remember { mutableStateOf(false) }; var showModels by remember { mutableStateOf(false) }; var contextSize by remember { mutableIntStateOf(2048) }; var maxTokens by remember { mutableIntStateOf(512) }; var webMode by remember { mutableStateOf(true) }; var loaded by remember { mutableStateOf(false) }; var loadError by remember { mutableStateOf("") }; var generationStats by remember { mutableStateOf(GenerationStats()) }
    fun refreshStoredModels() { val dir = File(activity.filesDir, "models"); storedModels = dir.listFiles()?.filter { it.isFile && it.extension.equals("gguf", true) }?.sortedByDescending { it.lastModified() } ?: emptyList() }
    val pickerTarget = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? -> uri ?: return@rememberLauncherForActivityResult; target = ModelSlot(uri.toString(), activity.displayName(uri) ?: "model.gguf"); loaded = false; loadError = "" }
    val pickerDraft = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? -> uri ?: return@rememberLauncherForActivityResult; draft = ModelSlot(uri.toString(), activity.displayName(uri) ?: "dspark.gguf"); draftEnabled = true; loaded = false; loadError = "" }
    LaunchedEffect(Unit) { refreshStoredModels() }
    DisposableEffect(Unit) { onDispose { engine.close() } }
    LaunchedEffect(messages.size, generating, searching) { if (messages.isNotEmpty()) listState.animateScrollToItem(messages.lastIndex) }
    fun saveCurrentChat() { if (messages.isEmpty()) return; val title = messages.firstOrNull { it.user }?.text.orEmpty().replace("\n", " ").take(42).ifBlank { "New chat" }; conversations = listOf(Conversation(currentChatId, title, messages)) + conversations.filterNot { it.id == currentChatId }; saveConversations(activity, conversations) }
    fun newChat() { saveCurrentChat(); currentChatId = UUID.randomUUID().toString(); messages = emptyList(); generationStats = GenerationStats(); scope.launch { drawerState.close() } }
    fun openConversation(c: Conversation) { saveCurrentChat(); currentChatId = c.id; messages = c.messages; generationStats = GenerationStats(); scope.launch { drawerState.close() } }
    fun loadModel() {
        if (target.uri.isEmpty()) return
        scope.launch {
            loaded = false; loadError = "Copying model to app storage…"
            val result = withContext(Dispatchers.IO) {
                try {
                    val dir = File(activity.filesDir, "models"); if (!dir.exists() && !dir.mkdirs()) return@withContext false to "could not create model directory"
                    fun copyModel(slot: ModelSlot, fallback: String): File { val name = slot.name.ifBlank { fallback }.replace(Regex("[^A-Za-z0-9._-]"), "_"); val out = File(dir, name); val part = File(dir, "$name.part"); val input = activity.contentResolver.openInputStream(Uri.parse(slot.uri)) ?: throw IllegalStateException("could not open selected model"); input.use { src -> part.outputStream().use { dst -> src.copyTo(dst, 1024 * 1024); dst.fd.sync() } }; if (!part.isFile || part.length() == 0L) { part.delete(); throw IllegalStateException("copied model is empty") }; if (out.exists()) out.delete(); if (!part.renameTo(out)) { part.delete(); throw IllegalStateException("could not finalize model file") }; out }
                    val t = copyModel(target, "model.gguf"); val d = if (draftEnabled && draft.uri.isNotEmpty()) copyModel(draft, "dspark.gguf") else null; loadError = if (d != null) "Loading Target + DSpark…" else "Loading GGUF…"; val ok = if (d != null) engine.loadModelFromPath(t.absolutePath, d.absolutePath, contextSize) else engine.loadModelFromPath(t.absolutePath, contextSize); ok to if (ok) "" else engine.lastError()
                } catch (e: Exception) { false to (e.message ?: "could not copy/load model") }
            }
            loaded = result.first; loadError = result.second; if (result.first) refreshStoredModels()
        }
    }
    fun send() {
        val text = prompt.trim(); if (text.isEmpty() || generating || searching || !loaded) return; prompt = ""; messages = messages + Message(true, text) + Message(false, ""); generationStats = GenerationStats(contextSize = contextSize); generating = true
        scope.launch { var sourceResults = emptyList<SearchResult>(); var searchContext = ""; try {
            if (webMode && SearchService.shouldSearch(text)) { searching = true; sourceResults = withContext(Dispatchers.IO) { search.search(text, 5) }; searchContext = search.toLlmContext(sourceResults); searching = false }
            val conversation = buildString { append("You are a helpful text-only local assistant. Answer naturally and accurately.\n"); if (searchContext.isNotBlank()) append("\nUntrusted web context:\n").append(searchContext); append("\n\nConversation:\n"); messages.dropLast(1).forEach { append(if (it.user) "User: " else "Assistant: ").append(it.text).append("\n") }; append("User: ").append(text).append("\nAssistant:") }
            val channel = Channel<StreamEvent>(Channel.UNLIMITED); val generation = async(Dispatchers.Default) { try { engine.generateStream(conversation, maxTokens, onToken = { channel.trySend(StreamEvent.Token(it)) }, onStats = { a,b,c,d -> channel.trySend(StreamEvent.Stats(GenerationStats(a,b,c,d))) }) } finally { channel.close() } }; val parser = ThinkStreamParser()
            for (event in channel) when (event) { is StreamEvent.Token -> { val e = parser.consume(event.text); if (e.thinking.isNotEmpty() || e.answer.isNotEmpty()) { val cur = messages.lastOrNull() ?: Message(false, ""); messages = messages.dropLast(1) + cur.copy(text = cur.text + e.answer, thinking = cur.thinking + e.thinking, sources = sourceResults) } }; is StreamEvent.Stats -> generationStats = event.value }
            val status = generation.await(); val e = parser.finish(); if (e.thinking.isNotEmpty() || e.answer.isNotEmpty()) { val cur = messages.lastOrNull() ?: Message(false, ""); messages = messages.dropLast(1) + cur.copy(text = cur.text + e.answer, thinking = cur.thinking + e.thinking, sources = sourceResults) }; if (status.startsWith("[")) { val cur = messages.lastOrNull() ?: Message(false, ""); messages = messages.dropLast(1) + cur.copy(text = if (cur.text.isEmpty()) status else cur.text, sources = sourceResults) }; saveCurrentChat()
        } finally { searching = false; generating = false } }
    }
    MaterialTheme(colorScheme = darkColorScheme()) { ModalNavigationDrawer(drawerState = drawerState, drawerContent = { ModalDrawerSheet { Row(Modifier.fillMaxWidth().padding(16.dp), verticalAlignment = Alignment.CenterVertically) { Text("トーク", style = MaterialTheme.typography.titleLarge); Spacer(Modifier.weight(1f)); TextButton({ newChat() }) { Text("新しいチャット") } }; HorizontalDivider(); conversations.forEach { c -> NavigationDrawerItem(label = { Text(c.title, maxLines = 2) }, selected = c.id == currentChatId, onClick = { openConversation(c) }, modifier = Modifier.padding(horizontal = 10.dp, vertical = 2.dp)) } } }) {
        Surface(Modifier.fillMaxSize()) { Scaffold(topBar = { TopAppBar(navigationIcon = { TextButton({ scope.launch { drawerState.open() } }) { Text("トーク") } }, title = { Text("Lfm Mobile") }, actions = { TextButton({ showModels = true }) { Text("Models") } }) }) { padding -> Column(Modifier.fillMaxSize().padding(padding)) { Row(Modifier.fillMaxWidth().padding(12.dp), verticalAlignment = Alignment.CenterVertically) { FilterChip(selected = webMode, onClick = { webMode = !webMode }, label = { Text(if (webMode) "Web search: Auto" else "Web search: Off") }); Spacer(Modifier.weight(1f)); Text(if (loaded) "Ready" else "Model not loaded") }; LazyColumn(state = listState, modifier = Modifier.weight(1f).fillMaxWidth().padding(horizontal = 14.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) { if (messages.isEmpty()) item { Welcome(target, draft, draftEnabled, loaded) }; items(messages) { MessageBubble(it) }; if (searching) item { Text("Searching the web…") }; if (generating) item { GenerationStatus(generationStats) } }; Row(Modifier.fillMaxWidth().padding(12.dp), verticalAlignment = Alignment.Bottom) { OutlinedTextField(prompt, { prompt = it }, Modifier.weight(1f), placeholder = { Text("Message") }, enabled = !generating && !searching && loaded, shape = RoundedCornerShape(24.dp), maxLines = 6); Spacer(Modifier.width(8.dp)); Button({ send() }, enabled = prompt.isNotBlank() && !generating && !searching && loaded) { Text("Send") } } } } }
    }
    if (showModels) AlertDialog(onDismissRequest = { showModels = false }, title = { Text("Models") }, text = { Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        ModelCard("Target model", target) { pickerTarget.launch(arrayOf("application/octet-stream", "application/x-gguf", "*/*")) }; ModelCard("DSpark draft (optional)", draft) { pickerDraft.launch(arrayOf("application/octet-stream", "application/x-gguf", "*/*")) }
        if (storedModels.isNotEmpty()) { Text("Stored models", fontWeight = FontWeight.SemiBold); storedModels.forEach { f -> TextButton(onClick = { target = ModelSlot(Uri.parse("content://local/${f.name}" ).toString(), f.name); loaded = false; loadError = "Stored model selected — press Load model" }) { Text(f.name, maxLines = 1) } } }
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) { Column(Modifier.weight(1f)) { Text("Draft acceleration", fontWeight = FontWeight.SemiBold); Text(if (draft.uri.isEmpty()) "Select a DSpark draft model first" else if (draftEnabled) "Enabled" else "Disabled") }; Switch(draftEnabled, { draftEnabled = it; loaded = false }, enabled = draft.uri.isNotEmpty()) }
        HorizontalDivider(); OutlinedTextField(contextSize.toString(), { it.toIntOrNull()?.coerceIn(512, 131072)?.let { v -> contextSize = v; loaded = false } }, label = { Text("Context size") }, singleLine = true); OutlinedTextField(maxTokens.toString(), { it.toIntOrNull()?.coerceIn(1, 8192)?.let { v -> maxTokens = v } }, label = { Text("Max tokens") }, singleLine = true); if (loadError.isNotBlank()) Text(loadError, color = MaterialTheme.colorScheme.error); Button({ loadModel() }, enabled = target.uri.isNotEmpty()) { Text(if (draftEnabled && draft.uri.isNotEmpty()) "Load Target + DSpark" else "Load model") }
    } }, confirmButton = { TextButton({ showModels = false }) { Text("Done") } })
}

@Composable private fun GenerationStatus(stats: GenerationStats) { Surface(Modifier.fillMaxWidth().padding(4.dp), shape = RoundedCornerShape(14.dp), color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = .55f)) { Row(Modifier.fillMaxWidth().padding(12.dp)) { Text("Generating…", fontWeight = FontWeight.SemiBold); Spacer(Modifier.width(12.dp)); Text(String.format("%.1f tok/s", stats.tokPerSec)); Spacer(Modifier.width(10.dp)); Text(String.format("%.1fs", stats.elapsedMs / 1000.0)); Spacer(Modifier.width(10.dp)); Text("ctx ${stats.contextUsed}/${stats.contextSize}") } } }
@Composable private fun ModelCard(title: String, slot: ModelSlot, onPick: () -> Unit) { Card { Column(Modifier.fillMaxWidth().padding(14.dp), verticalArrangement = Arrangement.spacedBy(7.dp)) { Text(title, fontWeight = FontWeight.SemiBold); Text(if (slot.name.isEmpty()) "No model selected" else slot.name, maxLines = 2); Text(if (slot.uri.isEmpty()) "Select a GGUF from device storage" else "Selected"); OutlinedButton(onPick) { Text("Choose") } } } }
@Composable private fun Welcome(target: ModelSlot, draft: ModelSlot, draftEnabled: Boolean, loaded: Boolean) { Column(Modifier.fillMaxWidth().padding(top = 70.dp), horizontalAlignment = Alignment.CenterHorizontally) { Text("Local AI", style = MaterialTheme.typography.headlineSmall); Text(if (loaded) target.name else "Choose a Target model in Models"); if (loaded && draftEnabled && draft.uri.isNotEmpty()) Text("DSpark enabled") } }
@Composable private fun MessageBubble(message: Message) { val clipboard = LocalContext.current.getSystemService(ClipboardManager::class.java); fun copy(t: String) { clipboard?.setPrimaryClip(ClipData.newPlainText("Lfm Mobile", t)) }; Column(Modifier.fillMaxWidth()) { if (!message.user && message.thinking.isNotEmpty()) Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(14.dp), color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = .55f)) { Column(Modifier.padding(14.dp)) { Row(Modifier.fillMaxWidth()) { Text("Thinking", fontWeight = FontWeight.SemiBold); Spacer(Modifier.weight(1f)); TextButton({ copy(message.thinking) }) { Text("コピー") } }; Text(message.thinking) } }; if (message.user || message.text.isNotEmpty()) Row(Modifier.fillMaxWidth(), horizontalArrangement = if (message.user) Arrangement.End else Arrangement.Start) { Surface(shape = RoundedCornerShape(18.dp), color = if (message.user) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant) { Text(message.text, Modifier.padding(16.dp)) } }; if (!message.user && message.text.isNotEmpty()) TextButton({ copy(message.text) }) { Text("回答をコピー") } } }
