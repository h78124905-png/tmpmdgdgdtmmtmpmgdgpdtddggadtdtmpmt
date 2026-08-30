package com.example.lfmmobile

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
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private data class Message(val user: Boolean, val text: String)
private data class ModelSlot(val uri: String = "", val name: String = "", val loaded: Boolean = false)
private const val TARGET = 0
private const val DSPARK = 1

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { ChatApp() }
    }

    fun displayName(uri: Uri): String? {
        contentResolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) return c.getString(0)
        }
        return null
    }
}

@Composable
private fun ChatApp() {
    val activity = LocalContext.current as ComponentActivity
    val engine = remember { LlamaEngine() }
    val scope = rememberCoroutineScope()
    val listState = rememberLazyListState()
    var target by remember { mutableStateOf(ModelSlot()) }
    var dspark by remember { mutableStateOf(ModelSlot()) }
    var activeRole by remember { mutableIntStateOf(TARGET) }
    var messages by remember { mutableStateOf(listOf<Message>()) }
    var prompt by remember { mutableStateOf("") }
    var generating by remember { mutableStateOf(false) }
    var showModels by remember { mutableStateOf(false) }
    var contextSize by remember { mutableIntStateOf(4096) }
    var maxTokens by remember { mutableIntStateOf(512) }
    var pickerRole by remember { mutableIntStateOf(TARGET) }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        uri ?: return@rememberLauncherForActivityResult
        val name = activity.displayName(uri) ?: uri.lastPathSegment?.substringAfterLast('/') ?: "model.gguf"
        val slot = ModelSlot(uri.toString(), name, false)
        if (pickerRole == TARGET) target = slot else dspark = slot
    }

    LaunchedEffect(messages.size, generating) {
        if (messages.isNotEmpty()) listState.animateScrollToItem(messages.lastIndex)
    }

    fun load(role: Int) {
        val slot = if (role == TARGET) target else dspark
        if (slot.uri.isEmpty()) return
        scope.launch {
            val ok = withContext(Dispatchers.IO) {
                activity.contentResolver.openFileDescriptor(Uri.parse(slot.uri), "r")?.use { pfd ->
                    engine.loadModelFromFd(role, pfd.detachFd(), contextSize)
                } ?: false
            }
            if (role == TARGET) target = target.copy(loaded = ok) else dspark = dspark.copy(loaded = ok)
        }
    }

    fun send() {
        val text = prompt.trim()
        val ready = if (activeRole == TARGET) target.loaded else dspark.loaded
        if (text.isEmpty() || generating || !ready) return
        messages = messages + Message(true, text)
        prompt = ""
        generating = true
        scope.launch {
            val answer = withContext(Dispatchers.Default) { engine.generate(activeRole, text, maxTokens) }
            messages = messages + Message(false, answer)
            generating = false
        }
    }

    MaterialTheme(colorScheme = darkColorScheme()) {
        Surface(Modifier.fillMaxSize()) {
            Scaffold(topBar = {
                TopAppBar(
                    title = { Text("Lfm Mobile", fontWeight = FontWeight.SemiBold) },
                    actions = {
                        TextButton({ messages = emptyList() }) { Text("New") }
                        TextButton({ showModels = true }) { Text("Models") }
                    }
                )
            }) { padding ->
                Column(Modifier.fillMaxSize().padding(padding)) {
                    Row(Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 6.dp), horizontalArrangement = Arrangement.Center) {
                        FilterChip(selected = activeRole == TARGET, onClick = { activeRole = TARGET }, label = { Text("Target") })
                        Spacer(Modifier.width(8.dp))
                        FilterChip(selected = activeRole == DSPARK, onClick = { activeRole = DSPARK }, label = { Text("dSpark") })
                    }
                    LazyColumn(
                        state = listState,
                        modifier = Modifier.weight(1f).fillMaxWidth().padding(horizontal = 14.dp),
                        verticalArrangement = Arrangement.spacedBy(14.dp),
                        contentPadding = PaddingValues(vertical = 18.dp)
                    ) {
                        if (messages.isEmpty()) item { Welcome(activeRole, target, dspark) }
                        items(messages) { MessageBubble(it) }
                        if (generating) item { Text("Thinking…", Modifier.padding(start = 12.dp)) }
                    }
                    Row(Modifier.fillMaxWidth().padding(12.dp), verticalAlignment = Alignment.Bottom) {
                        OutlinedTextField(
                            value = prompt, onValueChange = { prompt = it }, modifier = Modifier.weight(1f),
                            placeholder = { Text(if (activeRole == TARGET) "Message Target model" else "Message dSpark model") },
                            enabled = !generating && if (activeRole == TARGET) target.loaded else dspark.loaded,
                            shape = RoundedCornerShape(24.dp), maxLines = 6
                        )
                        Spacer(Modifier.width(8.dp))
                        Button(onClick = { send() }, enabled = prompt.isNotBlank() && !generating && if (activeRole == TARGET) target.loaded else dspark.loaded) { Text("Send") }
                    }
                }
            }
        }
    }

    if (showModels) {
        AlertDialog(
            onDismissRequest = { showModels = false },
            title = { Text("Models") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    ModelCard("Target model", target,
                        onPick = { pickerRole = TARGET; picker.launch(arrayOf("application/octet-stream", "application/x-gguf", "*/*")) },
                        onLoad = { load(TARGET) })
                    ModelCard("dSpark model", dspark,
                        onPick = { pickerRole = DSPARK; picker.launch(arrayOf("application/octet-stream", "application/x-gguf", "*/*")) },
                        onLoad = { load(DSPARK) })
                    HorizontalDivider()
                    OutlinedTextField(contextSize.toString(), { it.toIntOrNull()?.let { v -> contextSize = v } }, label = { Text("Context size") }, singleLine = true)
                    OutlinedTextField(maxTokens.toString(), { it.toIntOrNull()?.let { v -> maxTokens = v } }, label = { Text("Max tokens") }, singleLine = true)
                    Text("Dark mode only", style = MaterialTheme.typography.bodySmall)
                }
            },
            confirmButton = { TextButton({ showModels = false }) { Text("Done") } }
        )
    }
}

@Composable
private fun ModelCard(title: String, slot: ModelSlot, onPick: () -> Unit, onLoad: () -> Unit) {
    Card {
        Column(Modifier.fillMaxWidth().padding(14.dp), verticalArrangement = Arrangement.spacedBy(7.dp)) {
            Text(title, fontWeight = FontWeight.SemiBold)
            Text(if (slot.name.isEmpty()) "No model selected" else slot.name, maxLines = 2)
            Text(if (slot.loaded) "Loaded" else if (slot.uri.isEmpty()) "Select a downloaded GGUF" else "Selected — load when ready", style = MaterialTheme.typography.bodySmall)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = onPick) { Text("Choose") }
                Button(onClick = onLoad, enabled = slot.uri.isNotEmpty() && !slot.loaded) { Text("Load") }
            }
        }
    }
}

@Composable
private fun Welcome(activeRole: Int, target: ModelSlot, dspark: ModelSlot) {
    Column(Modifier.fillMaxWidth().padding(top = 70.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        Text(if (activeRole == TARGET) "Target model" else "dSpark model", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Medium)
        Text(if (activeRole == TARGET) target.name.ifEmpty { "Choose a model in Models" } else dspark.name.ifEmpty { "Choose a model in Models" }, style = MaterialTheme.typography.bodyMedium)
    }
}

@Composable
private fun MessageBubble(message: Message) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = if (message.user) Arrangement.End else Arrangement.Start) {
        Surface(shape = RoundedCornerShape(18.dp), color = if (message.user) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant) {
            Text(message.text, Modifier.padding(horizontal = 16.dp, vertical = 11.dp), style = MaterialTheme.typography.bodyLarge)
        }
    }
}
