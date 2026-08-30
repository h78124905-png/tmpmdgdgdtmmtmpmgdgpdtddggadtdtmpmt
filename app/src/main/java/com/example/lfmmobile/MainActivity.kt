package com.example.lfmmobile

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private data class Message(val user: Boolean, val text: String)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { ChatApp() }
    }
}

@Composable
private fun ChatApp() {
    val engine = remember { LlamaEngine() }
    val scope = rememberCoroutineScope()
    val listState = rememberLazyListState()
    var messages by remember { mutableStateOf(listOf<Message>()) }
    var prompt by remember { mutableStateOf("") }
    var modelPath by remember { mutableStateOf("") }
    var loaded by remember { mutableStateOf(false) }
    var generating by remember { mutableStateOf(false) }
    var showSettings by remember { mutableStateOf(false) }
    var showHistory by remember { mutableStateOf(false) }
    var contextSize by remember { mutableStateOf(4096) }
    var temperature by remember { mutableStateOf("0.7") }
    var maxTokens by remember { mutableStateOf("512") }

    LaunchedEffect(messages.size, generating) {
        if (messages.isNotEmpty()) listState.animateScrollToItem(messages.lastIndex)
    }

    fun newChat() { messages = emptyList(); prompt = "" }

    fun send() {
        val text = prompt.trim()
        if (text.isEmpty() || generating || !loaded) return
        messages = messages + Message(true, text)
        prompt = ""
        generating = true
        scope.launch {
            val answer = withContext(Dispatchers.Default) {
                engine.generate(text, maxTokens.toIntOrNull() ?: 512)
            }
            messages = messages + Message(false, answer)
            generating = false
        }
    }

    MaterialTheme {
        Surface(Modifier.fillMaxSize()) {
            Scaffold(
                topBar = {
                    TopAppBar(
                        title = { Text("Lfm Mobile", fontWeight = FontWeight.SemiBold) },
                        navigationIcon = { TextButton(onClick = { showHistory = true }) { Text("Chats") } },
                        actions = {
                            TextButton(onClick = { newChat() }) { Text("New") }
                            TextButton(onClick = { showSettings = true }) { Text("Settings") }
                        }
                    )
                }
            ) { padding ->
                Column(Modifier.fillMaxSize().padding(padding)) {
                    if (!loaded) {
                        ModelSetup(modelPath, { modelPath = it }) {
                            loaded = engine.loadModel(modelPath, contextSize)
                        }
                    } else {
                        LazyColumn(
                            state = listState,
                            modifier = Modifier.weight(1f).fillMaxWidth().padding(horizontal = 14.dp),
                            verticalArrangement = Arrangement.spacedBy(14.dp),
                            contentPadding = androidx.compose.foundation.layout.PaddingValues(vertical = 18.dp)
                        ) {
                            if (messages.isEmpty()) item { Welcome() }
                            items(messages) { MessageBubble(it) }
                            if (generating) item { TypingIndicator() }
                        }
                    }
                    Row(
                        Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp),
                        verticalAlignment = Alignment.Bottom
                    ) {
                        OutlinedTextField(
                            value = prompt,
                            onValueChange = { prompt = it },
                            modifier = Modifier.weight(1f),
                            placeholder = { Text(if (loaded) "Message Lfm Mobile" else "Load a GGUF model first") },
                            enabled = loaded && !generating,
                            shape = RoundedCornerShape(24.dp),
                            maxLines = 6
                        )
                        Spacer(Modifier.size(8.dp))
                        Button(onClick = { send() }, enabled = loaded && prompt.isNotBlank() && !generating) {
                            Text("Send")
                        }
                    }
                }
            }
        }
    }

    if (showSettings) {
        SettingsDialog(
            modelPath, { modelPath = it }, contextSize, { contextSize = it },
            temperature, { temperature = it }, maxTokens, { maxTokens = it },
            onApply = { loaded = engine.loadModel(modelPath, contextSize); showSettings = false },
            onDismiss = { showSettings = false }
        )
    }
    if (showHistory) {
        AlertDialog(
            onDismissRequest = { showHistory = false },
            title = { Text("Chats") },
            text = { Text(if (messages.isEmpty()) "No conversation yet." else "Current conversation\n${messages.count { it.user }} user messages") },
            confirmButton = { TextButton(onClick = { showHistory = false }) { Text("Close") } },
            dismissButton = { TextButton(onClick = { newChat(); showHistory = false }) { Text("New chat") } }
        )
    }
}

@Composable
private fun Welcome() {
    Column(Modifier.fillMaxWidth().padding(top = 90.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        Box(Modifier.size(58.dp).clip(CircleShape).background(MaterialTheme.colorScheme.primaryContainer), contentAlignment = Alignment.Center) {
            Text("L", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
        }
        Spacer(Modifier.height(16.dp))
        Text("How can I help you?", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Medium)
        Text("Private, local llama.cpp assistant", style = MaterialTheme.typography.bodyMedium)
    }
}

@Composable
private fun MessageBubble(message: Message) {
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = if (message.user) Arrangement.End else Arrangement.Start,
        verticalAlignment = Alignment.Top
    ) {
        if (!message.user) {
            Box(Modifier.size(30.dp).clip(CircleShape).background(MaterialTheme.colorScheme.primaryContainer), contentAlignment = Alignment.Center) {
                Text("L", fontWeight = FontWeight.Bold)
            }
            Spacer(Modifier.size(8.dp))
        }
        Card(
            colors = CardDefaults.cardColors(
                containerColor = if (message.user) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant
            ),
            shape = RoundedCornerShape(18.dp)
        ) {
            Text(message.text, Modifier.padding(horizontal = 16.dp, vertical = 11.dp), style = MaterialTheme.typography.bodyLarge)
        }
    }
}

@Composable
private fun TypingIndicator() {
    Text("Thinking…", Modifier.padding(start = 38.dp), style = MaterialTheme.typography.bodyMedium)
}

@Composable
private fun ModelSetup(path: String, onPath: (String) -> Unit, onLoad: () -> Unit) {
    Column(Modifier.fillMaxSize().padding(24.dp), verticalArrangement = Arrangement.Center) {
        Text("Load your GGUF model", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold)
        Spacer(Modifier.height(8.dp))
        Text("Enter the local model path, then start chatting.")
        Spacer(Modifier.height(20.dp))
        OutlinedTextField(path, onPath, Modifier.fillMaxWidth(), label = { Text("GGUF model path") }, singleLine = true)
        Spacer(Modifier.height(12.dp))
        Button(onClick = onLoad, enabled = path.isNotBlank(), Modifier.fillMaxWidth()) { Text("Load model") }
    }
}

@Composable
private fun SettingsDialog(
    path: String, onPath: (String) -> Unit,
    context: Int, onContext: (Int) -> Unit,
    temp: String, onTemp: (String) -> Unit,
    max: String, onMax: (String) -> Unit,
    onApply: () -> Unit, onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Model settings") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedTextField(path, onPath, label = { Text("GGUF path") }, singleLine = true)
                OutlinedTextField(context.toString(), { it.toIntOrNull()?.let(onContext) }, label = { Text("Context size") }, singleLine = true)
                OutlinedTextField(temp, onTemp, label = { Text("Temperature") }, singleLine = true)
                OutlinedTextField(max, onMax, label = { Text("Max tokens") }, singleLine = true)
            }
        },
        confirmButton = { TextButton(onClick = onApply, enabled = path.isNotBlank()) { Text("Apply & load") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } }
    )
}
