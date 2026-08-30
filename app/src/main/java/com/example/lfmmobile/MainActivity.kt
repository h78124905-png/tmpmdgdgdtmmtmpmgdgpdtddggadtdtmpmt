package com.example.lfmmobile

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { App() }
    }
}

@androidx.compose.runtime.Composable
private fun App() {
    val engine = remember { LlamaEngine() }
    var modelPath by remember { mutableStateOf("") }
    var prompt by remember { mutableStateOf("Hello") }
    var output by remember { mutableStateOf("llama.cpp is integrated. Select a GGUF model path to run inference.") }

    MaterialTheme {
        Surface(modifier = Modifier.fillMaxSize()) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(24.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text("Lfm Mobile", style = MaterialTheme.typography.headlineMedium)
                Text("llama.cpp / JNI / ARM64")

                OutlinedTextField(
                    value = modelPath,
                    onValueChange = { modelPath = it },
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("GGUF model path") },
                    singleLine = true
                )

                Button(
                    onClick = {
                        output = if (engine.loadModel(modelPath)) {
                            "Model loaded."
                        } else {
                            "Failed to load model."
                        }
                    },
                    enabled = modelPath.isNotBlank()
                ) {
                    Text("Load model")
                }

                OutlinedTextField(
                    value = prompt,
                    onValueChange = { prompt = it },
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Prompt") }
                )

                Button(
                    onClick = {
                        output = engine.generate(prompt, 128)
                    }
                ) {
                    Text("Generate")
                }

                Text(output)
            }
        }
    }
}
