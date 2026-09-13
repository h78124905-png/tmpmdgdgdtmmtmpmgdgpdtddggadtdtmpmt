package com.example.lfmmobile

class LlamaEngine : AutoCloseable {
    companion object { init { System.loadLibrary("lfm_native") } }

    private external fun nativeLoadModelFromPath(modelPath: String, draftModelPath: String, contextSize: Int): Boolean
    private external fun nativeGetLastError(): String
    private external fun nativeGenerate(prompt: String, maxTokens: Int): String
    private external fun nativeGenerateStream(prompt: String, maxTokens: Int, callback: Any)
    private external fun nativeGenerateToolStep(messagesJson: String, toolsJson: String, maxTokens: Int): String
    private external fun nativeUnloadModel()

    fun loadModelFromPath(modelPath: String, contextSize: Int = 4096): Boolean = nativeLoadModelFromPath(modelPath, "", contextSize)
    fun loadModelFromPath(modelPath: String, draftModelPath: String, contextSize: Int = 4096): Boolean = nativeLoadModelFromPath(modelPath, draftModelPath, contextSize)
    fun lastError(): String = nativeGetLastError()
    fun generate(prompt: String, maxTokens: Int = 128): String = nativeGenerate(prompt, maxTokens)

    fun generateStream(prompt: String, maxTokens: Int = 128, onToken: (String) -> Unit, onStats: (Double, Long, Int, Int) -> Unit = { _, _, _, _ -> }): String {
        val callback = object {
            @Suppress("unused") fun onToken(text: String) { onToken(text) }
            @Suppress("unused") fun onStats(tokPerSec: Double, elapsedMs: Long, contextUsed: Int, contextSize: Int) { onStats(tokPerSec, elapsedMs, contextUsed, contextSize) }
        }
        nativeGenerateStream(prompt, maxTokens, callback); return ""
    }

    fun generateToolStep(messagesJson: String, toolsJson: String, maxTokens: Int = 384): String =
        nativeGenerateToolStep(messagesJson, toolsJson, maxTokens)

    override fun close() { nativeUnloadModel() }
}
