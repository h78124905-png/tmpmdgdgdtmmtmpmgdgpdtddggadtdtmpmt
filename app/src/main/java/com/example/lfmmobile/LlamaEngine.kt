package com.example.lfmmobile

class LlamaEngine : AutoCloseable {
    companion object { init { System.loadLibrary("lfm_native") } }

    private external fun nativeLoadModelFromPath(modelPath: String, draftModelPath: String, contextSize: Int): Boolean
    private external fun nativeGetLastError(): String
    private external fun nativeGenerate(prompt: String, maxTokens: Int): String
    private external fun nativeGenerateStream(prompt: String, maxTokens: Int, callback: Any)
    private external fun nativeGenerateToolStep(messagesJson: String, toolsJson: String, maxTokens: Int, callback: Any): String
    private external fun nativeUnloadModel()
    private external fun nativeGetBackendInfo(): String

    fun loadModelFromPath(modelPath: String, contextSize: Int = 8192): Boolean = nativeLoadModelFromPath(modelPath, "", contextSize)
    fun loadModelFromPath(modelPath: String, draftModelPath: String, contextSize: Int = 8192): Boolean = nativeLoadModelFromPath(modelPath, draftModelPath, contextSize)
    fun lastError(): String = nativeGetLastError()
    fun backendInfo(): String = nativeGetBackendInfo()
    fun generate(prompt: String, maxTokens: Int = 1024): String = nativeGenerate(prompt, maxTokens)

    fun generateStream(prompt: String, maxTokens: Int = 1024, onToken: (String) -> Unit, onStats: (Double, Long, Int, Int) -> Unit = { _, _, _, _ -> }): String {
        val callback = object {
            @Suppress("unused") fun onToken(text: String) { onToken(text) }
            @Suppress("unused") fun onStats(tokPerSec: Double, elapsedMs: Long, contextUsed: Int, contextSize: Int) { onStats(tokPerSec, elapsedMs, contextUsed, contextSize) }
        }
        nativeGenerateStream(prompt, maxTokens, callback); return ""
    }

    fun generateToolStep(messagesJson: String, toolsJson: String, maxTokens: Int = 384, onProgress: (String, Long) -> Unit = { _, _ -> }, onToken: (String) -> Unit = {}): String {
        val callback = object {
            @Suppress("unused") fun onProgress(stage: String, elapsedMs: Long) { onProgress(stage, elapsedMs) }
            @Suppress("unused") fun onToken(text: String) { onToken(text) }
        }
        return nativeGenerateToolStep(messagesJson, toolsJson, maxTokens, callback)
    }

    override fun close() { nativeUnloadModel() }
}
