package com.example.lfmmobile

class LlamaEngine : AutoCloseable {
    companion object {
        init { System.loadLibrary("lfm_native") }
    }

    private external fun nativeLoadModelFromPath(modelPath: String, draftModelPath: String, contextSize: Int): Boolean
    private external fun nativeGetLastError(): String
    private external fun nativeGenerate(prompt: String, maxTokens: Int): String
    private external fun nativeGenerateStream(prompt: String, maxTokens: Int, callback: Any): String
    private external fun nativeUnloadModel()

    fun loadModelFromPath(modelPath: String, contextSize: Int = 4096): Boolean =
        nativeLoadModelFromPath(modelPath, "", contextSize)

    fun loadModelFromPath(modelPath: String, draftModelPath: String, contextSize: Int = 4096): Boolean =
        nativeLoadModelFromPath(modelPath, draftModelPath, contextSize)

    fun lastError(): String = nativeGetLastError()

    fun generate(prompt: String, maxTokens: Int = 128): String =
        nativeGenerate(prompt, maxTokens)

    fun generateStream(
        prompt: String,
        maxTokens: Int = 128,
        onToken: (String) -> Unit,
        onStats: (tokPerSec: Double, elapsedMs: Long, gpu: String, contextUsed: Int, contextSize: Int) -> Unit = { _, _, _, _, _ -> }
    ): String {
        val emitToken = onToken
        val emitStats = onStats
        val callback = object {
            @Suppress("unused")
            fun onToken(text: String) { emitToken(text) }

            @Suppress("unused")
            fun onStats(tokPerSec: Double, elapsedMs: Long, gpu: String, contextUsed: Int, contextSize: Int) {
                emitStats(tokPerSec, elapsedMs, gpu, contextUsed, contextSize)
            }
        }
        return nativeGenerateStream(prompt, maxTokens, callback)
    }

    override fun close() { nativeUnloadModel() }
}
