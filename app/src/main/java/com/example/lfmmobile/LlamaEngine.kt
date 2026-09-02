package com.example.lfmmobile

class LlamaEngine : AutoCloseable {
    companion object {
        init { System.loadLibrary("lfm_native") }
    }

    private external fun nativeLoadModelFromPath(modelPath: String, contextSize: Int): Boolean
    private external fun nativeGetLastError(): String
    private external fun nativeGenerate(prompt: String, maxTokens: Int): String
    private external fun nativeGenerateStream(prompt: String, maxTokens: Int, callback: Any): String
    private external fun nativeUnloadModel()

    fun loadModelFromPath(modelPath: String, contextSize: Int = 4096): Boolean =
        nativeLoadModelFromPath(modelPath, contextSize)

    fun lastError(): String = nativeGetLastError()

    fun generate(prompt: String, maxTokens: Int = 128): String =
        nativeGenerate(prompt, maxTokens)

    fun generateStream(prompt: String, maxTokens: Int = 128, onToken: (String) -> Unit): String {
        // Keep a different name for the Kotlin lambda so the callback method
        // does not accidentally call itself recursively.
        val emitToken = onToken
        val callback = object {
            @Suppress("unused")
            fun onToken(text: String) {
                emitToken(text)
            }
        }
        return nativeGenerateStream(prompt, maxTokens, callback)
    }

    override fun close() { nativeUnloadModel() }
}
