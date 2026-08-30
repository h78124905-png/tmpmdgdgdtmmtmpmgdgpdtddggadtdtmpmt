package com.example.lfmmobile

class LlamaEngine : AutoCloseable {
    companion object {
        init {
            System.loadLibrary("lfm_native")
        }
    }

    private external fun nativeLoadModel(path: String, contextSize: Int): Boolean
    private external fun nativeGenerate(prompt: String, maxTokens: Int): String
    private external fun nativeUnloadModel()

    fun loadModel(path: String, contextSize: Int = 4096): Boolean =
        nativeLoadModel(path, contextSize)

    fun generate(prompt: String, maxTokens: Int = 128): String =
        nativeGenerate(prompt, maxTokens)

    override fun close() {
        nativeUnloadModel()
    }
}
