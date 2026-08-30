package com.example.lfmmobile

class LlamaEngine : AutoCloseable {
    companion object {
        init { System.loadLibrary("lfm_native") }
    }

    private external fun nativeLoadModelsFromFds(targetFd: Int, draftFd: Int, contextSize: Int, draftMax: Int): Boolean
    private external fun nativeGenerate(prompt: String, maxTokens: Int): String
    private external fun nativeUnloadModel()

    fun loadModelsFromFds(targetFd: Int, draftFd: Int, contextSize: Int = 4096, draftMax: Int = 7): Boolean =
        nativeLoadModelsFromFds(targetFd, draftFd, contextSize, draftMax)

    fun generate(prompt: String, maxTokens: Int = 128): String = nativeGenerate(prompt, maxTokens)

    override fun close() { nativeUnloadModel() }
}
