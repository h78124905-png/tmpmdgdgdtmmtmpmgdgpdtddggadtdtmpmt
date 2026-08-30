package com.example.lfmmobile

class LlamaEngine : AutoCloseable {
    companion object {
        init { System.loadLibrary("lfm_native") }
    }

    private external fun nativeLoadModelFd(role: Int, fd: Int, contextSize: Int): Boolean
    private external fun nativeGenerate(role: Int, prompt: String, maxTokens: Int): String
    private external fun nativeUnloadModel()

    fun loadModelFromFd(role: Int, fd: Int, contextSize: Int = 4096): Boolean =
        nativeLoadModelFd(role, fd, contextSize)

    fun generate(role: Int, prompt: String, maxTokens: Int = 128): String =
        nativeGenerate(role, prompt, maxTokens)

    override fun close() { nativeUnloadModel() }
}
