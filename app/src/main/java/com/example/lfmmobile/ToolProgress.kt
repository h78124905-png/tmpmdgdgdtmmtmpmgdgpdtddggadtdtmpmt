package com.example.lfmmobile

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

/** Live, UI-observable progress for the synchronous native tool step. */
object ToolProgress {
    var stage by mutableStateOf("")
        private set
    var elapsedMs by mutableStateOf(0L)
        private set

    fun update(stage: String, elapsedMs: Long) {
        this.stage = stage
        this.elapsedMs = elapsedMs
    }

    fun reset() {
        stage = ""
        elapsedMs = 0L
    }
}

fun toolProgressLabel(stage: String): String = when {
    stage.startsWith("prefill ") -> "プロンプトを読み込み中（${stage.removePrefix("prefill ")})"
    else -> when (stage) {
    "validate_engine" -> "モデル確認"
    "jni_input_conversion" -> "入力変換"
    "parse_messages_json" -> "会話解析"
    "parse_tools_json" -> "ツール解析"
    "parse_messages_oaicompat" -> "会話形式解析"
    "parse_tools_oaicompat" -> "ツール形式解析"
    "chat_template_init" -> "テンプレート準備"
    "chat_template_apply" -> "プロンプト生成"
    "tokenize_tool_prompt" -> "トークン化"
    "init_tool_sampler", "init_tool_grammar", "init_tool_sampler_create" -> "サンプラー準備"
    "clear_tool_kv" -> "KVキャッシュ初期化"
    "prefill_tool_prompt" -> "プロンプトを読み込み中"
    "prefill_complete" -> "Prefill完了"
    "sample_first_tool_token" -> "最初のトークン計算"
    "generate_tool_tokens" -> "ツール呼び出し生成中"
    "parse_generated_tool_call" -> "ツール呼び出し解析"
    "build_result" -> "結果作成"
    "complete" -> "完了"
    else -> stage.ifBlank { "準備中" }
    }
}
