from pathlib import Path


def replace_once(path: str, old: str, new: str):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one match, found {count}: {old[:100]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# LFM2.5's official chat template opens <think> on the assistant generation
# prompt. Therefore the first streamed token is thinking content, not answer
# content. Starting the parser in UNKNOWN incorrectly puts the reasoning in
# the answer bubble until </think> arrives.
replace_once(
    "app/src/main/java/com/example/lfmmobile/MainActivity.kt",
    "    private var mode = Mode.UNKNOWN\n",
    "    private var mode = Mode.THINKING\n",
)

# User-selectable inference backend. Default to GPU/Vulkan because this is the
# intended fast path on the target Android device, while retaining CPU as a
# real comparison/fallback mode.
replace_once(
    "app/src/main/java/com/example/lfmmobile/MainActivity.kt",
    "    var generationStats by remember { mutableStateOf(GenerationStats()) }\n",
    "    var generationStats by remember { mutableStateOf(GenerationStats()) }\n    var useGpu by remember { mutableStateOf(true) }\n",
)

replace_once(
    "app/src/main/java/com/example/lfmmobile/MainActivity.kt",
    "                        engine.loadModelFromPath(targetFile.absolutePath, draftFile.absolutePath, contextSize)\n",
    "                        engine.loadModelFromPath(targetFile.absolutePath, draftFile.absolutePath, contextSize, useGpu)\n",
)
replace_once(
    "app/src/main/java/com/example/lfmmobile/MainActivity.kt",
    "                        engine.loadModelFromPath(targetFile.absolutePath, contextSize)\n",
    "                        engine.loadModelFromPath(targetFile.absolutePath, contextSize, useGpu)\n",
)

replace_once(
    "app/src/main/java/com/example/lfmmobile/MainActivity.kt",
    "                    OutlinedTextField(maxTokens.toString(), { it.toIntOrNull()?.coerceIn(1, 8192)?.let { v -> maxTokens = v } }, label = { Text(\"Max tokens\") }, singleLine = true)\n",
    "                    OutlinedTextField(maxTokens.toString(), { it.toIntOrNull()?.coerceIn(1, 8192)?.let { v -> maxTokens = v } }, label = { Text(\"Max tokens\") }, singleLine = true)\n                    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {\n                        Column(Modifier.weight(1f)) {\n                            Text(if (useGpu) \"GPU (Vulkan)\" else \"CPU\", fontWeight = FontWeight.SemiBold)\n                            Text(if (useGpu) \"Use Vulkan GPU acceleration\" else \"Use CPU inference\", style = MaterialTheme.typography.bodySmall)\n                        }\n                        Switch(checked = useGpu, onCheckedChange = { useGpu = it; loaded = false })\n                    }\n",
)

# Update the JNI bridge to pass the selected backend into native loading.
replace_once(
    "app/src/main/java/com/example/lfmmobile/LlamaEngine.kt",
    "    private external fun nativeLoadModelFromPath(modelPath: String, draftModelPath: String, contextSize: Int): Boolean\n",
    "    private external fun nativeLoadModelFromPath(modelPath: String, draftModelPath: String, contextSize: Int, useGpu: Boolean): Boolean\n",
)
replace_once(
    "app/src/main/java/com/example/lfmmobile/LlamaEngine.kt",
    "    fun loadModelFromPath(modelPath: String, contextSize: Int = 4096): Boolean =\n        nativeLoadModelFromPath(modelPath, \"\", contextSize)\n\n    fun loadModelFromPath(modelPath: String, draftModelPath: String, contextSize: Int = 4096): Boolean =\n        nativeLoadModelFromPath(modelPath, draftModelPath, contextSize)\n",
    "    fun loadModelFromPath(modelPath: String, contextSize: Int = 4096, useGpu: Boolean = true): Boolean =\n        nativeLoadModelFromPath(modelPath, \"\", contextSize, useGpu)\n\n    fun loadModelFromPath(modelPath: String, draftModelPath: String, contextSize: Int = 4096, useGpu: Boolean = true): Boolean =\n        nativeLoadModelFromPath(modelPath, draftModelPath, contextSize, useGpu)\n",
)

# Native backend selection. GPU mode explicitly binds llama.cpp to a Vulkan
# device; CPU mode sets zero GPU layers and does not bind a GPU device.
replace_once(
    "app/src/main/cpp/native-lib.cpp",
    "Java_com_example_lfmmobile_LlamaEngine_nativeLoadModelFromPath(\n        JNIEnv * env, jobject, jstring model_path, jstring draft_model_path, jint context_size) {",
    "Java_com_example_lfmmobile_LlamaEngine_nativeLoadModelFromPath(\n        JNIEnv * env, jobject, jstring model_path, jstring draft_model_path, jint context_size, jboolean use_gpu) {",
)
replace_once(
    "app/src/main/cpp/native-lib.cpp",
    "        llama_model_params model_params = llama_model_default_params();\n        model_params.n_gpu_layers = -1;\n        model_params.progress_callback = load_progress;\n        model_params.progress_callback_user_data = nullptr;\n\n        LOGI(\"[load] model load starting (Vulkan GPU offload requested)\");\n",
    "        llama_model_params model_params = llama_model_default_params();\n        model_params.n_gpu_layers = use_gpu ? -1 : 0;\n        model_params.progress_callback = load_progress;\n        model_params.progress_callback_user_data = nullptr;\n\n        ggml_backend_dev_t vulkan_device = nullptr;\n        if (use_gpu) {\n            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {\n                ggml_backend_dev_t dev = ggml_backend_dev_get(i);\n                const char * name = ggml_backend_dev_name(dev);\n                if (name && (std::string(name).find(\"Vulkan\") != std::string::npos ||\n                             std::string(name).find(\"vulkan\") != std::string::npos)) {\n                    vulkan_device = dev;\n                    break;\n                }\n            }\n            if (!vulkan_device) {\n                set_error(\"stage=gpu_backend; Vulkan GPU backend is not available\");\n                return JNI_FALSE;\n            }\n            ggml_backend_dev_t gpu_devices[] = { vulkan_device, nullptr };\n            model_params.devices = gpu_devices;\n            const char * desc = ggml_backend_dev_description(vulkan_device);\n            g_engine.gpu_name = desc && *desc ? std::string(\"Vulkan · \") + desc : std::string(\"Vulkan · \") + ggml_backend_dev_name(vulkan_device);\n            LOGI(\"[backend] forcing Vulkan GPU: %s\", g_engine.gpu_name.c_str());\n        } else {\n            g_engine.gpu_name = \"CPU\";\n            LOGI(\"[backend] CPU inference selected\");\n        }\n\n        LOGI(\"[load] model load starting; backend=%s\", g_engine.gpu_name.c_str());\n",
)
replace_once(
    "app/src/main/cpp/native-lib.cpp",
    "            spec_params.speculative.draft.n_gpu_layers = -1;\n",
    "            spec_params.speculative.draft.n_gpu_layers = use_gpu ? -1 : 0;\n",
)

print("Applied CPU/Vulkan selector and LFM2.5 thinking parser fix")
