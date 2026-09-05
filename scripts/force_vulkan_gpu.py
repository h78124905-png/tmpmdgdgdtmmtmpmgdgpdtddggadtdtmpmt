from pathlib import Path

path = Path("app/src/main/cpp/native-lib.cpp")
text = path.read_text(encoding="utf-8")
needle = "        model_params.progress_callback_user_data = nullptr;\n\n        LOGI(\"[load] model load starting (Vulkan GPU offload requested)\");"
replacement = "        model_params.progress_callback_user_data = nullptr;\n\n        // Explicitly bind the model to the Vulkan device instead of relying on\n        // backend auto-selection. n_gpu_layers=-1 still requests all layers.\n        ggml_backend_dev_t vulkan_device = nullptr;\n        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {\n            ggml_backend_dev_t dev = ggml_backend_dev_get(i);\n            const char * name = ggml_backend_dev_name(dev);\n            if (name && std::string(name).find(\"Vulkan\") != std::string::npos) {\n                vulkan_device = dev;\n                break;\n            }\n        }\n        if (!vulkan_device) {\n            set_error(\"stage=gpu_backend; Vulkan GPU backend is not available\");\n            llama_model_free(model);\n            return JNI_FALSE;\n        }\n        ggml_backend_dev_t gpu_devices[] = { vulkan_device, nullptr };\n        model_params.devices = gpu_devices;\n        LOGI(\"[load] forcing Vulkan device: %s\", ggml_backend_dev_name(vulkan_device));\n\n        LOGI(\"[load] model load starting (Vulkan GPU offload requested)\");"
if needle not in text:
    raise SystemExit("GPU insertion point not found; refusing to modify source")
path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
