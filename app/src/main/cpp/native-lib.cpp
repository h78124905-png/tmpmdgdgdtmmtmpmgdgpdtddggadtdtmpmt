#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <exception>
#include <string>
#include <thread>
#include <vector>

#include "llama.h"
#include "common.h"
#include "sampling.h"
#include "chat.h"

#define LOG_TAG "LfmMobile"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace {
struct Engine {
    common_sampler_ptr sampler;
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    const llama_vocab * vocab = nullptr;
    std::string last_error;
    bool backend_initialized = false;
};
Engine g_engine;

void free_engine() {
    g_engine.sampler.reset();
    if (g_engine.context) {
        LOGI("[load] freeing context");
        llama_free(g_engine.context);
    }
    if (g_engine.model) {
        LOGI("[load] freeing model");
        llama_model_free(g_engine.model);
    }
    g_engine.model = nullptr;
    g_engine.context = nullptr;
    g_engine.vocab = nullptr;
}

void set_error(const std::string & s) {
    g_engine.last_error = s;
    LOGE("%s", s.c_str());
}

std::string get_string(JNIEnv * env, jstring value) {
    if (!value) return {};
    const char * chars = env->GetStringUTFChars(value, nullptr);
    if (!chars) return {};
    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

bool load_progress(float progress, void *) {
    static int last_percent = -1;
    const int percent = std::clamp(static_cast<int>(progress * 100.0f), 0, 100);
    if (percent == 100 || percent >= last_percent + 10) {
        last_percent = percent;
        LOGI("[load] llama_model_load_from_file progress=%d%%", percent);
    }
    return true;
}

std::vector<common_chat_msg> build_messages(const std::string & prompt_text) {
    std::vector<common_chat_msg> messages;
    messages.push_back({"system", "You are a helpful local assistant. Answer naturally and accurately."});
    size_t pos = 0;
    while (pos < prompt_text.size()) {
        const size_t end = prompt_text.find('\n', pos);
        const std::string line = prompt_text.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        if (line.rfind("User: ", 0) == 0) {
            messages.push_back({"user", line.substr(6)});
        } else if (line.rfind("Assistant: ", 0) == 0) {
            messages.push_back({"assistant", line.substr(11)});
        }
        if (end == std::string::npos) break;
        pos = end + 1;
    }
    return messages;
}

std::string generate_impl(JNIEnv * env, const std::string & prompt_text, int max_tokens, jobject callback) {
    if (!g_engine.model || !g_engine.context || !g_engine.sampler)
        return "[model not loaded]";
    if (prompt_text.empty()) return {};

    llama_memory_clear(llama_get_memory(g_engine.context), false);
    common_params_sampling sampling;
    sampling.temp = 0.7f;
    sampling.top_k = 40;
    sampling.top_p = 0.95f;
    g_engine.sampler.reset(common_sampler_init(g_engine.model, sampling));
    if (!g_engine.sampler) return "[sampler init failed]";

    auto messages = build_messages(prompt_text);
    if (messages.size() <= 1) return "[chat format failed: no user message]";

    auto templates = common_chat_templates_init(g_engine.model, "");
    if (!templates) return "[chat template init failed]";

    common_chat_templates_inputs chat_inputs;
    chat_inputs.messages = std::move(messages);
    chat_inputs.add_generation_prompt = true;
    chat_inputs.use_jinja = true;

    const common_chat_params chat_params = common_chat_templates_apply(templates.get(), chat_inputs);
    if (chat_params.prompt.empty()) return "[chat template produced an empty prompt]";

    const llama_tokens input = common_tokenize(g_engine.context, chat_params.prompt, true, true);
    if (input.empty()) return "[tokenization failed]";

    const uint32_t n_ctx = llama_n_ctx(g_engine.context);
    if (input.size() + 1 >= n_ctx) return "[prompt exceeds context]";

    const uint32_t batch_size = std::min<uint32_t>(llama_n_batch(g_engine.context), input.size());
    llama_batch batch = llama_batch_init(batch_size, 0, 1);
    for (size_t i = 0; i < input.size(); ++i) {
        const bool want_logits = (i + 1 == input.size());
        common_batch_add(batch, input[i], static_cast<llama_pos>(i), {0}, want_logits);
    }

    if (llama_decode(g_engine.context, batch) != 0) {
        llama_batch_free(batch);
        return "[prompt decode failed]";
    }

    jmethodID on_token = nullptr;
    if (callback) {
        jclass callback_class = env->GetObjectClass(callback);
        on_token = env->GetMethodID(callback_class, "onToken", "(Ljava/lang/String;)V");
        env->DeleteLocalRef(callback_class);
        if (!on_token) {
            if (env->ExceptionCheck()) env->ExceptionClear();
            llama_batch_free(batch);
            return "[stream callback method not found]";
        }
    }

    std::string output;
    int n_past = static_cast<int>(input.size());
    const int n_predict = std::max(1, max_tokens);

    for (int i = 0; i < n_predict; ++i) {
        const llama_token next = common_sampler_sample(g_engine.sampler.get(), g_engine.context, -1);
        if (llama_vocab_is_eog(g_engine.vocab, next)) break;

        const std::string piece = common_token_to_piece(g_engine.context, next);
        output += piece;

        if (callback && on_token && !piece.empty()) {
            jstring jpiece = env->NewStringUTF(piece.c_str());
            env->CallVoidMethod(callback, on_token, jpiece);
            env->DeleteLocalRef(jpiece);
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
                llama_batch_free(batch);
                return "[stream callback failed]";
            }
        }

        common_sampler_accept(g_engine.sampler.get(), next, true);
        common_batch_clear(batch);
        common_batch_add(batch, next, n_past++, {0}, true);
        if (llama_decode(g_engine.context, batch) != 0) break;
    }

    llama_batch_free(batch);
    return output;
}
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeLoadModelFromPath(
        JNIEnv * env, jobject, jstring model_path, jint context_size) {
    LOGI("[load] JNI entered");
    const std::string path = get_string(env, model_path);
    if (path.empty()) {
        set_error("stage=path; model path is empty");
        return JNI_FALSE;
    }
    LOGI("[load] path obtained: %s", path.c_str());

    free_engine();
    g_engine.last_error.clear();
    if (!g_engine.backend_initialized) {
        LOGI("[load] llama_backend_init starting");
        llama_backend_init();
        g_engine.backend_initialized = true;
        LOGI("[load] llama_backend_init completed");
    }

    try {
        llama_model_params model_params = llama_model_default_params();
        // Use all eligible layers on the Vulkan backend. If Vulkan cannot be
        // initialized on a device, llama.cpp can report the backend failure
        // instead of silently pretending this is a CPU-only build.
        model_params.n_gpu_layers = -1;
        model_params.progress_callback = load_progress;
        model_params.progress_callback_user_data = nullptr;

        LOGI("[load] model load starting (Vulkan GPU offload)");
        llama_model * model = llama_model_load_from_file(path.c_str(), model_params);
        if (!model) {
            set_error("stage=model_load; llama_model_load_from_file returned null (Vulkan backend may be unavailable)");
            return JNI_FALSE;
        }
        LOGI("[load] model load completed with Vulkan GPU offload requested");

        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (!vocab) {
            llama_model_free(model);
            set_error("stage=model_validation; loaded model has no vocabulary");
            return JNI_FALSE;
        }

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = std::max(512, static_cast<int>(context_size));
        context_params.n_batch = std::min(context_params.n_ctx, 256u);
        context_params.n_ubatch = std::min(context_params.n_batch, 256u);
        context_params.n_seq_max = 1;
        context_params.n_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency() / 2));
        context_params.n_threads_batch = context_params.n_threads;

        LOGI("[load] context creation starting: n_ctx=%u n_batch=%u threads=%d",
             context_params.n_ctx, context_params.n_batch, context_params.n_threads);
        llama_context * context = llama_init_from_model(model, context_params);
        if (!context) {
            llama_model_free(model);
            set_error("stage=context_init; GGUF loaded but llama_init_from_model returned null");
            return JNI_FALSE;
        }
        LOGI("[load] context creation completed");

        common_params_sampling sampling;
        sampling.temp = 0.7f;
        sampling.top_k = 40;
        sampling.top_p = 0.95f;
        auto sampler = common_sampler_init(model, sampling);
        if (!sampler) {
            llama_free(context);
            llama_model_free(model);
            set_error("stage=sampler_init; model and context loaded but sampler initialization failed");
            return JNI_FALSE;
        }

        g_engine.model = model;
        g_engine.context = context;
        g_engine.vocab = vocab;
        g_engine.sampler.reset(sampler);
        LOGI("[load] model load completed successfully (Vulkan GPU offload requested)");
        return JNI_TRUE;
    } catch (const std::exception & e) {
        set_error(std::string("stage=exception; ") + e.what());
        free_engine();
        return JNI_FALSE;
    } catch (...) {
        set_error("stage=exception; unknown native exception");
        free_engine();
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeGetLastError(JNIEnv * env, jobject) {
    return env->NewStringUTF(g_engine.last_error.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeGenerate(
        JNIEnv * env, jobject, jstring prompt, jint max_tokens) {
    return env->NewStringUTF(generate_impl(env, get_string(env, prompt), max_tokens, nullptr).c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeGenerateStream(
        JNIEnv * env, jobject, jstring prompt, jint max_tokens, jobject callback) {
    try {
        const std::string result = generate_impl(env, get_string(env, prompt), max_tokens, callback);
        return env->NewStringUTF(result.c_str());
    } catch (const std::exception & e) {
        return env->NewStringUTF((std::string("[stream exception] ") + e.what()).c_str());
    } catch (...) {
        return env->NewStringUTF("[stream exception]");
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeUnloadModel(JNIEnv *, jobject) {
    free_engine();
    if (g_engine.backend_initialized) {
        LOGI("[load] llama_backend_free");
        llama_backend_free();
        g_engine.backend_initialized = false;
    }
}
