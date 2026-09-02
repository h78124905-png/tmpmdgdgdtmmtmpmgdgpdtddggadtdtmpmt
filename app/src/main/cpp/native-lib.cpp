#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <exception>
#include <string>
#include <vector>

#include "llama.h"
#include "common.h"
#include "sampling.h"

#define LOG_TAG "LfmMobile"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {
struct Engine {
    common_init_result_ptr init;
    common_sampler_ptr sampler;
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    const llama_vocab * vocab = nullptr;
    std::string last_error;
};
Engine g_engine;

void free_engine() {
    g_engine.sampler.reset();
    g_engine.init.reset();
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
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeLoadModelFromPath(
        JNIEnv * env, jobject, jstring model_path, jint context_size) {
    const std::string path = get_string(env, model_path);
    if (path.empty()) {
        set_error("stage=path; model path is empty");
        return JNI_FALSE;
    }

    free_engine();
    g_engine.last_error.clear();
    common_init();
    llama_backend_init();

    try {
        common_params params;
        params.model.path = path;
        params.n_ctx = std::max(512, static_cast<int>(context_size));
        params.n_batch = std::min(params.n_ctx, 512);
        params.n_ubatch = std::min(params.n_batch, 512);
        params.n_parallel = 1;

        auto init = common_init_from_params(params);
        if (!init) {
            set_error("stage=model_init; common_init_from_params returned no result");
            return JNI_FALSE;
        }
        if (!init->model()) {
            set_error("stage=model_init; llama.cpp could not load the selected GGUF");
            return JNI_FALSE;
        }
        if (!init->context()) {
            set_error("stage=context_init; GGUF loaded but llama.cpp could not create the context");
            return JNI_FALSE;
        }

        common_params_sampling sampling;
        sampling.temp = 0.7f;
        sampling.top_k = 40;
        sampling.top_p = 0.95f;
        auto sampler = common_sampler_init(init->model(), sampling);
        if (!sampler) {
            set_error("stage=sampler_init; model loaded but sampler initialization failed");
            return JNI_FALSE;
        }

        g_engine.init = std::move(init);
        g_engine.sampler.reset(sampler);
        g_engine.model = g_engine.init->model();
        g_engine.context = g_engine.init->context();
        g_engine.vocab = llama_model_get_vocab(g_engine.model);
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
    if (!g_engine.model || !g_engine.context || !g_engine.sampler)
        return env->NewStringUTF("[model not loaded]");

    const std::string prompt_text = get_string(env, prompt);
    if (prompt_text.empty()) return env->NewStringUTF("");

    llama_memory_clear(llama_get_memory(g_engine.context), false);
    common_params_sampling sampling;
    sampling.temp = 0.7f;
    sampling.top_k = 40;
    sampling.top_p = 0.95f;
    g_engine.sampler.reset(common_sampler_init(g_engine.model, sampling));
    if (!g_engine.sampler) return env->NewStringUTF("[sampler init failed]");

    const llama_tokens input = common_tokenize(g_engine.context, prompt_text, true, true);
    if (input.empty()) return env->NewStringUTF("[tokenization failed]");

    const uint32_t n_ctx = llama_n_ctx(g_engine.context);
    if (input.size() + 1 >= n_ctx)
        return env->NewStringUTF("[prompt exceeds context]");

    const uint32_t batch_size = std::min<uint32_t>(llama_n_batch(g_engine.context), input.size());
    llama_batch batch = llama_batch_init(batch_size, 0, 1);
    for (size_t i = 0; i < input.size(); ++i) {
        const bool want_logits = (i + 1 == input.size());
        common_batch_add(batch, input[i], static_cast<llama_pos>(i), {0}, want_logits);
    }

    if (llama_decode(g_engine.context, batch) != 0) {
        llama_batch_free(batch);
        return env->NewStringUTF("[prompt decode failed]");
    }

    std::string output;
    int n_past = static_cast<int>(input.size());
    const int n_predict = std::max(1, static_cast<int>(max_tokens));

    for (int i = 0; i < n_predict; ++i) {
        const llama_token next = common_sampler_sample(g_engine.sampler.get(), g_engine.context, -1);
        if (llama_vocab_is_eog(g_engine.vocab, next)) break;

        output += common_token_to_piece(g_engine.context, next);
        common_sampler_accept(g_engine.sampler.get(), next, true);

        common_batch_clear(batch);
        common_batch_add(batch, next, n_past++, {0}, true);
        if (llama_decode(g_engine.context, batch) != 0) break;
    }

    llama_batch_free(batch);
    return env->NewStringUTF(output.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeUnloadModel(JNIEnv *, jobject) {
    free_engine();
    llama_backend_free();
}
