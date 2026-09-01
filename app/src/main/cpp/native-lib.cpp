#include <jni.h>
#include <android/log.h>
#include <unistd.h>
#include <algorithm>
#include <exception>
#include <string>

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
};
Engine g_engine;

void free_engine() {
    g_engine.sampler.reset();
    g_engine.init.reset();
    g_engine.model = nullptr;
    g_engine.context = nullptr;
    g_engine.vocab = nullptr;
}
std::string fd_path(int fd) { return "/proc/self/fd/" + std::to_string(fd); }
std::string get_prompt(JNIEnv * env, jstring prompt) {
    const char * chars = env->GetStringUTFChars(prompt, nullptr);
    if (!chars) return {};
    std::string text(chars);
    env->ReleaseStringUTFChars(prompt, chars);
    return text;
}
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeLoadModelFromFd(
        JNIEnv *, jobject, jint model_fd, jint context_size) {
    if (model_fd < 0) return JNI_FALSE;
    free_engine();
    common_init();
    llama_backend_init();

    try {
        common_params params;
        params.model.path = fd_path(model_fd);
        params.n_ctx = std::max(512, static_cast<int>(context_size));
        params.n_batch = std::min(params.n_ctx, 512);
        params.n_ubatch = std::min(params.n_batch, 512);
        params.n_parallel = 1;

        auto init = common_init_from_params(params);
        if (!init || !init->model() || !init->context()) {
            LOGE("target model initialization failed");
            close(model_fd);
            return JNI_FALSE;
        }

        common_params_sampling sampling;
        sampling.temp = 0.7f;
        sampling.top_k = 40;
        sampling.top_p = 0.95f;
        auto sampler = common_sampler_init(init->model(), sampling);
        if (!sampler) {
            LOGE("sampler initialization failed");
            close(model_fd);
            return JNI_FALSE;
        }

        close(model_fd);
        g_engine.init = std::move(init);
        g_engine.sampler.reset(sampler);
        g_engine.model = g_engine.init->model();
        g_engine.context = g_engine.init->context();
        g_engine.vocab = llama_model_get_vocab(g_engine.model);
        return JNI_TRUE;
    } catch (const std::exception & e) {
        LOGE("model load failed: %s", e.what());
        close(model_fd);
        free_engine();
        return JNI_FALSE;
    } catch (...) {
        LOGE("model load failed with unknown exception");
        close(model_fd);
        free_engine();
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeGenerate(
        JNIEnv * env, jobject, jstring prompt, jint max_tokens) {
    if (!g_engine.model || !g_engine.context || !g_engine.sampler)
        return env->NewStringUTF("[model not loaded]");

    const std::string prompt_text = get_prompt(env, prompt);
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

    llama_batch batch = llama_batch_init(
        std::min<uint32_t>(llama_n_batch(g_engine.context), input.size()), 0, 1);
    for (size_t i = 0; i < input.size(); ++i)
        common_batch_add(batch, input[i], static_cast<llama_pos>(i), {0}, true);

    if (llama_decode(g_engine.context, batch) != 0) {
        llama_batch_free(batch);
        return env->NewStringUTF("[prompt decode failed]");
    }

    std::string output;
    llama_token last = input.back();
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
        last = next;
    }

    llama_batch_free(batch);
    return env->NewStringUTF(output.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeUnloadModel(JNIEnv *, jobject) {
    free_engine();
    llama_backend_free();
}
