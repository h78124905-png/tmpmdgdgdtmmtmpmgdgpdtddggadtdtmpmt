#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <string>
#include <vector>

#include "llama.h"

#define LOG_TAG "LfmMobile"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

struct Engine {
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    llama_sampler * sampler = nullptr;
    const llama_vocab * vocab = nullptr;
};

Engine g_engine;

void free_engine() {
    if (g_engine.sampler) {
        llama_sampler_free(g_engine.sampler);
        g_engine.sampler = nullptr;
    }
    if (g_engine.context) {
        llama_free(g_engine.context);
        g_engine.context = nullptr;
    }
    if (g_engine.model) {
        llama_model_free(g_engine.model);
        g_engine.model = nullptr;
    }
    g_engine.vocab = nullptr;
}

std::string token_piece(llama_token token) {
    char buffer[4096];
    const int n = llama_token_to_piece(g_engine.vocab, token, buffer, sizeof(buffer), 0, true);
    return n > 0 ? std::string(buffer, n) : std::string();
}

} // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeLoadModel(
        JNIEnv * env, jobject, jstring model_path, jint context_size) {
    const char * path = env->GetStringUTFChars(model_path, nullptr);
    if (!path) return JNI_FALSE;

    free_engine();
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    g_engine.model = llama_model_load_from_file(path, model_params);
    env->ReleaseStringUTFChars(model_path, path);

    if (!g_engine.model) {
        LOGE("Failed to load model");
        free_engine();
        return JNI_FALSE;
    }

    g_engine.vocab = llama_model_get_vocab(g_engine.model);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(std::max(256, static_cast<int>(context_size)));
    ctx_params.n_batch = std::min(ctx_params.n_ctx, 512u);
    ctx_params.no_perf = true;

    g_engine.context = llama_init_from_model(g_engine.model, ctx_params);
    if (!g_engine.context) {
        LOGE("Failed to create llama context");
        free_engine();
        return JNI_FALSE;
    }

    auto sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = true;
    g_engine.sampler = llama_sampler_chain_init(sampler_params);
    if (!g_engine.sampler) {
        free_engine();
        return JNI_FALSE;
    }
    llama_sampler_chain_add(g_engine.sampler, llama_sampler_init_greedy());

    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeGenerate(
        JNIEnv * env, jobject, jstring prompt, jint max_tokens) {
    if (!g_engine.model || !g_engine.context || !g_engine.sampler) {
        return env->NewStringUTF("[model not loaded]");
    }

    const char * prompt_chars = env->GetStringUTFChars(prompt, nullptr);
    if (!prompt_chars) return env->NewStringUTF("");
    const std::string prompt_text(prompt_chars);
    env->ReleaseStringUTFChars(prompt, prompt_chars);

    const int token_count = -llama_tokenize(
            g_engine.vocab, prompt_text.c_str(), prompt_text.size(), nullptr, 0, true, true);
    if (token_count <= 0) return env->NewStringUTF("");

    std::vector<llama_token> tokens(token_count);
    if (llama_tokenize(g_engine.vocab, prompt_text.c_str(), prompt_text.size(),
                       tokens.data(), tokens.size(), true, true) < 0) {
        return env->NewStringUTF("[tokenization failed]");
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(g_engine.context, batch) != 0) {
        return env->NewStringUTF("[decode failed]");
    }

    std::string output;
    const int limit = std::max(1, static_cast<int>(max_tokens));
    for (int i = 0; i < limit; ++i) {
        llama_token token = llama_sampler_sample(g_engine.sampler, g_engine.context, -1);
        if (llama_vocab_is_eog(g_engine.vocab, token)) break;

        output += token_piece(token);
        batch = llama_batch_get_one(&token, 1);
        if (llama_decode(g_engine.context, batch) != 0) break;
    }

    return env->NewStringUTF(output.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeUnloadModel(JNIEnv *, jobject) {
    free_engine();
    llama_backend_free();
}
