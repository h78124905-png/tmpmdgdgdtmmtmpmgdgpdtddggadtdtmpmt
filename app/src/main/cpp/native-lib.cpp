#include <jni.h>
#include <android/log.h>
#include <unistd.h>

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

Engine g_target;
Engine g_dspark;

Engine & engine_for(int role) { return role == 1 ? g_dspark : g_target; }

void free_engine(Engine & e) {
    if (e.sampler) { llama_sampler_free(e.sampler); e.sampler = nullptr; }
    if (e.context) { llama_free(e.context); e.context = nullptr; }
    if (e.model) { llama_model_free(e.model); e.model = nullptr; }
    e.vocab = nullptr;
}

std::string token_piece(Engine & e, llama_token token) {
    char buffer[4096];
    const int n = llama_token_to_piece(e.vocab, token, buffer, sizeof(buffer), 0, true);
    return n > 0 ? std::string(buffer, n) : std::string();
}

bool load_model_fd(Engine & e, int fd, int context_size) {
    if (fd < 0) return false;
    free_engine(e);
    llama_backend_init();

    const std::string path = "/proc/self/fd/" + std::to_string(fd);
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    e.model = llama_model_load_from_file(path.c_str(), model_params);
    close(fd);
    if (!e.model) { LOGE("Failed to load model from fd"); free_engine(e); return false; }

    e.vocab = llama_model_get_vocab(e.model);
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(std::max(256, context_size));
    ctx_params.n_batch = std::min(ctx_params.n_ctx, 512u);
    ctx_params.no_perf = true;
    e.context = llama_init_from_model(e.model, ctx_params);
    if (!e.context) { free_engine(e); return false; }

    auto sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = true;
    e.sampler = llama_sampler_chain_init(sampler_params);
    if (!e.sampler) { free_engine(e); return false; }
    llama_sampler_chain_add(e.sampler, llama_sampler_init_greedy());
    return true;
}
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeLoadModelFd(
        JNIEnv *, jobject, jint role, jint fd, jint context_size) {
    return load_model_fd(engine_for(role), fd, context_size) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeGenerate(
        JNIEnv * env, jobject, jint role, jstring prompt, jint max_tokens) {
    Engine & e = engine_for(role);
    if (!e.model || !e.context || !e.sampler) return env->NewStringUTF("[model not loaded]");

    const char * prompt_chars = env->GetStringUTFChars(prompt, nullptr);
    if (!prompt_chars) return env->NewStringUTF("");
    const std::string prompt_text(prompt_chars);
    env->ReleaseStringUTFChars(prompt, prompt_chars);

    const int token_count = -llama_tokenize(
        e.vocab, prompt_text.c_str(), prompt_text.size(), nullptr, 0, true, true);
    if (token_count <= 0) return env->NewStringUTF("");
    std::vector<llama_token> tokens(token_count);
    if (llama_tokenize(e.vocab, prompt_text.c_str(), prompt_text.size(),
                       tokens.data(), tokens.size(), true, true) < 0) {
        return env->NewStringUTF("[tokenization failed]");
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(e.context, batch) != 0) return env->NewStringUTF("[decode failed]");

    std::string output;
    const int limit = std::max(1, static_cast<int>(max_tokens));
    for (int i = 0; i < limit; ++i) {
        llama_token token = llama_sampler_sample(e.sampler, e.context, -1);
        if (llama_vocab_is_eog(e.vocab, token)) break;
        output += token_piece(e, token);
        batch = llama_batch_get_one(&token, 1);
        if (llama_decode(e.context, batch) != 0) break;
    }
    return env->NewStringUTF(output.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeUnloadModel(JNIEnv *, jobject) {
    free_engine(g_target);
    free_engine(g_dspark);
    llama_backend_free();
}
