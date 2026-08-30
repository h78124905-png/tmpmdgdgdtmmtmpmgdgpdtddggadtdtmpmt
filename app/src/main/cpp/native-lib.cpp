#include <jni.h>
#include <android/log.h>
#include <unistd.h>
#include <algorithm>
#include <exception>
#include <string>
#include <vector>

#include "llama.h"
#include "common.h"
#include "speculative.h"
#include "sampling.h"

#define LOG_TAG "LfmMobile"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {
struct Engine {
    common_init_result_ptr target_init;
    common_speculative_init_result_ptr draft_init;
    common_speculative_ptr spec;
    common_sampler_ptr sampler;
    llama_model * target_model = nullptr;
    llama_context * target_context = nullptr;
    llama_context * draft_context = nullptr;
    const llama_vocab * vocab = nullptr;
    int context_size = 4096;
    int draft_max = 7;
    std::string last_error;
};
Engine g_engine;

void free_engine() {
    g_engine.spec.reset();
    g_engine.sampler.reset();
    g_engine.draft_init.reset();
    g_engine.target_init.reset();
    g_engine.target_model = nullptr;
    g_engine.target_context = nullptr;
    g_engine.draft_context = nullptr;
    g_engine.vocab = nullptr;
    g_engine.last_error.clear();
}

void fail(const std::string & msg) {
    g_engine.last_error = msg;
    LOGE("%s", msg.c_str());
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
Java_com_example_lfmmobile_LlamaEngine_nativeLoadModelsFromFds(
        JNIEnv *, jobject, jint target_fd, jint draft_fd, jint context_size, jint draft_max) {
    if (target_fd < 0 || draft_fd < 0) return JNI_FALSE;
    free_engine();
    common_init();
    llama_backend_init();

    try {
        const std::string target_path = fd_path(target_fd);
        const std::string draft_path  = fd_path(draft_fd);

        common_params params;
        params.model.path = target_path;
        params.n_ctx = std::max(512, static_cast<int>(context_size));
        params.n_batch = std::min(params.n_ctx, 512);
        params.n_ubatch = std::min(params.n_batch, 512);
        params.n_parallel = 1;
        params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK };
        params.speculative.draft.mparams.path = draft_path;
        params.speculative.draft.n_max = std::clamp(static_cast<int>(draft_max), 1, 16);
        params.speculative.draft.p_min = 0.0f;

        auto target_init = common_init_from_params(params);
        if (!target_init || !target_init->model() || !target_init->context()) {
            fail("stage=target_init");
            close(target_fd);
            close(draft_fd);
            return JNI_FALSE;
        }

        common_params params_dft = common_base_params_to_speculative(params);
        params_dft.model.path = draft_path;
        params_dft.speculative.draft.mparams.path = draft_path;

        auto draft_init = common_speculative_init_from_params(
            params_dft, target_init->model(), target_init->context());
        if (!draft_init || !draft_init->model() || !draft_init->context()) {
            fail("stage=draft_init; target loaded but dSpark draft initialization failed");
            close(target_fd);
            close(draft_fd);
            return JNI_FALSE;
        }

        params.speculative.draft.ctx_tgt = target_init->context();
        params.speculative.draft.ctx_dft = draft_init->context();

        auto spec = common_speculative_init(params.speculative, 1);
        if (!spec) {
            fail("stage=speculative_init; target and dSpark contexts loaded but DSpark runtime initialization failed");
            close(target_fd);
            close(draft_fd);
            return JNI_FALSE;
        }

        common_params_sampling sampling;
        sampling.temp = 0.7f;
        sampling.top_k = 40;
        sampling.top_p = 0.95f;
        auto sampler = common_sampler_init(target_init->model(), sampling);
        if (!sampler) {
            fail("stage=sampler_init");
            close(target_fd);
            close(draft_fd);
            return JNI_FALSE;
        }

        close(target_fd);
        close(draft_fd);

        g_engine.target_init = std::move(target_init);
        g_engine.draft_init = std::move(draft_init);
        g_engine.spec.reset(spec);
        g_engine.sampler.reset(sampler);
        g_engine.target_model = g_engine.target_init->model();
        g_engine.target_context = g_engine.target_init->context();
        g_engine.draft_context = g_engine.draft_init->context();
        g_engine.vocab = llama_model_get_vocab(g_engine.target_model);
        g_engine.context_size = params.n_ctx;
        g_engine.draft_max = params.speculative.draft.n_max;
        g_engine.last_error.clear();
        return JNI_TRUE;
    } catch (const std::exception & e) {
        close(target_fd);
        close(draft_fd);
        free_engine();
        fail(std::string("stage=exception; ") + e.what());
        return JNI_FALSE;
    } catch (...) {
        close(target_fd);
        close(draft_fd);
        free_engine();
        fail("stage=exception; unknown native exception");
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeGenerate(JNIEnv * env, jobject, jstring prompt, jint max_tokens) {
    if (!g_engine.target_model || !g_engine.target_context || !g_engine.draft_context || !g_engine.spec || !g_engine.sampler)
        return env->NewStringUTF("[models not loaded]");

    const std::string prompt_text = get_prompt(env, prompt);
    if (prompt_text.empty()) return env->NewStringUTF("");

    llama_memory_clear(llama_get_memory(g_engine.target_context), false);
    llama_memory_clear(llama_get_memory(g_engine.draft_context), false);

    common_params_sampling sampling;
    sampling.temp = 0.7f;
    sampling.top_k = 40;
    sampling.top_p = 0.95f;
    g_engine.sampler.reset(common_sampler_init(g_engine.target_model, sampling));
    if (!g_engine.sampler) return env->NewStringUTF("[sampler init failed]");

    const llama_tokens input = common_tokenize(g_engine.target_context, prompt_text, true, true);
    if (input.empty()) return env->NewStringUTF("[tokenization failed]");
    const uint32_t n_ctx = llama_n_ctx(g_engine.target_context);
    if (input.size() + 1 >= n_ctx) return env->NewStringUTF("[prompt exceeds context]");

    const llama_seq_id seq_id = 0;
    const int n_predict = std::max(1, static_cast<int>(max_tokens));
    llama_tokens prompt_tgt(input.begin(), input.end() - 1);
    prompt_tgt.reserve(n_ctx);

    llama_batch prompt_batch = llama_batch_init(
        std::min<uint32_t>(llama_n_batch(g_engine.target_context), input.size()), 0, 1);
    for (size_t i = 0; i + 1 < input.size(); ++i)
        common_batch_add(prompt_batch, input[i], static_cast<llama_pos>(i), {seq_id}, false);

    if (prompt_batch.n_tokens > 0 && llama_decode(g_engine.target_context, prompt_batch) != 0) {
        llama_batch_free(prompt_batch);
        return env->NewStringUTF("[prompt decode failed]");
    }
    if (prompt_batch.n_tokens > 0 && !common_speculative_process(g_engine.spec.get(), prompt_batch)) {
        llama_batch_free(prompt_batch);
        return env->NewStringUTF("[dSpark prompt processing failed]");
    }
    llama_batch_free(prompt_batch);

    common_speculative_begin(g_engine.spec.get(), seq_id, prompt_tgt);
    llama_token id_last = input.back();
    int n_past = static_cast<int>(input.size()) - 1;
    int generated = 0;
    std::string output;
    llama_tokens draft;
    llama_batch batch_tgt = llama_batch_init(llama_n_batch(g_engine.target_context), 0, 1);

    while (generated < n_predict) {
        if (draft.empty()) {
            const int remaining = std::max(0, static_cast<int>(n_ctx) - n_past - 2);
            const int n_draft_max = std::min(g_engine.draft_max, remaining);
            if (n_draft_max <= 0) break;
            common_speculative_get_draft_params(g_engine.spec.get(), seq_id) = {
                true, n_draft_max, n_past, id_last, &prompt_tgt, &draft
            };
            common_speculative_draft(g_engine.spec.get());
            if (draft.empty()) break;
        }

        common_batch_clear(batch_tgt);
        common_batch_add(batch_tgt, id_last, n_past, {seq_id}, true);
        for (size_t i = 0; i < draft.size(); ++i)
            common_batch_add(batch_tgt, draft[i], n_past + static_cast<llama_pos>(i) + 1, {seq_id}, true);

        if (llama_decode(g_engine.target_context, batch_tgt) != 0) break;
        if (!common_speculative_process(g_engine.spec.get(), batch_tgt)) break;

        auto ids = common_sampler_sample_and_accept_n(g_engine.sampler.get(), g_engine.target_context, draft);
        if (ids.empty()) break;
        common_speculative_accept(g_engine.spec.get(), seq_id, static_cast<uint16_t>(ids.size() - 1));

        n_past += static_cast<int>(ids.size() - 1);
        for (size_t i = 0; i < ids.size() && generated < n_predict; ++i) {
            prompt_tgt.push_back(id_last);
            id_last = ids[i];
            if (llama_vocab_is_eog(g_engine.vocab, id_last)) break;
            output += common_token_to_piece(g_engine.target_context, id_last);
            ++generated;
        }

        draft.clear();
        if (llama_vocab_is_eog(g_engine.vocab, id_last)) break;
    }

    llama_batch_free(batch_tgt);
    return env->NewStringUTF(output.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeGetLastError(JNIEnv * env, jobject) {
    return env->NewStringUTF(g_engine.last_error.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeUnloadModel(JNIEnv *, jobject) {
    free_engine();
    llama_backend_free();
}
