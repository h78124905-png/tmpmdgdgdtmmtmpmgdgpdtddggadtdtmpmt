#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <string>
#include <thread>
#include <vector>

#include "llama.h"
#include "common.h"
#include "sampling.h"
#include "chat.h"
#include "speculative.h"
#include "ggml-backend.h"

#define LOG_TAG "LfmMobile"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace {
struct Engine {
    common_sampler_ptr sampler;
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    const llama_vocab * vocab = nullptr;
    common_speculative_init_result_ptr draft_init;
    common_speculative_ptr speculative;
    std::string last_error;
    std::string gpu_name = "CPU";
    bool backend_initialized = false;
};
Engine g_engine;

void free_engine() {
    g_engine.speculative.reset();
    g_engine.draft_init.reset();
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
    g_engine.gpu_name = "CPU";
}

void set_error(const std::string & s) {
    g_engine.last_error = s;
    LOGE("%s", s.c_str());
}

std::string utf16_to_utf8(const jchar * chars, jsize length) {
    std::string result;
    result.reserve(static_cast<size_t>(length) * 3);
    for (jsize i = 0; i < length; ++i) {
        uint32_t cp = chars[i];
        if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < length) {
            const uint32_t low = chars[i + 1];
            if (low >= 0xDC00 && low <= 0xDFFF) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                ++i;
            }
        }
        if (cp <= 0x7F) {
            result.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FF) {
            result.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp <= 0xFFFF) {
            result.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            result.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            result.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            result.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    return result;
}

std::string get_string(JNIEnv * env, jstring value) {
    if (!value) return {};
    const jsize length = env->GetStringLength(value);
    const jchar * chars = env->GetStringChars(value, nullptr);
    if (!chars) return {};
    const std::string result = utf16_to_utf8(chars, length);
    env->ReleaseStringChars(value, chars);
    return result;
}

jstring utf8_to_jstring(JNIEnv * env, const std::string & value) {
    std::vector<jchar> utf16;
    utf16.reserve(value.size());
    for (size_t i = 0; i < value.size();) {
        const unsigned char c = static_cast<unsigned char>(value[i]);
        uint32_t cp = 0;
        size_t bytes = 0;
        if (c <= 0x7F) {
            cp = c; bytes = 1;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < value.size()) {
            cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(value[i + 1]) & 0x3F); bytes = 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < value.size()) {
            cp = ((c & 0x0F) << 12) |
                 ((static_cast<unsigned char>(value[i + 1]) & 0x3F) << 6) |
                 (static_cast<unsigned char>(value[i + 2]) & 0x3F); bytes = 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < value.size()) {
            cp = ((c & 0x07) << 18) |
                 ((static_cast<unsigned char>(value[i + 1]) & 0x3F) << 12) |
                 ((static_cast<unsigned char>(value[i + 2]) & 0x3F) << 6) |
                 (static_cast<unsigned char>(value[i + 3]) & 0x3F); bytes = 4;
        } else {
            cp = 0xFFFD; bytes = 1;
        }
        i += bytes;
        if (cp <= 0xFFFF) {
            utf16.push_back(static_cast<jchar>(cp));
        } else if (cp <= 0x10FFFF) {
            cp -= 0x10000;
            utf16.push_back(static_cast<jchar>(0xD800 | (cp >> 10)));
            utf16.push_back(static_cast<jchar>(0xDC00 | (cp & 0x3FF)));
        } else {
            utf16.push_back(static_cast<jchar>(0xFFFD));
        }
    }
    return env->NewString(utf16.empty() ? nullptr : utf16.data(), static_cast<jsize>(utf16.size()));
}

// Token pieces are byte-oriented and may split a UTF-8 code point across
// tokens. Only send complete UTF-8 sequences over JNI.
size_t complete_utf8_prefix(const std::string & s) {
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        size_t need = 1;
        if (c <= 0x7F) need = 1;
        else if ((c & 0xE0) == 0xC0) need = 2;
        else if ((c & 0xF0) == 0xE0) need = 3;
        else if ((c & 0xF8) == 0xF0) need = 4;
        else { ++i; continue; }
        if (i + need > s.size()) break;
        bool ok = true;
        for (size_t j = 1; j < need; ++j) {
            const unsigned char x = static_cast<unsigned char>(s[i + j]);
            if ((x & 0xC0) != 0x80) { ok = false; break; }
        }
        if (!ok) { ++i; continue; }
        i += need;
    }
    return i;
}

void emit_utf8(JNIEnv * env, jobject callback, jmethodID on_token, std::string & pending, const std::string & piece) {
    pending += piece;
    const size_t ready = complete_utf8_prefix(pending);
    if (ready == 0) return;
    const std::string chunk = pending.substr(0, ready);
    pending.erase(0, ready);
    if (chunk.empty()) return;
    jstring jchunk = utf8_to_jstring(env, chunk);
    env->CallVoidMethod(callback, on_token, jchunk);
    env->DeleteLocalRef(jchunk);
    if (env->ExceptionCheck()) env->ExceptionClear();
}

void emit_final_utf8(JNIEnv * env, jobject callback, jmethodID on_token, std::string & pending) {
    if (pending.empty()) return;
    jstring jchunk = utf8_to_jstring(env, pending);
    env->CallVoidMethod(callback, on_token, jchunk);
    env->DeleteLocalRef(jchunk);
    pending.clear();
    if (env->ExceptionCheck()) env->ExceptionClear();
}

void emit_stats(JNIEnv * env, jobject callback, jmethodID on_stats, double tok_per_sec, int64_t elapsed_ms, int context_used, int context_size) {
    if (!callback || !on_stats) return;
    jstring jgpu = utf8_to_jstring(env, g_engine.gpu_name);
    env->CallVoidMethod(callback, on_stats, tok_per_sec, static_cast<jlong>(elapsed_ms), jgpu,
                        static_cast<jint>(context_used), static_cast<jint>(context_size));
    env->DeleteLocalRef(jgpu);
    if (env->ExceptionCheck()) env->ExceptionClear();
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

void detect_gpu_backend() {
    g_engine.gpu_name = "CPU";
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(dev);
        const char * desc = ggml_backend_dev_description(dev);
        if (name && (std::string(name).find("Vulkan") != std::string::npos ||
                     std::string(name).find("vulkan") != std::string::npos)) {
            g_engine.gpu_name = desc && *desc ? std::string("Vulkan · ") + desc : std::string("Vulkan · ") + name;
            LOGI("[backend] Vulkan GPU backend available: %s", g_engine.gpu_name.c_str());
            return;
        }
    }
    LOGI("[backend] Vulkan device not registered; CPU fallback");
}

std::vector<common_chat_msg> build_messages(const std::string & prompt_text) {
    std::vector<common_chat_msg> messages;
    messages.push_back({"system", "You are a helpful local assistant. Answer naturally and accurately."});
    size_t pos = 0;
    int current = -1;
    while (pos < prompt_text.size()) {
        const size_t end = prompt_text.find('\n', pos);
        const std::string line = prompt_text.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        if (line.rfind("User: ", 0) == 0) {
            messages.push_back({"user", line.substr(6)});
            current = static_cast<int>(messages.size()) - 1;
        } else if (line.rfind("Assistant: ", 0) == 0) {
            messages.push_back({"assistant", line.substr(11)});
            current = static_cast<int>(messages.size()) - 1;
        } else if (current >= 0) {
            messages[current].content += "\n" + line;
        }
        if (end == std::string::npos) break;
        pos = end + 1;
    }
    return messages;
}

bool decode_batch(llama_context * context, llama_batch & batch) {
    const int rc = llama_decode(context, batch);
    if (rc != 0) LOGE("llama_decode failed rc=%d", rc);
    return rc == 0;
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

    jmethodID on_token = nullptr;
    jmethodID on_stats = nullptr;
    if (callback) {
        jclass callback_class = env->GetObjectClass(callback);
        on_token = env->GetMethodID(callback_class, "onToken", "(Ljava/lang/String;)V");
        on_stats = env->GetMethodID(callback_class, "onStats", "(DJLjava/lang/String;II)V");
        env->DeleteLocalRef(callback_class);
        if (!on_token) {
            if (env->ExceptionCheck()) env->ExceptionClear();
            return "[stream callback method not found]";
        }
        if (!on_stats && env->ExceptionCheck()) env->ExceptionClear();
    }

    // Prefill. DSpark consumes the target hidden-state features produced here.
    const uint32_t n_batch = std::max<uint32_t>(1, llama_n_batch(g_engine.context));
    llama_batch batch = llama_batch_init(std::min<uint32_t>(n_batch, input.size()), 0, 1);
    for (size_t i = 0; i < input.size(); ++i) {
        common_batch_add(batch, input[i], static_cast<llama_pos>(i), {0}, true);
        if (batch.n_tokens == static_cast<int>(n_batch) || i + 1 == input.size()) {
            if (!decode_batch(g_engine.context, batch)) { llama_batch_free(batch); return "[prompt decode failed]"; }
            if (g_engine.speculative && !common_speculative_process(g_engine.speculative.get(), batch)) {
                llama_batch_free(batch);
                return "[speculative prefill failed]";
            }
            common_batch_clear(batch);
        }
    }
    llama_batch_free(batch);

    std::vector<llama_token> history = input;
    llama_pos n_past = static_cast<llama_pos>(input.size());
    llama_token sampled = common_sampler_sample(g_engine.sampler.get(), g_engine.context, static_cast<int>(input.size()) - 1);
    common_sampler_accept(g_engine.sampler.get(), sampled, true);

    std::string output;
    std::string utf8_pending;
    const auto t_start = std::chrono::steady_clock::now();
    int generated = 0;

    auto emit_generated = [&](llama_token id) -> bool {
        if (llama_vocab_is_eog(g_engine.vocab, id)) return false;
        const std::string piece = common_token_to_piece(g_engine.context, id);
        output += piece;
        if (callback && on_token && !piece.empty()) emit_utf8(env, callback, on_token, utf8_pending, piece);
        ++generated;
        return true;
    };

    // The first sampled token is the anchor used by DSpark, but it is also a
    // real output token and must not disappear from the UI.
    if (!emit_generated(sampled)) {
        if (callback && on_token) emit_final_utf8(env, callback, on_token, utf8_pending);
        return output;
    }

    if (g_engine.speculative) common_speculative_begin(g_engine.speculative.get(), 0, history);

    const int n_predict = std::max(1, max_tokens);
    int step = 1;
    while (step < n_predict) {
        if (g_engine.speculative) {
            std::vector<uint8_t> ckpt_tgt;
            std::vector<uint8_t> ckpt_dft;
            const bool use_ckpt = common_context_can_seq_rm(g_engine.context) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
            if (use_ckpt) {
                const size_t st = llama_state_seq_get_size_ext(g_engine.context, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                ckpt_tgt.resize(st);
                if (st) llama_state_seq_get_data_ext(g_engine.context, ckpt_tgt.data(), st, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                if (g_engine.draft_init && g_engine.draft_init->context()) {
                    auto * dctx = g_engine.draft_init->context();
                    const size_t sd = llama_state_seq_get_size_ext(dctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    ckpt_dft.resize(sd);
                    if (sd) llama_state_seq_get_data_ext(dctx, ckpt_dft.data(), sd, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }

            llama_tokens drafts;
            auto & dp = common_speculative_get_draft_params(g_engine.speculative.get(), 0);
            dp.drafting = true;
            dp.n_max = std::min(10, n_predict - step);
            dp.n_past = n_past;
            dp.id_last = sampled;
            dp.prompt = &history;
            dp.result = &drafts;
            common_speculative_draft(g_engine.speculative.get());

            if (drafts.empty()) {
                // A failed draft is not fatal; continue with target-only decoding.
                g_engine.speculative.reset();
                g_engine.draft_init.reset();
                continue;
            }

            const int n_verify = 1 + static_cast<int>(drafts.size());
            llama_batch verify = llama_batch_init(n_verify, 0, 1);
            common_batch_add(verify, sampled, n_past, {0}, true);
            for (size_t i = 0; i < drafts.size(); ++i) {
                common_batch_add(verify, drafts[i], n_past + 1 + static_cast<llama_pos>(i), {0}, true);
            }

            if (g_engine.draft_init && g_engine.draft_init->context()) {
                llama_memory_seq_rm(llama_get_memory(g_engine.draft_init->context()), 0, n_past, -1);
            }
            if (!decode_batch(g_engine.context, verify)) {
                llama_batch_free(verify);
                return "[speculative target decode failed]";
            }
            if (!common_speculative_process(g_engine.speculative.get(), verify)) {
                llama_batch_free(verify);
                return "[speculative process failed]";
            }

            std::vector<llama_token> ids = common_sampler_sample_and_accept_n(g_engine.sampler.get(), g_engine.context, drafts);
            llama_batch_free(verify);
            if (ids.empty()) return "[speculative sampling failed]";

            const size_t accepted = ids.size() - 1;
            common_speculative_accept(g_engine.speculative.get(), 0, static_cast<uint16_t>(accepted));
            const llama_pos next_n_past = n_past + static_cast<llama_pos>(accepted);
            const llama_pos remove_from = next_n_past + 1;

            if (use_ckpt && (!ckpt_tgt.empty() || !ckpt_dft.empty())) {
                if (!ckpt_tgt.empty()) llama_state_seq_set_data_ext(g_engine.context, ckpt_tgt.data(), ckpt_tgt.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                auto * dctx = g_engine.draft_init ? g_engine.draft_init->context() : nullptr;
                if (dctx && !ckpt_dft.empty()) llama_state_seq_set_data_ext(dctx, ckpt_dft.data(), ckpt_dft.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                if (accepted > 0) {
                    llama_batch redo = llama_batch_init(static_cast<int>(accepted), 0, 1);
                    for (size_t i = 0; i < accepted; ++i) {
                        common_batch_add(redo, ids[i], n_past + static_cast<llama_pos>(i), {0}, i + 1 == accepted);
                    }
                    if (!decode_batch(g_engine.context, redo)) { llama_batch_free(redo); return "[speculative rollback decode failed]"; }
                    llama_batch_free(redo);
                }
            } else {
                llama_memory_seq_rm(llama_get_memory(g_engine.context), 0, remove_from, -1);
                if (g_engine.draft_init && g_engine.draft_init->context()) {
                    llama_memory_seq_rm(llama_get_memory(g_engine.draft_init->context()), 0, remove_from, -1);
                }
            }

            for (const llama_token id : ids) {
                if (step >= n_predict) break;
                if (!emit_generated(id)) { step = n_predict; break; }
                ++step;
            }
            history.insert(history.end(), ids.begin(), ids.end() - 1);
            sampled = ids.back();
            n_past = next_n_past;
        } else {
            llama_batch next_batch = llama_batch_init(1, 0, 1);
            common_batch_add(next_batch, sampled, n_past, {0}, true);
            if (!decode_batch(g_engine.context, next_batch)) { llama_batch_free(next_batch); return "[token decode failed]"; }
            const llama_token next = common_sampler_sample(g_engine.sampler.get(), g_engine.context, 0);
            common_sampler_accept(g_engine.sampler.get(), next, true);
            llama_batch_free(next_batch);
            if (!emit_generated(next)) break;
            sampled = next;
            ++n_past;
            ++step;
        }

        if (callback && on_stats) {
            const auto now = std::chrono::steady_clock::now();
            const int64_t elapsed_ms = std::max<int64_t>(1, std::chrono::duration_cast<std::chrono::milliseconds>(now - t_start).count());
            emit_stats(env, callback, on_stats, static_cast<double>(generated) * 1000.0 / static_cast<double>(elapsed_ms), elapsed_ms,
                       static_cast<int>(n_past), static_cast<int>(n_ctx));
        }
    }

    if (callback && on_token) emit_final_utf8(env, callback, on_token, utf8_pending);
    return output;
}
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeLoadModelFromPath(
        JNIEnv * env, jobject, jstring model_path, jstring draft_model_path, jint context_size) {
    LOGI("[load] JNI entered");
    const std::string path = get_string(env, model_path);
    const std::string draft_path = get_string(env, draft_model_path);
    if (path.empty()) {
        set_error("stage=path; model path is empty");
        return JNI_FALSE;
    }

    free_engine();
    g_engine.last_error.clear();
    if (!g_engine.backend_initialized) {
        llama_backend_init();
        g_engine.backend_initialized = true;
        detect_gpu_backend();
    }

    try {
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = -1;
        model_params.progress_callback = load_progress;
        model_params.progress_callback_user_data = nullptr;

        LOGI("[load] model load starting (Vulkan GPU offload requested)");
        llama_model * model = llama_model_load_from_file(path.c_str(), model_params);
        if (!model) {
            set_error("stage=model_load; llama_model_load_from_file returned null");
            return JNI_FALSE;
        }
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
        llama_context * context = llama_init_from_model(model, context_params);
        if (!context) {
            llama_model_free(model);
            set_error("stage=context_init; GGUF loaded but llama_init_from_model returned null");
            return JNI_FALSE;
        }

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

        if (!draft_path.empty()) {
            LOGI("[spec] initializing LFM2.5 DSpark draft: %s", draft_path.c_str());
            common_params spec_params;
            spec_params.model.path = draft_path;
            spec_params.n_ctx = context_params.n_ctx;
            spec_params.n_batch = context_params.n_batch;
            spec_params.n_ubatch = context_params.n_ubatch;
            spec_params.n_parallel = 1;
            spec_params.n_sequences = 1;
            spec_params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK };
            spec_params.speculative.draft.mparams.path = draft_path;
            spec_params.speculative.draft.n_max = 10;
            spec_params.speculative.draft.n_min = 0;
            spec_params.speculative.draft.n_gpu_layers = -1;
            spec_params.speculative.draft.ctx_tgt = context;

            g_engine.draft_init = common_speculative_init_from_params(spec_params, model, context);
            if (!g_engine.draft_init || !g_engine.draft_init->model() || !g_engine.draft_init->context()) {
                free_engine();
                set_error("stage=dspark_init; could not initialize the selected DSpark draft. Check that it matches the target model.");
                return JNI_FALSE;
            }
            spec_params.speculative.draft.ctx_dft = g_engine.draft_init->context();
            g_engine.speculative.reset(common_speculative_init(spec_params.speculative, 1));
            if (!g_engine.speculative) {
                free_engine();
                set_error("stage=dspark_spec; common_speculative_init returned null");
                return JNI_FALSE;
            }
            LOGI("[spec] DSpark initialized successfully");
        }

        LOGI("[load] model load completed successfully; backend=%s; dspark=%s",
             g_engine.gpu_name.c_str(), g_engine.speculative ? "on" : "off");
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
    return utf8_to_jstring(env, g_engine.last_error);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeGenerate(JNIEnv * env, jobject, jstring prompt, jint max_tokens) {
    return utf8_to_jstring(env, generate_impl(env, get_string(env, prompt), max_tokens, nullptr));
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeGenerateStream(JNIEnv * env, jobject, jstring prompt, jint max_tokens, jobject callback) {
    try {
        const std::string result = generate_impl(env, get_string(env, prompt), max_tokens, callback);
        return utf8_to_jstring(env, result);
    } catch (const std::exception & e) {
        return utf8_to_jstring(env, std::string("[stream exception] ") + e.what());
    } catch (...) {
        return utf8_to_jstring(env, "[stream exception]");
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeUnloadModel(JNIEnv *, jobject) {
    free_engine();
}
