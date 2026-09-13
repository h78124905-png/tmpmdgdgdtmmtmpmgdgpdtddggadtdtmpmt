#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <string>
#include <unistd.h>
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
    llama_model *model = nullptr;
    llama_context *context = nullptr;
    const llama_vocab *vocab = nullptr;
    common_sampler_ptr sampler;
    std::string last_error;
    std::mutex mutex;
    bool backend_initialized = false;
};
Engine g_engine;

void set_error(const std::string &s) { g_engine.last_error = s; LOGE("%s", s.c_str()); }

void unload_locked() {
    g_engine.sampler.reset();
    if (g_engine.context) llama_free(g_engine.context);
    if (g_engine.model) llama_model_free(g_engine.model);
    g_engine.context = nullptr;
    g_engine.model = nullptr;
    g_engine.vocab = nullptr;
}

std::string jstring_to_utf8(JNIEnv *env, jstring value) {
    if (!value) return {};
    const jsize n = env->GetStringLength(value);
    const jchar *p = env->GetStringChars(value, nullptr);
    if (!p) return {};
    std::string out;
    out.reserve(static_cast<size_t>(n) * 2);
    for (jsize i = 0; i < n; ++i) {
        uint32_t cp = p[i];
        if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < n && p[i + 1] >= 0xDC00 && p[i + 1] <= 0xDFFF)
            cp = 0x10000 + ((cp - 0xD800) << 10) + (p[++i] - 0xDC00);
        if (cp <= 0x7F) out.push_back((char)cp);
        else if (cp <= 0x7FF) { out.push_back((char)(0xC0 | (cp >> 6))); out.push_back((char)(0x80 | (cp & 63))); }
        else if (cp <= 0xFFFF) { out.push_back((char)(0xE0 | (cp >> 12))); out.push_back((char)(0x80 | ((cp >> 6) & 63))); out.push_back((char)(0x80 | (cp & 63))); }
        else { out.push_back((char)(0xF0 | (cp >> 18))); out.push_back((char)(0x80 | ((cp >> 12) & 63))); out.push_back((char)(0x80 | ((cp >> 6) & 63))); out.push_back((char)(0x80 | (cp & 63))); }
    }
    env->ReleaseStringChars(value, p);
    return out;
}

jstring utf8_to_jstring(JNIEnv *env, const std::string &value) {
    std::vector<jchar> out;
    out.reserve(value.size());
    for (size_t i = 0; i < value.size();) {
        const unsigned char c = (unsigned char)value[i];
        uint32_t cp = 0; size_t n = 1;
        if (c <= 0x7F) cp = c;
        else if ((c & 0xE0) == 0xC0 && i + 1 < value.size()) { cp = ((c & 31) << 6) | ((unsigned char)value[i + 1] & 63); n = 2; }
        else if ((c & 0xF0) == 0xE0 && i + 2 < value.size()) { cp = ((c & 15) << 12) | (((unsigned char)value[i + 1] & 63) << 6) | ((unsigned char)value[i + 2] & 63); n = 3; }
        else if ((c & 0xF8) == 0xF0 && i + 3 < value.size()) { cp = ((c & 7) << 18) | (((unsigned char)value[i + 1] & 63) << 12) | (((unsigned char)value[i + 2] & 63) << 6) | ((unsigned char)value[i + 3] & 63); n = 4; }
        else cp = 0xFFFD;
        i += n;
        if (cp <= 0xFFFF) out.push_back((jchar)cp);
        else if (cp <= 0x10FFFF) { cp -= 0x10000; out.push_back((jchar)(0xD800 | (cp >> 10))); out.push_back((jchar)(0xDC00 | (cp & 0x3FF))); }
        else out.push_back((jchar)0xFFFD);
    }
    return env->NewString(out.data(), (jsize)out.size());
}

std::vector<common_chat_msg> parse_prompt(const std::string &prompt) {
    std::vector<common_chat_msg> messages;
    messages.push_back({"system", "You are a helpful local assistant. Answer naturally and accurately."});
    size_t pos = 0;
    while (pos < prompt.size()) {
        const size_t end = prompt.find('\n', pos);
        const std::string line = prompt.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        if (line.rfind("User: ", 0) == 0) messages.push_back({"user", line.substr(6)});
        else if (line.rfind("Assistant: ", 0) == 0) messages.push_back({"assistant", line.substr(11)});
        else if (!messages.empty()) messages.back().content += "\n" + line;
        if (end == std::string::npos) break;
        pos = end + 1;
    }
    return messages;
}

size_t complete_utf8_prefix(const std::string &s) {
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = (unsigned char)s[i];
        const size_t need = c <= 0x7F ? 1 : ((c & 0xE0) == 0xC0 ? 2 : ((c & 0xF0) == 0xE0 ? 3 : 4));
        if (i + need > s.size()) break;
        bool ok = true;
        for (size_t j = 1; j < need; ++j) if (((unsigned char)s[i + j] & 0xC0) != 0x80) { ok = false; break; }
        if (!ok) { ++i; continue; }
        i += need;
    }
    return i;
}

void emit_token(JNIEnv *env, jobject callback, jmethodID method, std::string &pending, const std::string &piece) {
    if (!callback || !method || piece.empty()) return;
    pending += piece;
    const size_t ready = complete_utf8_prefix(pending);
    if (!ready) return;
    const std::string chunk = pending.substr(0, ready);
    pending.erase(0, ready);
    jstring s = utf8_to_jstring(env, chunk);
    env->CallVoidMethod(callback, method, s);
    env->DeleteLocalRef(s);
    if (env->ExceptionCheck()) env->ExceptionClear();
}

void flush_tokens(JNIEnv *env, jobject callback, jmethodID method, std::string &pending) {
    if (!callback || !method || pending.empty()) return;
    jstring s = utf8_to_jstring(env, pending);
    env->CallVoidMethod(callback, method, s);
    env->DeleteLocalRef(s);
    pending.clear();
    if (env->ExceptionCheck()) env->ExceptionClear();
}

std::string generate_chat_impl(JNIEnv *env, const common_chat_params &, const llama_tokens &input, int max_tokens,
                               jobject callback, const std::function<void(const char *)> &progress_cb) {
    std::lock_guard<std::mutex> lock(g_engine.mutex);
    if (!g_engine.model || !g_engine.context || !g_engine.vocab) return "[model not loaded]";
    if (input.empty()) return "[empty prompt]";
    if (input.size() + (size_t)max_tokens >= (size_t)llama_n_ctx(g_engine.context)) return "[prompt exceeds context]";

    if (progress_cb) progress_cb("sampler_init");
    common_params_sampling sampling;
    sampling.temp = 0.7f; sampling.top_k = 40; sampling.top_p = 0.95f;
    g_engine.sampler.reset(common_sampler_init(g_engine.model, sampling));
    if (!g_engine.sampler) return "[sampler init failed]";

    jmethodID on_token = nullptr, on_stats = nullptr;
    if (callback) {
        jclass cls = env->GetObjectClass(callback);
        on_token = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;)V");
        on_stats = env->GetMethodID(cls, "onStats", "(DJII)V");
        if (env->ExceptionCheck()) env->ExceptionClear();
        env->DeleteLocalRef(cls);
        if (!on_token) return "[stream callback method not found]";
    }

    if (progress_cb) progress_cb("kv_clear");
    llama_memory_clear(llama_get_memory(g_engine.context), false);
    if (progress_cb) progress_cb("prefill");

    const auto prefill_start = std::chrono::steady_clock::now();
    const uint32_t n_batch = std::max<uint32_t>(1, llama_n_batch(g_engine.context));
    size_t prefill_tokens = 0;
    llama_batch batch = llama_batch_init((int)std::min<uint32_t>(n_batch, (uint32_t)input.size()), 0, 1);
    for (size_t i = 0; i < input.size(); ++i) {
        common_batch_add(batch, input[i], (llama_pos)i, {0}, i + 1 == input.size());
        if (batch.n_tokens == (int)n_batch || i + 1 == input.size()) {
            if (llama_decode(g_engine.context, batch) != 0) { llama_batch_free(batch); return "[prompt decode failed]"; }
            prefill_tokens += batch.n_tokens;
            common_batch_clear(batch);
            if (progress_cb) {
                const auto elapsed = std::max<int64_t>(1, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - prefill_start).count());
                progress_cb(("prefill " + std::to_string(prefill_tokens) + "/" + std::to_string(input.size()) + " tokens " + std::to_string(elapsed) + "ms").c_str());
            }
        }
    }
    llama_batch_free(batch);
    const auto prefill_ms = std::max<int64_t>(1, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - prefill_start).count());
    LOGI("CPU prefill: %zu tokens, %lld ms, %.2f tok/s", input.size(), (long long)prefill_ms, input.size() * 1000.0 / prefill_ms);
    if (progress_cb) progress_cb("prefill_complete");

    llama_token next = common_sampler_sample(g_engine.sampler.get(), g_engine.context, (int)input.size() - 1);
    common_sampler_accept(g_engine.sampler.get(), next, true);
    if (progress_cb) progress_cb("generation");

    std::string output, pending;
    const auto generation_start = std::chrono::steady_clock::now();
    int generated = 0;
    while (generated < std::max(1, max_tokens)) {
        if (llama_vocab_is_eog(g_engine.vocab, next)) break;
        const std::string piece = common_token_to_piece(g_engine.context, next);
        output += piece;
        emit_token(env, callback, on_token, pending, piece);
        ++generated;
        if (callback && on_stats) {
            const auto elapsed = std::max<int64_t>(1, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - generation_start).count());
            env->CallVoidMethod(callback, on_stats, (jdouble)(generated * 1000.0 / elapsed), (jlong)elapsed,
                                (jint)(input.size() + generated), (jint)llama_n_ctx(g_engine.context));
            if (env->ExceptionCheck()) env->ExceptionClear();
        }
        if (generated >= max_tokens) break;
        llama_batch one = llama_batch_init(1, 0, 1);
        common_batch_add(one, next, (llama_pos)(input.size() + generated - 1), {0}, true);
        if (llama_decode(g_engine.context, one) != 0) { llama_batch_free(one); return "[token decode failed]"; }
        next = common_sampler_sample(g_engine.sampler.get(), g_engine.context, 0);
        common_sampler_accept(g_engine.sampler.get(), next, true);
        llama_batch_free(one);
    }
    flush_tokens(env, callback, on_token, pending);
    return output;
}

bool build_chat(const std::string &prompt, common_chat_params &chat, llama_tokens &tokens) {
    auto messages = parse_prompt(prompt);
    if (messages.size() < 2) return false;
    auto templates = common_chat_templates_init(g_engine.model, "");
    if (!templates) return false;
    common_chat_templates_inputs inputs;
    inputs.messages = messages;
    inputs.add_generation_prompt = true;
    inputs.enable_thinking = true;
    chat = common_chat_templates_apply(templates.get(), inputs);
    if (chat.prompt.empty()) return false;
    tokens = common_tokenize(g_engine.vocab, chat.prompt, true);
    return !tokens.empty();
}
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeLoadModelFromPath(JNIEnv *env, jobject, jstring model_path, jstring, jint context_size) {
    try {
        const std::string path = jstring_to_utf8(env, model_path);
        if (path.empty()) { set_error("model path is empty"); return JNI_FALSE; }
        std::lock_guard<std::mutex> lock(g_engine.mutex);
        unload_locked();
        if (!g_engine.backend_initialized) { llama_backend_init(); g_engine.backend_initialized = true; }
        llama_model_params mp = llama_model_default_params();
#if LFM_VULKAN_AVAILABLE
        mp.n_gpu_layers = 999;
#else
        mp.n_gpu_layers = 0;
#endif
        llama_model *model = llama_model_load_from_file(path.c_str(), mp);
        if (!model) { set_error("llama_model_load_from_file failed"); return JNI_FALSE; }
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = std::max(512, (int)context_size);
        cp.n_batch = std::min<uint32_t>(512, cp.n_ctx);
        cp.n_ubatch = cp.n_batch;
        const long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
        const int threads = (int)std::max(1L, std::min(8L, cpu_count > 1 ? cpu_count - 1 : 1));
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        llama_context *ctx = llama_init_from_model(model, cp);
        if (!ctx) { llama_model_free(model); set_error("llama_init_from_model failed"); return JNI_FALSE; }
        g_engine.model = model; g_engine.context = ctx; g_engine.vocab = llama_model_get_vocab(model); g_engine.last_error.clear();
        LOGI("engine ready: backend=%s threads=%d context=%d", LFM_VULKAN_AVAILABLE ? "Vulkan/CPU fallback" : "CPU", threads, cp.n_ctx);
        return JNI_TRUE;
    } catch (const std::exception &e) { set_error(std::string("load exception: ") + e.what()); return JNI_FALSE; }
    catch (...) { set_error("load exception: unknown"); return JNI_FALSE; }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeGetLastError(JNIEnv *env, jobject) {
    std::lock_guard<std::mutex> lock(g_engine.mutex);
    return utf8_to_jstring(env, g_engine.last_error);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeGetBackendInfo(JNIEnv *env, jobject) {
    std::lock_guard<std::mutex> lock(g_engine.mutex);
    if (!LFM_VULKAN_AVAILABLE) return utf8_to_jstring(env, "CPU backend / DSpark target-only");
    return utf8_to_jstring(env, llama_supports_gpu_offload() ? "Vulkan build / GPU offload available / DSpark target-only" : "Vulkan build / CPU fallback / DSpark target-only");
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeGenerate(JNIEnv *env, jobject, jstring prompt, jint max_tokens) {
    try {
        common_chat_params chat; llama_tokens tokens;
        if (!build_chat(jstring_to_utf8(env, prompt), chat, tokens)) return utf8_to_jstring(env, "[chat template/tokenization failed]");
        return utf8_to_jstring(env, generate_chat_impl(env, chat, tokens, std::max(1, (int)max_tokens), nullptr, {}));
    } catch (const std::exception &e) { set_error(std::string("generate exception: ") + e.what()); return utf8_to_jstring(env, "[generation exception]"); }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeGenerateStream(JNIEnv *env, jobject, jstring prompt, jint max_tokens, jobject callback) {
    try {
        common_chat_params chat; llama_tokens tokens;
        if (!build_chat(jstring_to_utf8(env, prompt), chat, tokens)) return;
        generate_chat_impl(env, chat, tokens, std::max(1, (int)max_tokens), callback, {});
    } catch (const std::exception &e) { set_error(std::string("stream exception: ") + e.what()); }
    catch (...) { set_error("stream exception: unknown"); }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeUnloadModel(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_engine.mutex);
    unload_locked();
}
