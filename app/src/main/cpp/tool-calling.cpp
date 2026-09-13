#include "native-lib.cpp"

#include <cstdio>
#include <string>

#include "json.h"

namespace {
std::string tool_json_escape(const std::string & s) {
    std::string out; out.reserve(s.size() + 16);
    for (const unsigned char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) { char b[7]; std::snprintf(b, sizeof(b), "\\u%04x", c); out += b; }
                else out.push_back(static_cast<char>(c));
        }
    }
    return out;
}

std::string make_final(const common_chat_msg & msg) {
    return "{\"type\":\"final\",\"content\":\"" + tool_json_escape(msg.content) +
           "\",\"reasoning\":\"" + tool_json_escape(msg.reasoning_content) + "\"}";
}

std::string make_tools(const common_chat_msg & msg) {
    std::string out = "{\"type\":\"tool_calls\",\"calls\":[";
    for (size_t i = 0; i < msg.tool_calls.size(); ++i) {
        if (i) out += ',';
        const auto & c = msg.tool_calls[i];
        out += "{\"name\":\"" + tool_json_escape(c.name) + "\",\"arguments\":\"" + tool_json_escape(c.arguments) +
               "\",\"id\":\"" + tool_json_escape(c.id) + "\"}";
    }
    return out + "],\"reasoning\":\"" + tool_json_escape(msg.reasoning_content) + "\"}";
}
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeGenerateToolStep(JNIEnv * env, jobject, jstring messages_json, jstring tools_json, jint max_tokens) {
    auto get_string = [&](jstring value) -> std::string {
        if (!value) return {};
        const jsize n = env->GetStringLength(value); const jchar * p = env->GetStringChars(value, nullptr);
        std::string r; r.reserve(static_cast<size_t>(n) * 2);
        for (jsize i = 0; i < n; ++i) {
            uint32_t cp = p[i];
            if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < n && p[i + 1] >= 0xDC00 && p[i + 1] <= 0xDFFF) { cp = 0x10000 + ((cp - 0xD800) << 10) + (p[++i] - 0xDC00); }
            if (cp <= 0x7f) r.push_back(static_cast<char>(cp));
            else if (cp <= 0x7ff) { r.push_back(static_cast<char>(0xc0 | (cp >> 6))); r.push_back(static_cast<char>(0x80 | (cp & 63))); }
            else if (cp <= 0xffff) { r.push_back(static_cast<char>(0xe0 | (cp >> 12))); r.push_back(static_cast<char>(0x80 | ((cp >> 6) & 63))); r.push_back(static_cast<char>(0x80 | (cp & 63))); }
            else { r.push_back(static_cast<char>(0xf0 | (cp >> 18))); r.push_back(static_cast<char>(0x80 | ((cp >> 12) & 63))); r.push_back(static_cast<char>(0x80 | ((cp >> 6) & 63))); r.push_back(static_cast<char>(0x80 | (cp & 63))); }
        }
        env->ReleaseStringChars(value, p); return r;
    };
    try {
        if (!g_engine.model || !g_engine.context || !g_engine.vocab) return env->NewStringUTF("{\"type\":\"error\",\"error\":\"model not loaded\"}");
        const common_json messages_value = common_json::parse(get_string(messages_json));
        const common_json tools_value = common_json::parse(get_string(tools_json));
        const auto messages = common_chat_msgs_parse_oaicompat(messages_value);
        const auto tools = common_chat_tools_parse_oaicompat(tools_value);
        if (tools.empty()) return env->NewStringUTF("{\"type\":\"error\",\"error\":\"no tools supplied\"}");

        auto templates = common_chat_templates_init(g_engine.model, "");
        if (!templates) return env->NewStringUTF("{\"type\":\"error\",\"error\":\"chat template init failed\"}");
        common_chat_templates_inputs inputs;
        inputs.messages = messages;
        inputs.tools = tools;
        inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
        inputs.parallel_tool_calls = false;
        inputs.add_generation_prompt = true;
        inputs.use_jinja = true;
        inputs.enable_thinking = true;
        const common_chat_params chat = common_chat_templates_apply(templates.get(), inputs);
        if (chat.prompt.empty()) return env->NewStringUTF("{\"type\":\"error\",\"error\":\"empty chat prompt\"}");

        const llama_tokens prompt_tokens = common_tokenize(g_engine.context, chat.prompt, true, true);
        const uint32_t n_ctx = llama_n_ctx(g_engine.context);
        if (prompt_tokens.empty() || prompt_tokens.size() + 1 >= n_ctx) return env->NewStringUTF("{\"type\":\"error\",\"error\":\"prompt exceeds context\"}");

        common_params_sampling sampling;
        sampling.temp = 0.2f; sampling.top_k = 40; sampling.top_p = 0.95f;
        if (!chat.grammar.empty()) {
            sampling.grammar = common_grammar(COMMON_GRAMMAR_TYPE_TOOL_CALLS, chat.grammar);
            sampling.generation_prompt = chat.generation_prompt;
        }
        common_sampler_ptr sampler(common_sampler_init(g_engine.model, sampling));
        if (!sampler) return env->NewStringUTF("{\"type\":\"error\",\"error\":\"tool sampler init failed\"}");

        llama_memory_clear(llama_get_memory(g_engine.context), false);
        const uint32_t n_batch = std::max<uint32_t>(1, llama_n_batch(g_engine.context));
        llama_batch batch = llama_batch_init(std::min<uint32_t>(n_batch, static_cast<uint32_t>(prompt_tokens.size())), 0, 1);
        for (size_t i = 0; i < prompt_tokens.size(); ++i) {
            common_batch_add(batch, prompt_tokens[i], static_cast<llama_pos>(i), {0}, i + 1 == prompt_tokens.size());
            if (batch.n_tokens == static_cast<int>(n_batch) || i + 1 == prompt_tokens.size()) {
                if (llama_decode(g_engine.context, batch) != 0) { llama_batch_free(batch); return env->NewStringUTF("{\"type\":\"error\",\"error\":\"tool prompt decode failed\"}"); }
                common_batch_clear(batch);
            }
        }
        llama_batch_free(batch);

        llama_token next = common_sampler_sample(sampler.get(), g_engine.context, static_cast<int>(prompt_tokens.size()) - 1);
        common_sampler_accept(sampler.get(), next, true);
        std::string generated;
        const int limit = std::max(1, static_cast<int>(max_tokens));
        for (int i = 0; i < limit; ++i) {
            if (llama_vocab_is_eog(g_engine.vocab, next)) break;
            generated += common_token_to_piece(g_engine.context, next);
            llama_batch b = llama_batch_init(1, 0, 1);
            common_batch_add(b, next, static_cast<llama_pos>(prompt_tokens.size() + i), {0}, true);
            if (llama_decode(g_engine.context, b) != 0) { llama_batch_free(b); return env->NewStringUTF("{\"type\":\"error\",\"error\":\"tool decode failed\"}"); }
            next = common_sampler_sample(sampler.get(), g_engine.context, 0);
            common_sampler_accept(sampler.get(), next, true);
            llama_batch_free(b);
        }

        common_chat_parser_params parser(chat);
        parser.parse_tool_calls = true;
        const common_chat_msg parsed = common_chat_parse(generated, false, parser);
        const std::string result = parsed.tool_calls.empty() ? make_final(parsed) : make_tools(parsed);
        return env->NewStringUTF(result.c_str());
    } catch (const std::exception & e) {
        const std::string result = std::string("{\"type\":\"error\",\"error\":\"") + tool_json_escape(e.what()) + "\"}";
        return env->NewStringUTF(result.c_str());
    } catch (...) {
        return env->NewStringUTF("{\"type\":\"error\",\"error\":\"unknown tool-generation error\"}");
    }
}
