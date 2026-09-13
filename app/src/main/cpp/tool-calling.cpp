#include "native-lib.cpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
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

jstring tool_result(JNIEnv * env, const std::string & json) { return utf8_to_jstring(env, json); }

void report_progress(JNIEnv * env, jobject callback, const char * stage, long long elapsed_ms) {
    if (!callback) return;
    jclass cls = env->GetObjectClass(callback);
    if (!cls) { if (env->ExceptionCheck()) env->ExceptionClear(); return; }
    jmethodID method = env->GetMethodID(cls, "onProgress", "(Ljava/lang/String;J)V");
    if (!method) { if (env->ExceptionCheck()) env->ExceptionClear(); env->DeleteLocalRef(cls); return; }
    jstring jstage = env->NewStringUTF(stage);
    if (!jstage) { if (env->ExceptionCheck()) env->ExceptionClear(); env->DeleteLocalRef(cls); return; }
    env->CallVoidMethod(callback, method, jstage, static_cast<jlong>(elapsed_ms));
    if (env->ExceptionCheck()) env->ExceptionClear();
    env->DeleteLocalRef(jstage);
    env->DeleteLocalRef(cls);
}

std::string make_error(const std::string & message) {
    return "{\"type\":\"error\",\"error\":\"" + tool_json_escape(message) + "\"}";
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

struct ToolTiming {
    using Clock = std::chrono::steady_clock;
    Clock::time_point total_start = Clock::now();
    Clock::time_point last = total_start;
    long long input_ms=0, parse_messages_ms=0, parse_tools_ms=0, template_ms=0, tokenize_ms=0;
    long long sampler_ms=0, kv_clear_ms=0, prefill_ms=0, first_sample_ms=0, generation_ms=0, parse_result_ms=0;
    long long mark() { auto now=Clock::now(); auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(now-last).count(); last=now; return ms; }
    long long total_ms() const { return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now()-total_start).count(); }
    std::string json() const {
        return "{\"total_ms\":"+std::to_string(total_ms())+",\"input_ms\":"+std::to_string(input_ms)+
        ",\"parse_messages_ms\":"+std::to_string(parse_messages_ms)+",\"parse_tools_ms\":"+std::to_string(parse_tools_ms)+
        ",\"template_ms\":"+std::to_string(template_ms)+",\"tokenize_ms\":"+std::to_string(tokenize_ms)+
        ",\"sampler_ms\":"+std::to_string(sampler_ms)+",\"kv_clear_ms\":"+std::to_string(kv_clear_ms)+
        ",\"prefill_ms\":"+std::to_string(prefill_ms)+",\"first_sample_ms\":"+std::to_string(first_sample_ms)+
        ",\"generation_ms\":"+std::to_string(generation_ms)+",\"parse_result_ms\":"+std::to_string(parse_result_ms)+"}";
    }
};

std::string add_timing(const std::string & result, const ToolTiming & t) {
    if (!result.empty() && result.back()=='}') return result.substr(0,result.size()-1)+",\"timing\":"+t.json()+"}";
    return result;
}
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_lfmmobile_LlamaEngine_nativeGenerateToolStep(JNIEnv * env, jobject, jstring messages_json, jstring tools_json, jint max_tokens, jobject callback) {
    std::string stage="entry";
    ToolTiming timing;
    auto progress = [&](const char * s) { stage=s; report_progress(env, callback, s, timing.total_ms()); };
    try {
        auto get_string = [&](jstring value)->std::string {
            if (!value) return {};
            const jsize n=env->GetStringLength(value); const jchar * p=env->GetStringChars(value,nullptr);
            if (!p) { if(env->ExceptionCheck()) env->ExceptionClear(); throw std::runtime_error("JNI GetStringChars failed"); }
            std::string r; r.reserve(static_cast<size_t>(n)*2);
            for(jsize i=0;i<n;++i){ uint32_t cp=p[i]; if(cp>=0xD800&&cp<=0xDBFF&&i+1<n&&p[i+1]>=0xDC00&&p[i+1]<=0xDFFF) cp=0x10000+((cp-0xD800)<<10)+(p[++i]-0xDC00);
                if(cp<=0x7f) r.push_back((char)cp); else if(cp<=0x7ff){r.push_back((char)(0xc0|(cp>>6)));r.push_back((char)(0x80|(cp&63)));}
                else if(cp<=0xffff){r.push_back((char)(0xe0|(cp>>12)));r.push_back((char)(0x80|((cp>>6)&63)));r.push_back((char)(0x80|(cp&63)));}
                else{r.push_back((char)(0xf0|(cp>>18)));r.push_back((char)(0x80|((cp>>12)&63)));r.push_back((char)(0x80|((cp>>6)&63)));r.push_back((char)(0x80|(cp&63)));}}
            env->ReleaseStringChars(value,p); return r;
        };

        progress("validate_engine");
        if(!g_engine.model||!g_engine.context||!g_engine.vocab) return tool_result(env,make_error("model not loaded"));
        progress("jni_input_conversion");
        const std::string messages_text=get_string(messages_json), tools_text=get_string(tools_json); timing.input_ms=timing.mark();
        progress("parse_messages_json"); const common_json messages_value=common_json::parse(messages_text); timing.parse_messages_ms=timing.mark();
        progress("parse_tools_json"); const common_json tools_value=common_json::parse(tools_text); timing.parse_tools_ms=timing.mark();
        progress("parse_messages_oaicompat"); const auto messages=common_chat_msgs_parse_oaicompat(messages_value); timing.parse_messages_ms+=timing.mark();
        progress("parse_tools_oaicompat"); const auto tools=common_chat_tools_parse_oaicompat(tools_value); timing.parse_tools_ms+=timing.mark();
        if(tools.empty()) return tool_result(env,make_error("no tools supplied"));

        progress("chat_template_init"); auto templates=common_chat_templates_init(g_engine.model,"");
        if(!templates) return tool_result(env,make_error("chat template init failed")); timing.template_ms=timing.mark();
        progress("chat_template_apply"); common_chat_templates_inputs inputs; inputs.messages=messages; inputs.tools=tools; inputs.tool_choice=COMMON_CHAT_TOOL_CHOICE_AUTO; inputs.parallel_tool_calls=false; inputs.add_generation_prompt=true; inputs.use_jinja=true; inputs.enable_thinking=true;
        const common_chat_params chat=common_chat_templates_apply(templates.get(),inputs); timing.template_ms+=timing.mark();
        if(chat.prompt.empty()) return tool_result(env,make_error("empty chat prompt"));

        progress("tokenize_tool_prompt"); const llama_tokens prompt_tokens=common_tokenize(g_engine.context,chat.prompt,true,true); timing.tokenize_ms=timing.mark();
        const uint32_t n_ctx=llama_n_ctx(g_engine.context); if(prompt_tokens.empty()) return tool_result(env,make_error("tool prompt tokenization failed")); if(prompt_tokens.size()+1>=n_ctx) return tool_result(env,make_error("prompt exceeds context"));

        progress("init_tool_sampler"); common_params_sampling sampling; sampling.temp=.2f; sampling.top_k=40; sampling.top_p=.95f;
        if(!chat.grammar.empty()){ progress("init_tool_grammar"); try{sampling.grammar=common_grammar(COMMON_GRAMMAR_TYPE_TOOL_CALLS,chat.grammar); sampling.generation_prompt=chat.generation_prompt;}catch(...){sampling.grammar=common_grammar();sampling.generation_prompt.clear();} }
        progress("init_tool_sampler_create"); common_sampler_ptr sampler;
        try{sampler.reset(common_sampler_init(g_engine.model,sampling));}
        catch(...){sampling.grammar=common_grammar();sampling.generation_prompt.clear();sampler.reset(common_sampler_init(g_engine.model,sampling));}
        timing.sampler_ms=timing.mark(); if(!sampler) return tool_result(env,make_error("tool sampler init failed"));

        progress("clear_tool_kv"); llama_memory_clear(llama_get_memory(g_engine.context),false); timing.kv_clear_ms=timing.mark();
        const uint32_t n_batch=std::max<uint32_t>(1,llama_n_batch(g_engine.context)); llama_batch batch=llama_batch_init(std::min<uint32_t>(n_batch,(uint32_t)prompt_tokens.size()),0,1);
        progress("prefill_tool_prompt");
        for(size_t i=0;i<prompt_tokens.size();++i){common_batch_add(batch,prompt_tokens[i],(llama_pos)i,{0},i+1==prompt_tokens.size()); if(batch.n_tokens==(int)n_batch||i+1==prompt_tokens.size()){if(llama_decode(g_engine.context,batch)!=0){llama_batch_free(batch);return tool_result(env,make_error("tool prompt decode failed"));}common_batch_clear(batch);}}
        llama_batch_free(batch); timing.prefill_ms=timing.mark(); report_progress(env,callback,"prefill_complete",timing.total_ms());

        progress("sample_first_tool_token"); llama_token next=common_sampler_sample(sampler.get(),g_engine.context,(int)prompt_tokens.size()-1); common_sampler_accept(sampler.get(),next,true); timing.first_sample_ms=timing.mark();
        std::string generated; const int limit=std::max(1,(int)max_tokens); progress("generate_tool_tokens");
        for(int i=0;i<limit;++i){if(llama_vocab_is_eog(g_engine.vocab,next))break; generated+=common_token_to_piece(g_engine.context,next); llama_batch b=llama_batch_init(1,0,1); common_batch_add(b,next,(llama_pos)(prompt_tokens.size()+i),{0},true); if(llama_decode(g_engine.context,b)!=0){llama_batch_free(b);return tool_result(env,make_error("tool decode failed"));} next=common_sampler_sample(sampler.get(),g_engine.context,0); common_sampler_accept(sampler.get(),next,true); llama_batch_free(b);}
        timing.generation_ms=timing.mark();
        progress("parse_generated_tool_call"); common_chat_parser_params parser(chat); parser.parse_tool_calls=true; const common_chat_msg parsed=common_chat_parse(generated,false,parser); timing.parse_result_ms=timing.mark();
        const std::string result=parsed.tool_calls.empty()?make_final(parsed):make_tools(parsed); report_progress(env,callback,"complete",timing.total_ms());
        return tool_result(env,add_timing(result,timing));
    }catch(const std::exception&e){return tool_result(env,make_error("native tool-step exception at "+stage+": "+(e.what()&&*e.what()?e.what():"<empty what()>")));}
    catch(...){return tool_result(env,make_error("native tool-step unknown exception at "+stage));}
}
