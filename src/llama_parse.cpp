#include "mls/llama_parse.hpp"
#include <nlohmann/json.hpp>
using json = nlohmann::json;


#if __has_include(<nlohmann/json.hpp>)
  #include <nlohmann/json.hpp>
  using json = nlohmann::json;
#else
  #include "third_party/nlohmann/json.hpp"
  using json = nlohmann::json;
#endif

namespace mls {

LlamaResult parse_chat_completions_response(const std::string& json_text) {
  LlamaResult r;
  try {
    auto j = json::parse(json_text);

    // content: choices[0].message.content
    if (j.contains("choices") && j["choices"].is_array() && !j["choices"].empty()) {
      auto& c0 = j["choices"][0];
      if (c0.contains("message") && c0["message"].contains("content")) {
        r.content = c0["message"]["content"].get<std::string>();
      }
    }

    // usage tokens if present
    if (j.contains("usage") && j["usage"].is_object()) {
      auto& u = j["usage"];
      r.usage.has_usage = true;
      if (u.contains("prompt_tokens"))     r.usage.prompt_tokens     = u["prompt_tokens"].get<int>();
      if (u.contains("completion_tokens")) r.usage.completion_tokens = u["completion_tokens"].get<int>();
      if (u.contains("total_tokens"))      r.usage.total_tokens      = u["total_tokens"].get<int>();
    }

    r.ok = true;
    return r;
  } catch (const std::exception& e) {
    r.ok = false;
    r.err = e.what();
    return r;
  }
}

} // namespace mls
