#pragma once
#include <string>
#include <cstdint>

namespace mls {

struct LlamaUsage {
  int prompt_tokens = 0;
  int completion_tokens = 0;
  int total_tokens = 0;
  bool has_usage = false;
};

struct LlamaResult {
  std::string content;   // assistant text
  LlamaUsage usage;
  bool ok = false;
  std::string err;
};

LlamaResult parse_chat_completions_response(const std::string& json_text);

} // namespace mls
