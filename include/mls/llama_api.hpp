#pragma once
#include <string>

namespace mls {
// Builds an OpenAI-compatible chat completion request body.
// llama.cpp's llama-server supports /v1/chat/completions when OpenAI-compatible mode is enabled.
std::string build_chat_body(const std::string& user_text, int n_predict);
} // namespace mls
