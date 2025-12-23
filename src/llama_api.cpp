#include "mls/llama_api.hpp"
#include <sstream>

namespace mls {

std::string build_chat_body(const std::string& user_text, int n_predict) {
  std::ostringstream o;
  o << "{"
    << "\"messages\":[{\"role\":\"user\",\"content\":\"";
  for (char c : user_text) {
    if (c == '\\' || c == '\"') o << '\\';
    if (c == '\n') o << "\\n";
    else o << c;
  }
  o << "\"}],"
    << "\"temperature\":0.2,"
    << "\"stream\":false,"
    << "\"n_predict\":" << n_predict
    << "}";
  return o.str();
}

} // namespace mls
