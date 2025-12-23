#pragma once
#include <string>

namespace mls {

struct HttpResp {
  long status = 0;         // HTTP status code (0 if request failed before response)
  std::string body;        // response body or error message
  std::string err;         // non-empty on transport errors (curl)
};

HttpResp http_post_json(const std::string& url,
                        const std::string& json_body,
                        int timeout_ms);

} // namespace mls
