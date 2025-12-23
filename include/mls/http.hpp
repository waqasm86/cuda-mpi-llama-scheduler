#pragma once
#include <string>

namespace mls {

struct HttpResp {
  long status = 0;
  std::string body;
  std::string err; // empty if ok
};

void http_global_init();
void http_global_cleanup();

HttpResp http_post_json(const std::string& url,
                        const std::string& json_body,
                        int timeout_ms,
                        int connect_timeout_ms = 3000,
                        int retries = 2);

} // namespace mls
