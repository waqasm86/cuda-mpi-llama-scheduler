#include "mls/http.hpp"
#include <curl/curl.h>
#include <stdexcept>

namespace {

size_t write_cb(char* ptr, size_t size, size_t nmemb, void* userdata) {
  auto* s = static_cast<std::string*>(userdata);
  s->append(ptr, size * nmemb);
  return size * nmemb;
}

bool is_retryable(CURLcode rc) {
  return rc == CURLE_COULDNT_CONNECT ||
         rc == CURLE_OPERATION_TIMEDOUT ||
         rc == CURLE_COULDNT_RESOLVE_HOST ||
         rc == CURLE_RECV_ERROR ||
         rc == CURLE_SEND_ERROR;
}

} // namespace

namespace mls {

void http_global_init() {
  curl_global_init(CURL_GLOBAL_DEFAULT);
}

void http_global_cleanup() {
  curl_global_cleanup();
}

HttpResp http_post_json(const std::string& url,
                        const std::string& json_body,
                        int timeout_ms,
                        int connect_timeout_ms,
                        int retries) {
  HttpResp last{};

  for (int attempt = 0; attempt <= retries; ++attempt) {
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("curl_easy_init failed");

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    std::string out;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)json_body.size());

    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, connect_timeout_ms);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout_ms);

    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &out);

    CURLcode rc = curl_easy_perform(curl);

    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    last.status = status;
    last.body = std::move(out);
    last.err = (rc == CURLE_OK) ? "" : curl_easy_strerror(rc);

    if (rc == CURLE_OK) {
      return last;
    }

    if (!is_retryable(rc) || attempt == retries) {
      return last;
    }
  }

  return last;
}

} // namespace mls
