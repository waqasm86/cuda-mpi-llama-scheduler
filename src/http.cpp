#include "mls/http.hpp"

#include <curl/curl.h>
#include <mutex>
#include <string>

namespace {

// libcurl write callback
size_t write_cb(char* ptr, size_t size, size_t nmemb, void* userdata) {
  auto* s = static_cast<std::string*>(userdata);
  s->append(ptr, size * nmemb);
  return size * nmemb;
}

// ensure curl_global_init is called once per process
void curl_global_init_once() {
  static std::once_flag once;
  std::call_once(once, []() {
    curl_global_init(CURL_GLOBAL_DEFAULT);
  });
}

} // namespace

namespace mls {

HttpResp http_post_json(const std::string& url,
                        const std::string& json_body,
                        int timeout_ms) {
  curl_global_init_once();

  HttpResp r{};
  CURL* curl = curl_easy_init();
  if (!curl) {
    r.err = "curl_easy_init failed";
    return r;
  }

  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");
  headers = curl_slist_append(headers, "Accept: application/json");

  std::string out;
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

  // POST body
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)json_body.size());

  // capture response body
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &out);

  // timeouts
  if (timeout_ms <= 0) timeout_ms = 60000;
  curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, (long)timeout_ms);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, 5000L);

  // keep it quiet unless you debug
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

  // some sane defaults for local llama-server
  curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);
  curl_easy_setopt(curl, CURLOPT_TCP_KEEPIDLE, 30L);
  curl_easy_setopt(curl, CURLOPT_TCP_KEEPINTVL, 15L);

  // perform
  CURLcode rc = curl_easy_perform(curl);

  long status = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
  r.status = status;

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  if (rc != CURLE_OK) {
    r.err = curl_easy_strerror(rc);
    r.body = out.empty() ? "" : out; // sometimes partial body exists
    return r;
  }

  r.body = std::move(out);
  return r;
}

} // namespace mls
