// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "http_client.hpp"

#ifdef HAVE_CURL

#include <curl/curl.h>

namespace cv { namespace vlm {

namespace {

size_t writeCallback(char* ptr, size_t size, size_t nmemb, void* userdata)
{
    std::string* out = static_cast<std::string*>(userdata);
    out->append(ptr, size * nmemb);
    return size * nmemb;
}

struct CurlGlobalInit
{
    CurlGlobalInit() { curl_global_init(CURL_GLOBAL_DEFAULT); }
    ~CurlGlobalInit() { curl_global_cleanup(); }
};

} // namespace

HttpResponse httpPostJson(const std::string& url, const std::string& jsonBody,
                          const std::vector<std::string>& headers)
{
    static CurlGlobalInit globalInit;

    CURL* curl = curl_easy_init();
    if (!curl)
        CV_Error(Error::StsError, "vlm: failed to initialize libcurl");

    struct curl_slist* headerList = nullptr;
    for (const std::string& h : headers)
        headerList = curl_slist_append(headerList, h.c_str());

    std::string responseBody;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonBody.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)jsonBody.size());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headerList);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseBody);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK)
    {
        std::string err = curl_easy_strerror(res);
        curl_slist_free_all(headerList);
        curl_easy_cleanup(curl);
        CV_Error(Error::StsError, "vlm: HTTP request failed: " + err);
    }

    long statusCode = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &statusCode);

    curl_slist_free_all(headerList);
    curl_easy_cleanup(curl);

    HttpResponse response;
    response.statusCode = statusCode;
    response.body = responseBody;
    return response;
}

}} // namespace cv::vlm

#else // !HAVE_CURL

namespace cv { namespace vlm {

HttpResponse httpPostJson(const std::string&, const std::string&, const std::vector<std::string>&)
{
    CV_Error(Error::StsNotImplemented,
             "vlm: cloud model types require OpenCV to be built with libcurl available");
}

}} // namespace cv::vlm

#endif // HAVE_CURL
