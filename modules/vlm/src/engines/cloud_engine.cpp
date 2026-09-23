// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "cloud_engine.hpp"
#include "../vlm_model_base.hpp"
#include "../http_client.hpp"
#include "../base64.hpp"
#include "../json_parser.hpp"

#include <cctype>
#include <cstdio>
#include <cstring>

namespace cv { namespace vlm {

namespace {

std::string jsonEscape(const std::string& s)
{
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s)
    {
        switch (c)
        {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20)
            {
                char buf[8];
                snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            }
            else
                out += (char)c;
        }
    }
    return out;
}

const char* const REDACTED = "<redacted>";

// Some providers echo the offending credential back in the body of an authentication error,
// so nothing from a response body reaches an exception message -- which a caller may log --
// without the key being taken out of it first. The known prefixes cover the case where the
// provider echoes only part of the key, which an exact match on apiKey would miss.
std::string redactSecrets(const std::string& text, const String& apiKey)
{
    const size_t redactedLen = strlen(REDACTED);
    std::string out = text;

    if (apiKey.size() >= 8)
    {
        for (size_t at = out.find(apiKey); at != std::string::npos;
             at = out.find(apiKey, at + redactedLen))
            out.replace(at, apiKey.size(), REDACTED);
    }

    static const char* const prefixes[] = { "sk-", "xai-", "AIza", "gsk_" };
    for (const char* prefix : prefixes)
    {
        const size_t prefixLen = strlen(prefix);
        size_t at = 0;
        while ((at = out.find(prefix, at)) != std::string::npos)
        {
            size_t end = at + prefixLen;
            while (end < out.size() &&
                   (isalnum((unsigned char)out[end]) || out[end] == '-' || out[end] == '_'))
                end++;
            out.replace(at, end - at, std::string(prefix) + REDACTED);
            at += prefixLen + redactedLen;
        }
    }
    return out;
}

void checkHttpStatus(const HttpResponse& response, const String& provider, const String& apiKey)
{
    if (response.statusCode < 200 || response.statusCode >= 300)
    {
        const std::string snippet = redactSecrets(response.body.substr(0, 500), apiKey);
        CV_Error(Error::StsError,
                 cv::format("vlm: %s request failed with HTTP status %ld: %s",
                            provider.c_str(), response.statusCode, snippet.c_str()));
    }
}

String extractOrThrow(const JsonValue& node, const String& provider)
{
    if (node.type() != JsonValue::STRING)
        CV_Error(Error::StsError,
                 "vlm: unexpected " + provider + " response shape (missing generated text)");
    return node.asString();
}

class CloudVLMModel CV_FINAL : public VLMModelBase
{
public:
    CloudVLMModel(VLMModelType modelType, const String& modelName, const String& apiKey)
        : modelType_(modelType), modelName_(modelName), apiKey_(apiKey)
    {
        CV_CheckFalse(apiKey.empty(), "vlm: api_key is required for cloud VLMModelType values");
        // Deliberately names no model: providers retire names, and a suggestion baked in
        // here goes stale silently and sends users chasing a 404 that is not their fault.
        CV_CheckFalse(modelName.empty(),
                      "vlm: a model name (model_dir) is required for cloud VLMModelType "
                      "values. Take it from the provider's own model list -- OpenAI and Grok "
                      "serve one at GET /v1/models, Gemini at "
                      "generativelanguage.googleapis.com/v1beta/models, Anthropic at "
                      "api.anthropic.com/v1/models");
    }

    void setPreferableDevice(const String&) CV_OVERRIDE
    {
    }

    void reset() CV_OVERRIDE
    {
    }

    int lastTokensUsed() const CV_OVERRIDE
    {
        return lastTokensUsed_;
    }

    String infer(InputArray image, const String& prompt, int max_new_tokens) CV_OVERRIDE
    {
        Mat img = image.getMat();
        CV_CheckFalse(img.empty(), "vlm: input image is empty");

        std::vector<uchar> encoded;
        imencode(".png", img, encoded);
        std::string imageB64 = base64Encode(encoded.data(), encoded.size());
        String actualPrompt = prompt.empty() ? "OCR" : prompt;

        switch (modelType_)
        {
        case VLM_MODEL_OPENAI:    return inferOpenAICompatible("https://api.openai.com/v1/chat/completions", imageB64, actualPrompt, max_new_tokens);
        case VLM_MODEL_GROK:      return inferOpenAICompatible("https://api.x.ai/v1/chat/completions", imageB64, actualPrompt, max_new_tokens);
        case VLM_MODEL_ANTHROPIC: return inferAnthropic(imageB64, actualPrompt, max_new_tokens);
        case VLM_MODEL_GEMINI:    return inferGemini(imageB64, actualPrompt, max_new_tokens);
        default:
            CV_Error(Error::StsBadArg, "vlm: not a cloud VLMModelType");
        }
    }

private:
    String inferOpenAICompatible(const String& url, const std::string& imageB64,
                                  const String& prompt, int maxNewTokens)
    {
        std::ostringstream body;
        body << "{"
             << "\"model\":\"" << jsonEscape(modelName_) << "\","
             << "\"max_tokens\":" << maxNewTokens << ","
             << "\"messages\":[{\"role\":\"user\",\"content\":["
             << "{\"type\":\"text\",\"text\":\"" << jsonEscape(prompt) << "\"},"
             << "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64," << imageB64 << "\"}}"
             << "]}]}";

        std::vector<std::string> headers = {
            "Content-Type: application/json",
            "Authorization: Bearer " + apiKey_
        };

        HttpResponse response = httpPostJson(url, body.str(), headers);
        checkHttpStatus(response, "OpenAI-compatible", apiKey_);

        const JsonValue root = jsonParse(response.body);
        lastTokensUsed_ = root["usage"]["total_tokens"].asInt(-1);
        const JsonValue& choices = root["choices"];
        if (choices.size() == 0)
            CV_Error(Error::StsError,
                     "vlm: unexpected OpenAI-compatible response shape (missing choices)");
        return extractOrThrow(choices[0]["message"]["content"], "OpenAI-compatible");
    }

    String inferAnthropic(const std::string& imageB64, const String& prompt, int maxNewTokens)
    {
        std::ostringstream body;
        body << "{"
             << "\"model\":\"" << jsonEscape(modelName_) << "\","
             << "\"max_tokens\":" << maxNewTokens << ","
             << "\"messages\":[{\"role\":\"user\",\"content\":["
             << "{\"type\":\"image\",\"source\":{\"type\":\"base64\",\"media_type\":\"image/png\",\"data\":\"" << imageB64 << "\"}},"
             << "{\"type\":\"text\",\"text\":\"" << jsonEscape(prompt) << "\"}"
             << "]}]}";

        std::vector<std::string> headers = {
            "Content-Type: application/json",
            "x-api-key: " + apiKey_,
            "anthropic-version: 2023-06-01"
        };

        HttpResponse response = httpPostJson("https://api.anthropic.com/v1/messages", body.str(), headers);
        checkHttpStatus(response, "Anthropic", apiKey_);

        const JsonValue root = jsonParse(response.body);
        const int inputTokens = root["usage"]["input_tokens"].asInt(0);
        const int outputTokens = root["usage"]["output_tokens"].asInt(0);
        lastTokensUsed_ = (inputTokens > 0 || outputTokens > 0) ? inputTokens + outputTokens : -1;
        const JsonValue& content = root["content"];
        if (content.size() == 0)
            CV_Error(Error::StsError,
                     "vlm: unexpected Anthropic response shape (missing content)");
        return extractOrThrow(content[0]["text"], "Anthropic");
    }

    String inferGemini(const std::string& imageB64, const String& prompt, int maxNewTokens)
    {
        std::ostringstream body;
        body << "{"
             << "\"contents\":[{\"parts\":["
             << "{\"text\":\"" << jsonEscape(prompt) << "\"},"
             << "{\"inline_data\":{\"mime_type\":\"image/png\",\"data\":\"" << imageB64 << "\"}}"
             << "]}],"
             << "\"generationConfig\":{\"maxOutputTokens\":" << maxNewTokens << "}"
             << "}";

        String url = cv::format("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent",
                                 modelName_.c_str());
        std::vector<std::string> headers = {
            "Content-Type: application/json",
            "x-goog-api-key: " + apiKey_
        };

        HttpResponse response = httpPostJson(url, body.str(), headers);
        checkHttpStatus(response, "Gemini", apiKey_);

        const JsonValue root = jsonParse(response.body);
        lastTokensUsed_ = root["usageMetadata"]["totalTokenCount"].asInt(-1);
        const JsonValue& candidates = root["candidates"];
        if (candidates.size() == 0)
            CV_Error(Error::StsError,
                     "vlm: unexpected Gemini response shape (missing candidates)");
        const JsonValue& parts = candidates[0]["content"]["parts"];
        if (parts.size() == 0)
            CV_Error(Error::StsError,
                     "vlm: unexpected Gemini response shape (missing content parts)");
        return extractOrThrow(parts[0]["text"], "Gemini");
    }

    VLMModelType modelType_;
    String modelName_;
    String apiKey_;
    int lastTokensUsed_ = -1;
};

} // namespace

Ptr<VLMModel> createCloudModel(VLMModelType model_type, const String& model_name, const String& api_key)
{
    return makePtr<CloudVLMModel>(model_type, model_name, api_key);
}

}} // namespace cv::vlm
