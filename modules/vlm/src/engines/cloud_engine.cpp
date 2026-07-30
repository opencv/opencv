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

#include <cstdio>

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

void checkHttpStatus(const HttpResponse& response, const String& provider)
{
    if (response.statusCode < 200 || response.statusCode >= 300)
    {
        std::string snippet = response.body.substr(0, 500);
        CV_Error(Error::StsError, cv::format("vlm: %s request failed with HTTP status %ld: %s",
                                              provider.c_str(), response.statusCode, snippet.c_str()));
    }
}

FileStorage parseJson(const std::string& body, const String& provider)
{
    FileStorage fs(body, FileStorage::READ | FileStorage::MEMORY | FileStorage::FORMAT_JSON);
    if (!fs.isOpened())
        CV_Error(Error::StsError, "vlm: could not parse " + provider + " response as JSON");
    return fs;
}

String extractOrThrow(const FileNode& node, const String& provider)
{
    if (node.empty())
        CV_Error(Error::StsError, "vlm: unexpected " + provider + " response shape (missing generated text)");
    return (String)node;
}

int readIntOr(const FileNode& node, int fallback)
{
    return node.empty() ? fallback : (int)node;
}

class CloudVLMModel CV_FINAL : public VLMModelBase
{
public:
    CloudVLMModel(VLMModelType modelType, const String& modelName, const String& apiKey)
        : modelType_(modelType), modelName_(modelName), apiKey_(apiKey)
    {
        CV_CheckFalse(apiKey.empty(), "vlm: api_key is required for cloud VLMModelType values");
        CV_CheckFalse(modelName.empty(),
                      "vlm: a model name (model_dir) is required for cloud VLMModelType values, "
                      "e.g. 'gpt-4o', 'claude-3-5-sonnet-20241022', 'gemini-2.0-flash', 'grok-2-vision-1212' "
                      "-- check the provider's current model list, names change over time");
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
        checkHttpStatus(response, "OpenAI-compatible");

        FileStorage fs = parseJson(response.body, "OpenAI-compatible");
        lastTokensUsed_ = readIntOr(fs["usage"]["total_tokens"], -1);
        FileNode choices = fs["choices"];
        if (choices.size() == 0)
            CV_Error(Error::StsError, "vlm: unexpected OpenAI-compatible response shape (missing choices)");
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
        checkHttpStatus(response, "Anthropic");

        FileStorage fs = parseJson(response.body, "Anthropic");
        int inputTokens = readIntOr(fs["usage"]["input_tokens"], 0);
        int outputTokens = readIntOr(fs["usage"]["output_tokens"], 0);
        lastTokensUsed_ = (inputTokens > 0 || outputTokens > 0) ? inputTokens + outputTokens : -1;
        FileNode content = fs["content"];
        if (content.size() == 0)
            CV_Error(Error::StsError, "vlm: unexpected Anthropic response shape (missing content)");
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
        checkHttpStatus(response, "Gemini");

        FileStorage fs = parseJson(response.body, "Gemini");
        lastTokensUsed_ = readIntOr(fs["usageMetadata"]["totalTokenCount"], -1);
        FileNode candidates = fs["candidates"];
        if (candidates.size() == 0)
            CV_Error(Error::StsError, "vlm: unexpected Gemini response shape (missing candidates)");
        FileNode parts = candidates[0]["content"]["parts"];
        if (parts.size() == 0)
            CV_Error(Error::StsError, "vlm: unexpected Gemini response shape (missing content parts)");
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
