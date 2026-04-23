// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "net_impl.hpp"

#include <opencv2/core/utils/logger.hpp>

#ifdef HAVE_ONNXRUNTIME_GENAI
#include <ort_genai.h>

#define OGA_CHECK(call) \
    do \
    { \
        OgaResult* _r = (call); \
        if (_r != nullptr) \
        { \
            std::string _msg(OgaResultGetError(_r)); \
            OgaDestroyResult(_r); \
            CV_Error(cv::Error::StsError, "ORT-GenAI: " + _msg); \
        } \
    } while (0)
#endif

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

struct LLM::Impl {
    Tokenizer opencv_tokenizer_;
    bool useOpencvTokenizer_;
    Net net_;

    Impl() : useOpencvTokenizer_(false) {}
};

LLM::LLM() : impl_(makePtr<LLM::Impl>()) {}

LLM::~LLM() {}

LLM LLM::create(const String& modelPath, int tokenizerType,
                 const String& tokenizerConfigPath, int engine)
{
    LLM llm;

    if (tokenizerType == TOKENIZER_OPENCV_BPE)
    {
        if (tokenizerConfigPath.empty())
        {
            llm.impl_->opencv_tokenizer_ = Tokenizer::load(modelPath);
        }
        else
        {
            llm.impl_->opencv_tokenizer_ = Tokenizer::load(tokenizerConfigPath);
            llm.impl_->net_ = readNetFromONNX(modelPath, engine);
        }
        llm.impl_->useOpencvTokenizer_ = true;
    }
    else if (tokenizerType == TOKENIZER_ORT_GENAI)
    {
        llm.impl_->net_ = readNetFromONNX(modelPath, ENGINE_ORT_GENAI);
        llm.impl_->useOpencvTokenizer_ = false;
        CV_LOG_INFO(NULL, "DNN/LLM: Successfully initialized OGA model for " << modelPath);
    }
    else
    {
        CV_Error(cv::Error::StsBadArg, "Unknown tokenizerType: " + std::to_string(tokenizerType));
    }

    return llm;
}

Mat LLM::tokenize(const String& text) const
{
    CV_Assert(impl_);
    if (impl_->useOpencvTokenizer_)
        return impl_->opencv_tokenizer_.tokenize(text);
#ifdef HAVE_ONNXRUNTIME_GENAI
    auto* ni = impl_->net_.getImpl();
    CV_Assert(ni->oga_tokenizer);
    auto sequences = OgaSequences::Create();
    ni->oga_tokenizer->Encode(text.c_str(), *sequences);
    const int32_t* ptr = sequences->SequenceData(0);
    size_t len = sequences->SequenceCount(0);
    Mat tokens(1, (int)len, CV_32S);
    memcpy(tokens.data, ptr, len * sizeof(int32_t));
    return tokens;
#else
    CV_Error(Error::StsNotImplemented, "tokenize with ORT GenAI requires build with WITH_ONNXRUNTIME_GENAI=ON");
#endif
}

String LLM::detokenize(InputArray tokenIds) const
{
    CV_Assert(impl_);
    if (impl_->useOpencvTokenizer_)
        return impl_->opencv_tokenizer_.detokenize(tokenIds);
#ifdef HAVE_ONNXRUNTIME_GENAI
    auto* ni = impl_->net_.getImpl();
    Mat m = tokenIds.getMat();
    const int32_t* ptr = m.ptr<int32_t>();
    size_t count = (size_t)m.total();

    if (ni->oga_processor)
    {
        const char* outStr = nullptr;
        OGA_CHECK(OgaProcessorDecode(ni->oga_processor.get(), ptr, count, &outStr));
        String result(outStr ? outStr : "");
        OgaDestroyString(outStr);
        return result;
    }

    CV_Assert(ni->oga_tokenizer);
    OgaTokenizerStream* streamPtr = nullptr;
    OGA_CHECK(OgaCreateTokenizerStream(ni->oga_tokenizer.get(), &streamPtr));
    std::string result;
    for (size_t i = 0; i < count; ++i)
    {
        const char* chunk = nullptr;
        OGA_CHECK(OgaTokenizerStreamDecode(streamPtr, ptr[i], &chunk));
        if (chunk)
            result += chunk;
    }
    OgaDestroyTokenizerStream(streamPtr);
    return String(result);
#else
    CV_Error(Error::StsNotImplemented, "detokenize with ORT GenAI requires build with WITH_ONNXRUNTIME_GENAI=ON");
#endif
}

std::vector<int> LLM::encode(const std::string& text)
{
    CV_Assert(impl_);
    if (impl_->useOpencvTokenizer_)
        return impl_->opencv_tokenizer_.encode(text);
#ifdef HAVE_ONNXRUNTIME_GENAI
    auto* ni = impl_->net_.getImpl();
    CV_Assert(ni->oga_tokenizer);
    auto sequences = OgaSequences::Create();
    ni->oga_tokenizer->Encode(text.c_str(), *sequences);
    const int32_t* ptr = sequences->SequenceData(0);
    size_t len = sequences->SequenceCount(0);
    return std::vector<int>(ptr, ptr + len);
#else
    CV_Error(Error::StsNotImplemented, "encode with ORT GenAI requires build with WITH_ONNXRUNTIME_GENAI=ON");
#endif
}

std::string LLM::decode(const std::vector<int>& tokens)
{
    CV_Assert(impl_);
    if (impl_->useOpencvTokenizer_)
        return impl_->opencv_tokenizer_.decode(tokens);
#ifdef HAVE_ONNXRUNTIME_GENAI
    auto* ni = impl_->net_.getImpl();
    CV_Assert(ni->oga_tokenizer);
    OgaTokenizerStream* streamPtr = nullptr;
    OGA_CHECK(OgaCreateTokenizerStream(ni->oga_tokenizer.get(), &streamPtr));
    std::string result;
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        const char* chunk = nullptr;
        OGA_CHECK(OgaTokenizerStreamDecode(streamPtr, (int32_t)tokens[i], &chunk));
        if (chunk)
            result += chunk;
    }
    OgaDestroyTokenizerStream(streamPtr);
    return result;
#else
    CV_Error(Error::StsNotImplemented, "decode with ORT GenAI requires build with WITH_ONNXRUNTIME_GENAI=ON");
#endif
}

#ifdef HAVE_ONNXRUNTIME_GENAI
void LLM::setInputImagePath(const String& path)
{
    CV_Assert(impl_);
    impl_->net_.getImpl()->oga_image_path = path;
}

void LLM::setPrompt(const String& prompt)
{
    CV_Assert(impl_);
    impl_->net_.getImpl()->oga_raw_prompt = prompt;
}

void LLM::setSearchOption(const String& name, double value)
{
    CV_Assert(impl_);
    impl_->net_.getImpl()->oga_search_options_number[std::string(name)] = value;
}

void LLM::setSearchOptionBool(const String& name, bool value)
{
    CV_Assert(impl_);
    impl_->net_.getImpl()->oga_search_options_bool[std::string(name)] = value;
}

void LLM::setGuidance(const String& type, const String& data, bool enableFfTokens)
{
    CV_Assert(impl_);
    auto* ni = impl_->net_.getImpl();
    ni->oga_guidance_type      = std::string(type);
    ni->oga_guidance_data      = std::string(data);
    ni->oga_guidance_ff_tokens = enableFfTokens;
}

String LLM::applyChatTemplate(const String& messages, const String& templateStr,
                               const String& tools, bool addGenerationPrompt) const
{
    CV_Assert(impl_);
    auto* ni = impl_->net_.getImpl();
    CV_Assert(ni->oga_tokenizer);
    const char* tmpl = templateStr.empty() ? nullptr : templateStr.c_str();
    const char* tls  = tools.empty()       ? nullptr : tools.c_str();
    OgaString result = ni->oga_tokenizer->ApplyChatTemplate(tmpl, messages.c_str(), tls, addGenerationPrompt);
    return String(result.p_);
}

String LLM::getModelType() const
{
    CV_Assert(impl_);
    auto* ni = impl_->net_.getImpl();
    CV_Assert(ni->oga_model);
    OgaString t = ni->oga_model->GetType();
    return String(t.p_);
}

String LLM::getDeviceType() const
{
    CV_Assert(impl_);
    auto* ni = impl_->net_.getImpl();
    CV_Assert(ni->oga_model);
    OgaString t = ni->oga_model->GetDeviceType();
    return String(t.p_);
}
#else
void LLM::setInputImagePath(const String&)
{
    CV_Error(cv::Error::StsNotImplemented, "setInputImagePath requires ONNX Runtime GenAI support");
}

void LLM::setPrompt(const String&)
{
    CV_Error(cv::Error::StsNotImplemented, "setPrompt requires ONNX Runtime GenAI support");
}

void LLM::setSearchOption(const String&, double)
{
    CV_Error(cv::Error::StsNotImplemented, "setSearchOption requires ONNX Runtime GenAI support");
}

void LLM::setSearchOptionBool(const String&, bool)
{
    CV_Error(cv::Error::StsNotImplemented, "setSearchOptionBool requires ONNX Runtime GenAI support");
}

void LLM::setGuidance(const String&, const String&, bool)
{
    CV_Error(cv::Error::StsNotImplemented, "setGuidance requires ONNX Runtime GenAI support");
}

String LLM::applyChatTemplate(const String&, const String&, const String&, bool) const
{
    CV_Error(cv::Error::StsNotImplemented, "applyChatTemplate requires ONNX Runtime GenAI support");
}

String LLM::getModelType() const
{
    CV_Error(cv::Error::StsNotImplemented, "getModelType requires ONNX Runtime GenAI support");
}

String LLM::getDeviceType() const
{
    CV_Error(cv::Error::StsNotImplemented, "getDeviceType requires ONNX Runtime GenAI support");
}
#endif  // HAVE_ONNXRUNTIME_GENAI

// ---- Inference ----

Net LLM::getNet() const
{
    CV_Assert(impl_);
    return impl_->net_;
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
