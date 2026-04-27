// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "net_impl.hpp"
#include "tokenizer/tokenizer_impl.hpp"

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

#ifdef HAVE_ONNXRUNTIME_GENAI
struct OgaTokenizerImpl : public Tokenizer::Impl {
    Net::Impl* ni;

    explicit OgaTokenizerImpl(Net::Impl* ni_) : ni(ni_) {}

    std::vector<int> encode(const std::string& text) override
    {
        CV_Assert(ni->oga_tokenizer);
        auto sequences = OgaSequences::Create();
        ni->oga_tokenizer->Encode(text.c_str(), *sequences);
        const int32_t* ptr = sequences->SequenceData(0);
        size_t len = sequences->SequenceCount(0);
        return std::vector<int>(ptr, ptr + len);
    }

    std::string decode(const std::vector<int>& tokens) override
    {
        if (ni->oga_processor)
        {
            const char* outStr = nullptr;
            OGA_CHECK(OgaProcessorDecode(ni->oga_processor.get(),
                      (const int32_t*)tokens.data(), tokens.size(), &outStr));
            std::string result(outStr ? outStr : "");
            OgaDestroyString(outStr);
            return result;
        }

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
    }
};
#endif

struct LLM::Impl {
    Tokenizer tokenizer_;
    Net net_;

    Impl() {}
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
            llm.impl_->tokenizer_ = Tokenizer::load(modelPath);
        }
        else
        {
            llm.impl_->tokenizer_ = Tokenizer::load(tokenizerConfigPath);
            llm.impl_->net_ = readNetFromONNX(modelPath, engine);
        }
    }
    else if (tokenizerType == TOKENIZER_ORT_GENAI)
    {
        llm.impl_->net_ = readNetFromONNX(modelPath, ENGINE_ORT_GENAI);
#ifdef HAVE_ONNXRUNTIME_GENAI
        llm.impl_->tokenizer_ = Tokenizer(Ptr<Tokenizer::Impl>(new OgaTokenizerImpl(llm.impl_->net_.getImpl())));
#endif
        CV_LOG_INFO(NULL, "DNN/LLM: Successfully initialized OGA model for " << modelPath);
    }
    else
    {
        CV_Error(cv::Error::StsBadArg, "Unknown tokenizerType: " + std::to_string(tokenizerType));
    }

    return llm;
}

Tokenizer LLM::getTokenizer() const
{
    CV_Assert(impl_);
    return impl_->tokenizer_;
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

Mat LLM::run()
{
    CV_Assert(impl_);
    return impl_->net_.forward();
}

Mat LLM::run(const std::vector<int>& tokens, const String& inputName)
{
    CV_Assert(impl_);
    Mat tokensMat(1, (int)tokens.size(), CV_32S);
    std::memcpy(tokensMat.data, tokens.data(), tokens.size() * sizeof(int));
    if (inputName.empty())
        impl_->net_.setInput(tokensMat);
    else
        impl_->net_.setInput(tokensMat, inputName);
    return impl_->net_.forward();
}

Mat LLM::run(const std::vector<Mat>& inputs, const std::vector<String>& inputNames)
{
    CV_Assert(impl_);
    CV_Assert(inputs.size() == inputNames.size());
    for (size_t i = 0; i < inputs.size(); i++)
        impl_->net_.setInput(inputs[i], inputNames[i]);
    return impl_->net_.forward();
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
