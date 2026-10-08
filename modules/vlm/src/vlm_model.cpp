// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "engines/paddleocr_vl_engine.hpp"
#include "engines/granite_docling_engine.hpp"
#include "engines/cloud_engine.hpp"

namespace cv { namespace vlm {

VLMModel::~VLMModel() {}

namespace {

// The variable each provider's own SDK reads; an explicit api_key argument still wins.
const char* apiKeyEnvVar(VLMModelType model_type)
{
    switch (model_type)
    {
    case VLM_MODEL_OPENAI:    return "OPENAI_API_KEY";
    case VLM_MODEL_ANTHROPIC: return "ANTHROPIC_API_KEY";
    case VLM_MODEL_GEMINI:    return "GEMINI_API_KEY";
    case VLM_MODEL_GROK:      return "XAI_API_KEY";
    default:                  return nullptr;
    }
}

String resolveApiKey(VLMModelType model_type, const String& api_key)
{
    if (!api_key.empty())
        return api_key;

    const char* var = apiKeyEnvVar(model_type);
    CV_Assert(var != nullptr);
    String fromEnv = utils::getConfigurationParameterString(var, "");
    if (fromEnv.empty())
        CV_Error(Error::StsBadArg,
                 cv::format("vlm: no API key -- pass api_key to create(), or set %s", var));
    return fromEnv;
}

} // namespace

Ptr<VLMModel> create(VLMModelType model_type, const String& model_dir,
                      const String& api_key)
{
    switch (model_type)
    {
    case VLM_MODEL_PADDLEOCR_VL:
        return createPaddleOCRVLModel(model_dir);
    case VLM_MODEL_GRANITE_DOCLING:
        return createGraniteDoclingModel(model_dir);
    case VLM_MODEL_OPENAI:
    case VLM_MODEL_ANTHROPIC:
    case VLM_MODEL_GEMINI:
    case VLM_MODEL_GROK:
        return createCloudModel(model_type, model_dir, resolveApiKey(model_type, api_key));
    default:
        CV_Error(Error::StsBadArg, cv::format("vlm: unknown VLMModelType: %d", (int)model_type));
    }
}

String inferFile(const Ptr<VLMModel>& model, const String& path, const String& prompt,
                 int max_new_tokens)
{
    CV_Assert(model);

    Mat image = imread(path, IMREAD_COLOR);
    if (image.empty())
        CV_Error(Error::StsError, "vlm: could not read input file: " + path);

    model->reset();
    return model->infer(image, prompt, max_new_tokens);
}

}} // namespace cv::vlm
