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

namespace {

int engineFromString(const String& engine)
{
    if (engine == "opencv")
        return dnn::ENGINE_OPENCV;
    if (engine == "ort")
        return dnn::ENGINE_ORT;
    CV_Error(Error::StsBadArg, "vlm: unknown engine '" + engine + "' (expected 'opencv' or 'ort')");
}

} // namespace

Ptr<VLMModel> create(VLMModelType model_type, const String& model_dir,
                      const String& engine, const String& device, const String& api_key)
{
    switch (model_type)
    {
    case VLM_MODEL_PADDLEOCR_VL:
        return createPaddleOCRVLModel(model_dir, engineFromString(engine), device);
    case VLM_MODEL_GRANITE_DOCLING:
        return createGraniteDoclingModel(model_dir, engineFromString(engine), device);
    case VLM_MODEL_OPENAI:
    case VLM_MODEL_ANTHROPIC:
    case VLM_MODEL_GEMINI:
    case VLM_MODEL_GROK:
        return createCloudModel(model_type, model_dir, api_key);
    default:
        CV_Error(Error::StsBadArg, cv::format("vlm: unknown VLMModelType: %d", (int)model_type));
    }
}

}} // namespace cv::vlm
