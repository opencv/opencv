// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_VLM_LOCAL_MODEL_BASE_HPP
#define OPENCV_VLM_LOCAL_MODEL_BASE_HPP

#include "opencv2/dnn.hpp"
#include "vlm_model_base.hpp"

namespace cv { namespace vlm {

class LocalVLMModelBase : public VLMModelBase
{
public:
    String infer(InputArray image, const String& prompt, int max_new_tokens) CV_OVERRIDE;

    void reset() CV_OVERRIDE;
    void setPreferableDevice(const String& device) CV_OVERRIDE;
    int lastTokensUsed() const CV_OVERRIDE;

protected:
    void registerNets(dnn::Net& visionNet, dnn::Net& embedNet, dnn::Net& decoderNet);

    void setLastTokensUsed(int tokens);

    // Any engine-specific geometry the prompt needs (grid rows/cols, tile rows/cols, ...)
    // must be written to dimsOut so buildPrompt() below can consume it.
    virtual Mat runVisionEncoder(const Mat& imageBgr, Vec2i& dimsOut) = 0;

    virtual String buildPrompt(const Vec2i& dims, const String& userPrompt) const = 0;

    virtual String defaultPrompt() const = 0;

    dnn::Tokenizer tokenizer_;
    int imageTokenId_ = 0, eosTokenId_ = 2;

private:
    dnn::Net* visionNet_ = nullptr;
    dnn::Net* embedNet_ = nullptr;
    dnn::Net* decoderNet_ = nullptr;
    int lastTokensUsed_ = -1;
};

}} // namespace cv::vlm

#endif // OPENCV_VLM_LOCAL_MODEL_BASE_HPP
