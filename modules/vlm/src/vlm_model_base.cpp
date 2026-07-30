// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "vlm_model_base.hpp"

namespace cv { namespace vlm {

VLMModel::~VLMModel() {}

std::vector<String> VLMModelBase::inferDocument(const String& input_path, const String& prompt,
                                                 int max_new_tokens)
{
    Mat image = imread(input_path, IMREAD_COLOR);
    if (image.empty())
        CV_Error(Error::StsError, "vlm: could not read input file: " + input_path);

    reset();
    return { infer(image, prompt, max_new_tokens) };
}

}} // namespace cv::vlm
