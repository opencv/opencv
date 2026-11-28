// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_VLM_GENERATION_HPP
#define OPENCV_VLM_GENERATION_HPP

#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"

namespace cv { namespace vlm {

int argmaxLastToken(const Mat& logits);

void scatterImageFeatures(Mat& inputsEmbeds, const std::vector<int>& tokens,
                           int imageTokenId, const Mat& imageFeatures);

std::vector<int> generateWithKVCache(dnn::Net& embedNet, dnn::Net& decoderNet,
                                      const Mat& promptInputsEmbeds, int promptLen,
                                      int maxNewTokens, int eosTokenId);

}} // namespace cv::vlm

#endif // OPENCV_VLM_GENERATION_HPP
