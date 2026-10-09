// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_VLM_PADDLEOCR_VL_PREPROCESS_HPP
#define OPENCV_VLM_PADDLEOCR_VL_PREPROCESS_HPP

#include "opencv2/core.hpp"

namespace cv { namespace vlm {

void smartResize(int height, int width, int factor, int minPixels, int maxPixels,
                  int& outHeight, int& outWidth);

Mat preprocessImage(const Mat& imageBgr, int patchSize, int mergeSize, int minPixels,
                     int maxPixels, const Vec3f& mean, const Vec3f& std_, float rescaleFactor,
                     int& gridH, int& gridW);

String buildPaddleOCRVLPrompt(const String& prompt, int imageTokenRepeats);

}} // namespace cv::vlm

#endif // OPENCV_VLM_PADDLEOCR_VL_PREPROCESS_HPP
