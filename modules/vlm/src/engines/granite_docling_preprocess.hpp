// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_VLM_GRANITE_DOCLING_PREPROCESS_HPP
#define OPENCV_VLM_GRANITE_DOCLING_PREPROCESS_HPP

#include "opencv2/core.hpp"

namespace cv { namespace vlm {

void resizeAA(const Mat& src, Mat& dst, Size size);

Mat tileImage(const Mat& imageBgr, int longestEdge, int tileSize,
              const Vec3f& mean, const Vec3f& std_, int& rowsOut, int& colsOut);

String buildGraniteDoclingPrompt(int rows, int cols, int imageSeqLen, const String& userText);

}} // namespace cv::vlm

#endif // OPENCV_VLM_GRANITE_DOCLING_PREPROCESS_HPP
