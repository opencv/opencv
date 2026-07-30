// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_VLM_GRANITE_DOCLING_ENGINE_HPP
#define OPENCV_VLM_GRANITE_DOCLING_ENGINE_HPP

#include "opencv2/vlm.hpp"

namespace cv { namespace vlm {

Ptr<VLMModel> createGraniteDoclingModel(const String& model_dir, int engine, const String& device);

}} // namespace cv::vlm

#endif // OPENCV_VLM_GRANITE_DOCLING_ENGINE_HPP
