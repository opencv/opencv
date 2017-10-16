// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include <opencv2/dnn/nms.inl.hpp>

namespace cv
{
namespace dnn
{

void NMSBoxes(const std::vector<Rect>& bboxes, const std::vector<float>& scores,
                          const float score_threshold, const float nms_threshold,
                          const float eta, const int top_k, std::vector<int>& indices)
{
    NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, NMSOverlap<Rect>());
}

}// dnn
}// cv
